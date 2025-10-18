"""
Create a 3x3 frame grid image from a video, extract audio, and transcribe with Whisper.
"""

import argparse
import os
from pathlib import Path
import math
import numpy as np
from PIL import Image
from datetime import timedelta
import torch

# MoviePy is a convenient wrapp er over ffmpeg for frames + audio
from moviepy.editor import VideoFileClip

# Whisper (openai-whisper)
import whisper 

def nice_basename(p: Path) -> str:
    return p.stem.replace(" ", "_")


def format_ts(seconds: float) -> str:
    """Format seconds as H:MM:SS.mmm (for debug/logging)."""
    td = timedelta(seconds=seconds)
    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60
    return f"{hours:d}:{minutes:02d}:{secs:06.3f}"


def pick_uniform_times(duration: float, n: int) -> list[float]:
    """
    Pick n times uniformly across (0, duration), excluding exact endpoints to avoid black/blank frames.
    If duration is very small, gracefully clamp.
    """
    if duration <= 0:
        return [0.0] * n
    # Exclude first and last ~2% of the clip to avoid fades/blank frames
    start = duration * 0.02
    end = duration * 0.98
    if end <= start:
        start, end = 0.0, duration
    times = np.linspace(start, end, n).tolist()
    return times


def extract_frames_grid(
    clip: VideoFileClip,
    times: list[float],
    grid_rows: int = 3,
    grid_cols: int = 3,
    cell_width: int | None = None,
    cell_height: int | None = None,
    resample=Image.Resampling.LANCZOS,
) -> Image.Image:
    """
    Grab frames at specified times and compose into a grid image (grid_rows x grid_cols).
    Optionally resizes frames into (cell_width x cell_height). If not provided, it infers a good size.
    """
    assert grid_rows * grid_cols == len(times), "Grid size must match number of times/frames."

    # Get frames as numpy arrays, then to PIL
    frames_pil: list[Image.Image] = []
    for t in times:
        frame_np = clip.get_frame(t)  # (H, W, 3) uint8
        frames_pil.append(Image.fromarray(frame_np))

    # Infer cell size if unspecified: target ~ 320px width per cell, preserve aspect by scaling height
    if cell_width is None or cell_height is None:
        # Use the median frame size as a reference
        widths = [im.width for im in frames_pil]
        heights = [im.height for im in frames_pil]
        med_w = int(np.median(widths))
        med_h = int(np.median(heights))
        # ~320px cell width (adjust if video is tiny/huge)
        base_w = max(160, min(480, 320 if med_w >= 320 else med_w))
        scale = base_w / med_w
        cell_w = base_w
        cell_h = int(round(med_h * scale))
    else:
        cell_w, cell_h = int(cell_width), int(cell_height)

    # Resize all frames to the same cell size (letterboxing to preserve aspect nicely)
    resized: list[Image.Image] = []
    for im in frames_pil:
        # Fit inside (cell_w, cell_h) preserving aspect ratio; pad with black
        r = min(cell_w / im.width, cell_h / im.height)
        new_w = max(1, int(round(im.width * r)))
        new_h = max(1, int(round(im.height * r)))
        im_resized = im.resize((new_w, new_h), resample=resample)

        canvas = Image.new("RGB", (cell_w, cell_h), color=(0, 0, 0))
        offset = ((cell_w - new_w) // 2, (cell_h - new_h) // 2)
        canvas.paste(im_resized, offset)
        resized.append(canvas)

    # Compose grid
    grid_w = grid_cols * cell_w
    grid_h = grid_rows * cell_h
    grid = Image.new("RGB", (grid_w, grid_h), color=(0, 0, 0))

    idx = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            x = c * cell_w
            y = r * cell_h
            grid.paste(resized[idx], (x, y))
            idx += 1

    return grid


def extract_audio(clip: VideoFileClip, out_path: Path, bitrate="192k"):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pick codec based on extension
    ext = out_path.suffix.lower()
    if ext == ".mp3":
        codec = "libmp3lame"
    elif ext in [".wav"]:
        codec = "pcm_s16le"
    else:
        codec = "aac"

    clip.audio.write_audiofile(
        str(out_path),
        bitrate=bitrate,
        codec=codec,
        verbose=False,
        logger=None
    )
    return out_path


# def transcribe_whisper(
#     audio_path: Path,
#     model_name: str = "base",
#     device: str | None = None,
#     language: str | None = None,
#     make_srt: bool = True,
# ) -> tuple[str, list[dict]]:
#     """
#     Run Whisper transcription. Returns (plain_text, segments).
#     If make_srt=True, caller can write SRT from segments.
#     """
#     model = whisper.load_model(model_name, device=device)
#     # fp16 off on CPU by default
#     result = model.transcribe(str(audio_path), language=language)
#     text = result.get("text", "").strip()
#     segments = result.get("segments", []) or []
#     return text, segments

def transcribe_whisper(
    audio_path: "Path",
    model_name: str = "base",
    device: str | None = None,
    language: str | None = None,
    make_srt: bool = True,
) -> tuple[str, list[dict]]:

    # Decide device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try GPU, fallback to CPU on CUDA errors / arch mismatch
    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as e:
        msg = str(e).lower()
        if ("cuda" in msg and
            ("no kernel image" in msg or "cuda initialization" in msg or "driver" in msg)):
            print("ℹ️ CUDA issue detected; retrying on CPU.")
            # Optionally make sure CUDA is ignored for this process:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            model = whisper.load_model(model_name, device="cpu")
        else:
            raise

    # fp16 off on CPU by default; on older GPUs you can also disable fp16 explicitly:
    # (only relevant if you *do* run on GPU and it supports fp16 poorly)
    # model.model.half()  # whisper uses half on GPU internally; leave as-is for CPU

    result = model.transcribe(str(audio_path), language=language)
    text = result.get("text", "").strip()
    segments = result.get("segments", []) or []
    return text, segments


def write_srt(segments: list[dict], out_path: Path) -> None:
    """
    Write SRT from Whisper segments.
    """
    def to_srt_ts(t: float) -> str:
        # SRT expects , as milliseconds separator: HH:MM:SS,mmm
        td = timedelta(seconds=max(t, 0.0))
        total_ms = int(round(td.total_seconds() * 1000))
        ms = total_ms % 1000
        secs = (total_ms // 1000) % 60
        mins = (total_ms // 60000) % 60
        hours = total_ms // 3600000
        return f"{hours:02d}:{mins:02d}:{secs:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(segments, 1):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        txt = (seg.get("text") or "").strip()
        lines.append(f"{i}")
        lines.append(f"{to_srt_ts(start)} --> {to_srt_ts(end)}")
        lines.append(txt)
        lines.append("")  # blank line

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(
    video: str | Path,
    out_dir: str | Path = "outputs",
    frames: int = 9,
    grid_rows: int = 3,
    grid_cols: int = 3,
    cell_width: int | None = None,
    cell_height: int | None = None,
    grid_filename: str | None = None,
    audio_filename: str | None = None,
    audio_bitrate: str = "192k",
    whisper_model: str = "base",
    whisper_device: str | None = None,   # e.g. "cuda" or "cpu"
    language: str | None = None,         # e.g. "en", "hi"; None → autodetect
    write_srt: bool = True,
    save_grid: bool = True,
    return_grid: bool = True,
) -> dict:
    """
    Orchestrate: sample 9 uniformly spaced frames → build 3x3 grid → extract audio → transcribe with Whisper.

    Returns a dict with useful artifacts/paths for notebooks:
      {
        'duration': float,
        'times': list[float],
        'grid_img': PIL.Image or None,
        'grid_path': Path,
        'audio_path': Path,
        'transcript_text': str,
        'txt_path': Path,
        'srt_path': Path | None,
        'segments': list[dict]
      }
    """
    # Resolve paths
    in_path = Path(video).expanduser().resolve()
    assert in_path.exists(), f"Input not found: {in_path}"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base = nice_basename(in_path)

    # Validate grid size vs frames
    assert grid_rows * grid_cols == frames, \
        f"'frames' must equal grid_rows*grid_cols ({grid_rows}x{grid_cols} = {grid_rows*grid_cols})"

    # Open video
    clip = VideoFileClip(str(in_path))
    duration = float(clip.duration or 0.0)
    if duration <= 0:
        clip.close()
        raise ValueError("Could not read a valid duration from the video.")

    # 1) Pick times and build grid
    times = pick_uniform_times(duration, frames=9)
    grid_img = extract_frames_grid(
        clip=clip,
        times=times,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        cell_width=cell_width,
        cell_height=cell_height,
    )

    grid_name = grid_filename or f"{base}_grid_{grid_rows}x{grid_cols}.jpg"
    grid_path = out_dir / grid_name
    if save_grid:
        grid_img.save(grid_path, quality=95, subsampling=1)

    # 2) Extract audio (chooses codec by extension)
    audio_name = audio_filename or f"{base}.m4a"  # AAC is broadly supported
    audio_path = out_dir / audio_name
    extract_audio(clip, audio_path, bitrate=audio_bitrate)

    # 3) Transcribe with Whisper
    text, segments = transcribe_whisper(
        audio_path=audio_path,
        model_name=whisper_model,
        device=whisper_device,
        language=language,
        make_srt=write_srt
    )

    txt_path = out_dir / f"{base}.txt"
    txt_path.write_text((text or "").strip() + "\n", encoding="utf-8")

    srt_path = None
    if write_srt and segments:
        srt_path = out_dir / f"{base}.srt"
        write_srt(segments, srt_path)

    # Cleanup
    clip.close()

    return {
        "duration": duration,
        "times": times,
        "grid_img": grid_img if return_grid else None,
        "grid_path": grid_path,
        "audio_path": audio_path,
        "transcript_text": text,
        "txt_path": txt_path,
        "srt_path": srt_path,
        "segments": segments,
    }
 
if __name__ == "__main__":
    main()
