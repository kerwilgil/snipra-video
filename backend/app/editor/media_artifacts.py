"""Lazy, project-scoped media artifacts for the Editor preview.

These helpers only invoke the existing allowlisted FFmpeg wrapper and never
accept paths or filter expressions from the browser or an agent.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.core.config import settings
from app.core.media import make_thumbnail, run
from app.editor.models import EditorAsset

THUMBNAIL_PROFILE = "thumb_v1"
WAVEFORM_PROFILE = "waveform_v1"
PROXY_PROFILE = "proxy_720p_v1"


def _artifact_dir(asset: EditorAsset) -> Path:
    path = settings.editor_dir / asset.project_id / "artifacts" / asset.id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_replace(temporary: Path, target: Path) -> Path:
    if not temporary.is_file():
        raise RuntimeError("FFmpeg did not create the requested artifact")
    os.replace(temporary, target)
    return target


def thumbnail(asset: EditorAsset, at_us: int = 0) -> Path:
    bounded = max(0, min(at_us, max(0, asset.duration_us - 1)))
    target = _artifact_dir(asset) / f"{THUMBNAIL_PROFILE}-{bounded}.jpg"
    if target.is_file():
        return target
    temporary = target.with_suffix(".partial.jpg")
    make_thumbnail(asset.original_path, temporary, bounded / 1_000_000)
    return _atomic_replace(temporary, target)


def waveform(asset: EditorAsset) -> Path | None:
    if not asset.has_audio:
        return None
    target = _artifact_dir(asset) / f"{WAVEFORM_PROFILE}.png"
    if target.is_file():
        return target
    temporary = target.with_suffix(".partial.png")
    run(
        [
            settings.ffmpeg_path,
            "-y",
            "-i",
            asset.original_path,
            "-filter_complex",
            "aformat=channel_layouts=mono,showwavespic=s=1200x120:colors=39d7e8",
            "-frames:v",
            "1",
            str(temporary),
        ],
        timeout=180,
    )
    return _atomic_replace(temporary, target)


def proxy(asset: EditorAsset) -> Path | None:
    """Create a conservative 720p proxy only when source dimensions justify it."""
    if asset.kind != "video" or max(asset.width, asset.height) <= 1280:
        return None
    target = _artifact_dir(asset) / f"{PROXY_PROFILE}.mp4"
    if target.is_file():
        return target
    temporary = target.with_suffix(".partial.mp4")
    run(
        [
            settings.ffmpeg_path,
            "-y",
            "-i",
            asset.original_path,
            "-vf",
            "scale=-2:720:force_original_aspect_ratio=decrease",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        timeout=1800,
    )
    return _atomic_replace(temporary, target)
