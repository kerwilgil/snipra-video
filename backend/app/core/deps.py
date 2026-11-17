"""Verificación de dependencias del sistema (FFmpeg/FFprobe)."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from app.core.config import settings


@dataclass
class BinaryStatus:
    name: str
    path: str
    available: bool
    version: str = ""


def _check(name: str, configured: str) -> BinaryStatus:
    resolved = shutil.which(configured) or (configured if configured != name else None)
    if not resolved:
        return BinaryStatus(name=name, path=configured, available=False)
    try:
        out = subprocess.run(
            [resolved, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = out.stdout.splitlines()[0] if out.stdout else ""
        return BinaryStatus(name=name, path=resolved, available=True, version=version)
    except (OSError, subprocess.SubprocessError):
        return BinaryStatus(name=name, path=resolved, available=False)


def check_ffmpeg() -> dict[str, BinaryStatus]:
    return {
        "ffmpeg": _check("ffmpeg", settings.ffmpeg_path),
        "ffprobe": _check("ffprobe", settings.ffprobe_path),
    }


def require_ffmpeg() -> None:
    status = check_ffmpeg()
    missing = [name for name, s in status.items() if not s.available]
    if missing:
        raise RuntimeError(
            f"Faltan binarios requeridos: {', '.join(missing)}. "
            "Instala FFmpeg (winget install Gyan.FFmpeg) y reinicia la terminal."
        )
