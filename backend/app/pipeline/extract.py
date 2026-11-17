"""Corte preciso de un segmento del video fuente con FFmpeg."""

from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.core.media import probe, run

AUDIO_FADE_SECONDS = 0.03


def _audio_fade_filter(duration: float) -> str:
    fade = min(AUDIO_FADE_SECONDS, max(0.0, duration / 2))
    fade_out_start = max(0.0, duration - fade)
    return f"afade=t=in:st=0:d={fade:.3f},afade=t=out:st={fade_out_start:.3f}:d={fade:.3f}"


def extract_clip(
    src: str | Path,
    start: float,
    end: float,
    out: str | Path,
    pix_fmt: str = "yuv420p",
) -> Path:
    """Corta [start, end] re-encodificando para garantizar cortes exactos.

    `pix_fmt` permite preservar 10-bit (yuv420p10le) en fuentes HDR para que
    el tone map del render final tenga precisión; el comportamiento por
    defecto (SDR) sigue siendo yuv420p 8-bit como siempre.
    """
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end - start)
    has_audio = probe(src).has_audio
    cmd = [
        settings.ffmpeg_path,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        # Fuerza 8-bit 4:2:0 salvo petición explícita de mayor precisión:
        # sin esto, una fuente AV1/VP9 10-bit (cada vez mas comun en
        # YouTube/Facebook) puede producir un libx264 de salida en
        # yuv420p10le -- sigue siendo H.264 valido, pero WhatsApp y varios
        # reproductores lo rechazan igual que rechazarian AV1.
        "-pix_fmt",
        pix_fmt,
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    if has_audio:
        cmd[-7:-7] = ["-af", _audio_fade_filter(duration)]
    run(cmd, timeout=1800)
    return out_path
