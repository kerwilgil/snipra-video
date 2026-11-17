"""Utilidades de FFmpeg/ffprobe (sondeo y ejecución de comandos)."""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings


class FFmpegError(RuntimeError):
    pass


class FFmpegCanceledError(FFmpegError):
    pass


_cancel_check: ContextVar[object | None] = ContextVar("media_cancel_check", default=None)


def set_cancel_check(check) -> Token:
    return _cancel_check.set(check)


def reset_cancel_check(token: Token) -> None:
    _cancel_check.reset(token)


def _cancel_requested() -> bool:
    check = _cancel_check.get()
    return bool(check and check())


def _run_environment() -> dict[str, str] | None:
    """Env para FFmpeg.

    En Windows, ffmpeg con filtros que usan fontconfig (drawtext, subtitles,
    etc.) CRASHEA (exit 3221225477) si no puede localizar `fonts.conf`. BtbN
    no lo instala. Creamos uno mínimo en storage si no hay FONTCONFIG_FILE.
    Solo tiene efecto en NT y SOLO si el usuario no ha configurado uno propio.
    """
    import os

    env = dict(os.environ)
    if os.name == "nt" and not env.get("FONTCONFIG_FILE"):
        fonts_conf = settings.storage_dir / "fonts.conf"
        if not fonts_conf.exists():
            fonts_conf.write_text(
                '<?xml version="1.0"?>\n'
                '<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n'
                "<fontconfig>\n"
                "  <dir>C:\\Windows\\Fonts</dir>\n"
                "  <config></config>\n"
                "</fontconfig>\n",
                encoding="utf-8",
            )
        env["FONTCONFIG_FILE"] = str(fonts_conf)
    return env


def run(
    cmd: list[str], timeout: int | None = None, cwd: str | Path | None = None
) -> subprocess.CompletedProcess:
    """Ejecuta un comando y lanza FFmpegError si falla."""
    if _cancel_requested():
        raise FFmpegCanceledError("Job cancelado antes de iniciar FFmpeg")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=_run_environment(),
    )
    started = time.monotonic()
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            if _cancel_requested():
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                raise FFmpegCanceledError("Job cancelado durante FFmpeg") from None
            if timeout is not None and time.monotonic() - started > timeout:
                process.kill()
                process.communicate()
                raise subprocess.TimeoutExpired(cmd, timeout) from None
    proc = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-8:]
        head = " ".join(cmd[:3])
        raise FFmpegError(f"Comando falló ({proc.returncode}): {head}…\n" + "\n".join(tail))
    return proc


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int
    has_audio: bool
    # --- Metadatos extendidos (media foundation 1.0) ---
    # Todos con default para no romper consumidores existentes.
    fps: float | None = None
    fps_num: int | None = None
    fps_den: int | None = None
    # True = VFR sospechado (avg vs r divergen), False = CFR, None = desconocido.
    variable_frame_rate_suspected: bool | None = None
    codec: str | None = None
    codec_profile: str | None = None
    pixel_format: str | None = None
    bit_depth: int | None = None
    # Rotación declarada por el contenedor (tag rotate o display matrix, grados).
    # FFmpeg la aplica automáticamente al decodificar: es informativa, no hay
    # que duplicarla con filtros de transposición.
    rotation: int | None = None
    color_range: str | None = None
    color_space: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None
    # True = HDR real (PQ/HLG confirmado por transfer), False = SDR,
    # None = no determinable. BT.2020 con transfer no-HDR NO es HDR:
    # es wide gamut SDR.
    hdr: bool | None = None
    hdr_format: str | None = None
    dolby_vision: bool | None = None
    dolby_vision_profile: int | None = None
    video_bitrate: int | None = None
    audio_stream_count: int = 0
    audio_codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    channel_layout: str | None = None
    audio_bitrate: int | None = None


# Transfer characteristics que implican HDR real (no basta con BT.2020).
_HD_TRANSFERS = {
    "smpte2084": "pq",  # SMPTE ST 2084 (PQ)
    "arib-std-b67": "hlg",  # ARIB STD-B67 (HLG)
}

_BIT_DEPTH_RE = re.compile(r"(\d{2})(?:le|be)?$")
_8BIT_PREFIXES = ("yuv", "gray", "nv", "rgb", "bgr", "gbr")


def _parse_rate(value: object) -> tuple[int, int] | None:
    """Convierte '30000/1001' en (30000, 1001). None si es 0/0 o inválido."""
    if not isinstance(value, str) or "/" not in value:
        return None
    try:
        num, den = value.split("/", 1)
        num_i, den_i = int(num), int(den)
    except ValueError:
        return None
    if num_i <= 0 or den_i <= 0:
        return None
    return num_i, den_i


def _rotation_from_stream(stream: dict) -> int | None:
    """Rotación desde side_data display matrix o tag rotate (grados)."""
    for side in stream.get("side_data_list") or []:
        rotation = side.get("rotation")
        if rotation is not None:
            try:
                return int(rotation)
            except (TypeError, ValueError):
                continue
    raw = (stream.get("tags") or {}).get("rotate")
    if raw is not None:
        try:
            return int(raw) % 360
        except (TypeError, ValueError):
            return None
    return None


def _hdr_from_stream(stream: dict) -> tuple[bool | None, str | None]:
    """(hdr, hdr_format) a partir de color_transfer. Distingue wide-gamut SDR."""
    transfer = str(stream.get("color_transfer") or "").strip().lower()
    if not transfer:
        return None, None
    hdr_format = _HD_TRANSFERS.get(transfer)
    if hdr_format:
        return True, hdr_format
    return False, "sdr"


def _dolby_vision_from_stream(stream: dict) -> tuple[bool, int | None]:
    """(dolby_vision, dv_profile) desde side data DOVI. Solo si ffprobe lo expone."""
    for side in stream.get("side_data_list") or []:
        side_type = str(side.get("side_data_type") or "").lower()
        if "dolby vision" not in side_type and "dovi" not in side_type:
            continue
        profile = side.get("dv_profile")
        try:
            return True, int(profile) if profile is not None else None
        except (TypeError, ValueError):
            return True, None
    return False, None


def _bit_depth_from_stream(stream: dict) -> int | None:
    raw = stream.get("bits_per_raw_sample")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    pix_fmt = str(stream.get("pix_fmt") or "")
    match = _BIT_DEPTH_RE.search(pix_fmt)
    if match:
        return int(match.group(1))
    if pix_fmt.startswith(_8BIT_PREFIXES):
        return 8
    return None


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def parse_probe_json(data: dict) -> VideoInfo:
    """Construye VideoInfo desde el JSON de ffprobe (-show_format -show_streams).

    Función pura (sin subprocess) para poder testear el parser con fixtures.
    """
    duration = 0.0
    fmt = data.get("format") or {}
    if fmt.get("duration"):
        try:
            duration = float(fmt["duration"])
        except (TypeError, ValueError):
            duration = 0.0

    video: dict | None = None
    audio_streams: list[dict] = []
    for stream in data.get("streams") or []:
        codec_type = stream.get("codec_type")
        if codec_type == "video" and video is None:
            video = stream
        elif codec_type == "audio":
            audio_streams.append(stream)

    width = height = 0
    fps = fps_num = fps_den = None
    vfr: bool | None = None
    if video is not None:
        width = int(video.get("width", 0) or 0)
        height = int(video.get("height", 0) or 0)
        if not duration and video.get("duration"):
            with contextlib.suppress(TypeError, ValueError):
                duration = float(video["duration"])
        r_rate = _parse_rate(video.get("r_frame_rate"))
        avg_rate = _parse_rate(video.get("avg_frame_rate"))
        if r_rate and avg_rate:
            fps_num, fps_den = avg_rate
            fps = fps_num / fps_den
            # VFR conservador: solo si r y avg son ambos determinables y
            # divergen más de un 1%. Nunca inventar VFR.
            r_fps = r_rate[0] / r_rate[1]
            vfr = abs(fps - r_fps) / max(fps, 1e-9) > 0.01
        elif r_rate or avg_rate:
            fps_num, fps_den = r_rate or avg_rate  # type: ignore[assignment]
            fps = fps_num / fps_den  # type: ignore[operator]

    hdr, hdr_format = _hdr_from_stream(video or {})
    dolby_vision, dv_profile = _dolby_vision_from_stream(video or {})

    audio = audio_streams[0] if audio_streams else None

    return VideoInfo(
        duration=duration,
        width=width,
        height=height,
        has_audio=bool(audio_streams),
        fps=round(fps, 3) if fps is not None else None,
        fps_num=fps_num,
        fps_den=fps_den,
        variable_frame_rate_suspected=vfr,
        codec=str(video.get("codec_name")) if video else None,
        codec_profile=str(video.get("profile")) if video else None,
        pixel_format=str(video.get("pix_fmt")) if video else None,
        bit_depth=_bit_depth_from_stream(video) if video else None,
        rotation=_rotation_from_stream(video) if video else None,
        color_range=str(video.get("color_range")) if video else None,
        color_space=str(video.get("color_space")) if video else None,
        color_transfer=str(video.get("color_transfer")) if video else None,
        color_primaries=str(video.get("color_primaries")) if video else None,
        hdr=hdr,
        hdr_format=hdr_format,
        dolby_vision=dolby_vision if video else None,
        dolby_vision_profile=dv_profile,
        video_bitrate=_int_or_none((video or {}).get("bit_rate")),
        audio_stream_count=len(audio_streams),
        audio_codec=str(audio.get("codec_name")) if audio else None,
        sample_rate=_int_or_none(audio.get("sample_rate")) if audio else None,
        channels=_int_or_none(audio.get("channels")) if audio else None,
        channel_layout=str(audio.get("channel_layout")) if audio else None,
        audio_bitrate=_int_or_none(audio.get("bit_rate")) if audio else None,
    )


def probe(path: str | Path) -> VideoInfo:
    """Sondeo completo del medio con ffprobe (format + streams + side data)."""
    cmd = [
        settings.ffprobe_path,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = run(cmd, timeout=60)
    data = json.loads(proc.stdout or "{}")
    return parse_probe_json(data)


def extract_audio(src: str | Path, out_wav: str | Path) -> Path:
    """Extrae audio a WAV mono 16 kHz (óptimo para Whisper)."""
    out = Path(out_wav)
    cmd = [
        settings.ffmpeg_path,
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out),
    ]
    run(cmd, timeout=1800)
    return out


def extract_audio_segment(
    src: str | Path,
    out_wav: str | Path,
    *,
    start: float,
    end: float,
) -> Path:
    """Extrae un tramo de audio a WAV mono 16 kHz preservando offsets fuera del archivo."""
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end - start)
    cmd = [
        settings.ffmpeg_path,
        "-y",
        "-ss",
        f"{max(0.0, start):.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out),
    ]
    run(cmd, timeout=1800)
    return out


def make_thumbnail(src: str | Path, out_jpg: str | Path, at_seconds: float = 0.0) -> Path:
    """Genera un thumbnail JPG en el segundo indicado."""
    out = Path(out_jpg)
    cmd = [
        settings.ffmpeg_path,
        "-y",
        "-ss",
        f"{max(0.0, at_seconds):.3f}",
        "-i",
        str(src),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(out),
    ]
    run(cmd, timeout=120)
    return out
