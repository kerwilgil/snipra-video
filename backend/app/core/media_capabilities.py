"""Capability Doctor: diagnóstico real de la instalación multimedia.

Inspecciona el FFmpeg/FFprobe reales (versiones, encoders, filters) y deriva
features a partir de capabilities comprobadas. Nunca marca MISSING cuando el
comando de introspección falló: eso es UNKNOWN.

Contrato:
    {
      "ffmpeg_version": str,
      "ffprobe_version": str,
      "overall_status": "ok" | "degraded" | "unknown",
      "encoders": {nombre: "available" | "missing" | "unknown"},
      "filters": {nombre: "available" | "missing" | "unknown"},
      "features": {
        "hdr_to_sdr", "stabilization", "video_denoise", "audio_cleanup",
        "subtitles", "social_export"
      : "available" | "missing" | "unknown" | "deshake_fallback"},
      "backends": {"stabilization": "vidstab" | "deshake" | "", "video_denoise": ...},
      "warnings": [str]
    }

El resultado se cachea por proceso (consultar es barato tras el primer uso);
`refresh=True` fuerza un re-diagnóstico completo.
"""

from __future__ import annotations

import re
import subprocess
import threading

from app.core.config import settings

AVAILABLE = "available"
MISSING = "missing"
UNKNOWN = "unknown"

# Encoders relevantes para el pipeline de clips de Snipra.
ENCODERS = [
    "libx264",
    "libx265",
    "aac",
    "prores_ks",
    "libsvtav1",
    "libaom-av1",
]

# Filtros relevantes (render, HDR, estabilización, denoise, audio, QC).
FILTERS = [
    "subtitles",
    "drawtext",
    "loudnorm",
    "silencedetect",
    "scdet",
    "scale",
    "crop",
    "overlay",
    "xfade",
    "zscale",
    "tonemap",
    "vidstabdetect",
    "vidstabtransform",
    "deshake",
    "hqdn3d",
    "nlmeans",
    "afftdn",
    "anlmdn",
    "highpass",
    "lowpass",
    "acompressor",
    "alimiter",
]

_LOCK = threading.Lock()
_CACHE: dict | None = None

_VERSION_RE = re.compile(r"version\s+(\S+)")
# Líneas de `ffmpeg -filters`: " $ TSC name A->A    Descripción"
_FILTER_LINE_RE = re.compile(r"^\s*\$?\s*[A-Z.]{2,8}\s+([a-z0-9_]+)\s")
# Líneas de `ffmpeg -encoders`: " V....D name      Descripción"
_ENCODER_LINE_RE = re.compile(r"^\s*[A-Z.]{4,8}\s+([a-zA-Z0-9_]+)\s")


def _run_introspection(cmd: list[str], timeout: int = 20) -> tuple[bool, str]:
    """Ejecuta un comando de introspección. Devuelve (exitoso, salida)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return False, ""
    if proc.returncode != 0:
        return False, (proc.stdout or "") + "\n" + (proc.stderr or "")
    return True, (proc.stdout or "") + "\n" + (proc.stderr or "")


def _parse_version(text: str) -> str:
    match = _VERSION_RE.search(text or "")
    return match.group(1) if match else ""


def _parse_filters_output(text: str) -> set[str] | None:
    """Nombres de filtros desde `ffmpeg -filters`. None si no se pudo parsear."""
    names: set[str] = set()
    for line in (text or "").splitlines():
        match = _FILTER_LINE_RE.match(line)
        if match:
            names.add(match.group(1))
    return names if names else None


def _parse_encoders_output(text: str) -> set[str] | None:
    """Nombres de encoders desde `ffmpeg -encoders`. None si no se pudo parsear."""
    names: set[str] = set()
    for line in (text or "").splitlines():
        match = _ENCODER_LINE_RE.match(line)
        if match:
            names.add(match.group(1))
    return names if names else None


def _status_for(name: str, parsed: set[str] | None) -> str:
    if parsed is None:
        return UNKNOWN
    return AVAILABLE if name in parsed else MISSING


def _build_capabilities() -> dict:
    warnings: list[str] = []

    ff_ok, ff_out = _run_introspection([settings.ffmpeg_path, "-version"])
    fp_ok, fp_out = _run_introspection([settings.ffprobe_path, "-version"])
    ffmpeg_version = _parse_version(ff_out) if ff_ok else ""
    ffprobe_version = _parse_version(fp_out) if fp_ok else ""

    if not ff_ok:
        warnings.append("No se pudo interrogar a FFmpeg: el diagnóstico es limitado.")
    if not fp_ok:
        warnings.append("No se pudo interrogar a FFprobe: el diagnóstico es limitado.")

    filters_ok, filters_text = (
        _run_introspection([settings.ffmpeg_path, "-hide_banner", "-filters"])
        if ff_ok
        else (False, "")
    )
    encoders_ok, encoders_text = (
        _run_introspection([settings.ffmpeg_path, "-hide_banner", "-encoders"])
        if ff_ok
        else (False, "")
    )
    filter_names = _parse_filters_output(filters_text) if filters_ok else None
    encoder_names = _parse_encoders_output(encoders_text) if encoders_ok else None
    if filters_ok and filter_names is None:
        warnings.append("No se pudo interpretar la lista de filtros de FFmpeg.")
    if encoders_ok and encoder_names is None:
        warnings.append("No se pudo interpretar la lista de encoders de FFmpeg.")

    encoders = {name: _status_for(name, encoder_names) for name in ENCODERS}
    filters = {name: _status_for(name, filter_names) for name in FILTERS}

    # Features derivadas de capabilities reales.
    def has_filter(name: str) -> bool:
        return filters.get(name) == AVAILABLE

    def has_encoder(name: str) -> bool:
        return encoders.get(name) == AVAILABLE

    backends: dict[str, str] = {}

    # HDR→SDR: zscale + tonemap (pipeline de tone mapping real).
    hdr_to_sdr = UNKNOWN
    if filter_names is not None:
        hdr_to_sdr = AVAILABLE if (has_filter("zscale") and has_filter("tonemap")) else MISSING
        if hdr_to_sdr == MISSING:
            warnings.append(
                "HDR→SDR no disponible: faltan los filtros zscale y/o tonemap."
            )

    # Estabilización: vidstab completo > fallback deshake documentado.
    stabilization = UNKNOWN
    if filter_names is not None:
        if has_filter("vidstabdetect") and has_filter("vidstabtransform"):
            stabilization = AVAILABLE
            backends["stabilization"] = "vidstab"
        elif has_filter("deshake"):
            stabilization = "deshake_fallback"
            backends["stabilization"] = "deshake"
            warnings.append(
                "Estabilización limitada: vidstab no está disponible, se usaría "
                "deshake como fallback (no es equivalente a vidstab)."
            )
        else:
            stabilization = MISSING
            warnings.append("Estabilización no disponible: sin vidstab ni deshake.")

    video_denoise = UNKNOWN
    if filter_names is not None:
        if has_filter("hqdn3d"):
            video_denoise = AVAILABLE
            backends["video_denoise"] = "hqdn3d"
        elif has_filter("nlmeans"):
            video_denoise = AVAILABLE
            backends["video_denoise"] = "nlmeans"
        else:
            video_denoise = MISSING
            warnings.append("Reducción de ruido no disponible: sin hqdn3d ni nlmeans.")

    audio_cleanup = UNKNOWN
    if filter_names is not None:
        noise_reduction = has_filter("afftdn") or has_filter("anlmdn")
        if has_filter("highpass") and noise_reduction:
            audio_cleanup = AVAILABLE
        else:
            audio_cleanup = MISSING
            warnings.append("Limpieza de audio no disponible: faltan filtros de voz.")

    subtitles = UNKNOWN
    if filter_names is not None:
        subtitles = AVAILABLE if has_filter("subtitles") else MISSING
        if subtitles == MISSING:
            warnings.append("Quemado de subtítulos no disponible: falta el filtro subtitles.")

    social_export = UNKNOWN
    if encoder_names is not None:
        social_export = (
            AVAILABLE if (has_encoder("libx264") and has_encoder("aac")) else MISSING
        )
        if social_export == MISSING:
            warnings.append("Export social no disponible: faltan libx264 y/o aac.")

    features = {
        "hdr_to_sdr": hdr_to_sdr,
        "stabilization": stabilization,
        "video_denoise": video_denoise,
        "audio_cleanup": audio_cleanup,
        "subtitles": subtitles,
        "social_export": social_export,
    }

    if not ff_ok or not fp_ok:
        overall = UNKNOWN
    elif all(status == AVAILABLE for status in features.values()):
        overall = "ok"
    else:
        overall = "degraded"

    return {
        "ffmpeg_version": ffmpeg_version,
        "ffprobe_version": ffprobe_version,
        "overall_status": overall,
        "encoders": encoders,
        "filters": filters,
        "features": features,
        "backends": backends,
        "warnings": warnings,
    }


def get_media_capabilities(refresh: bool = False) -> dict:
    """Diagnóstico de capabilities multimedia (cacheado por proceso)."""
    global _CACHE
    with _LOCK:
        if _CACHE is None or refresh:
            _CACHE = _build_capabilities()
        return _CACHE


def has_filter(name: str) -> bool:
    return get_media_capabilities()["filters"].get(name) == AVAILABLE


def has_encoder(name: str) -> bool:
    return get_media_capabilities()["encoders"].get(name) == AVAILABLE


def feature_status(name: str) -> str:
    return get_media_capabilities()["features"].get(name, UNKNOWN)
