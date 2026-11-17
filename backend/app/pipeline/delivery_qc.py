"""Delivery QC: quality gate de salida estructurado (media foundation 1.0).

Verifica el archivo RENDERIZADO contra las reglas del platform profile.
No sustituye al quality gate existente de render: lo complementa con
checks de entrega (codec, color, loudness, faststart, safe zones).

Separación de severidades:
- FAIL: fallo MECÁNICO (resolución, codec, color, loudness roto…).
- WARN: aviso EDITORIAL o de plataforma (duración sobre el máximo,
  fps sobre el máximo, safe zones, VFR sospechado).

`passed` = sin ningún FAIL. Los WARN no bloquean el job pero se reportan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import settings
from app.core.media import VideoInfo, probe, run
from app.pipeline.reframe import target_size

PASS = "pass"
WARN = "warn"
FAIL = "fail"

_EBUR128_I_RE = re.compile(r"I:\s+(-?\d+(?:\.\d+)?)\s*LUFS")
_EBUR128_PEAK_RE = re.compile(r"Peak:\s+(-?\d+(?:\.\d+)?)\s*dBFS")
_ASS_PLAYRES_RE = re.compile(r"PlayRes[XY]:\s*(\d+)")
_ASS_MARGIN_V_RE = re.compile(r"^Style:.*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*\d+\s*$")


@dataclass
class QCCheck:
    id: str
    status: str
    actual: object = None
    expected: object = None
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "actual": self.actual,
            "expected": self.expected,
            "message": self.message,
        }


@dataclass
class DeliveryQCResult:
    passed: bool
    profile: str
    checks: list[QCCheck] = field(default_factory=list)
    source_hdr_format: str | None = None
    output_hdr_format: str | None = None

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "profile": self.profile,
            "source_hdr_format": self.source_hdr_format,
            "output_hdr_format": self.output_hdr_format,
            "checks": [check.to_dict() for check in self.checks],
        }

    @property
    def failures(self) -> list[QCCheck]:
        return [check for check in self.checks if check.status == FAIL]

    def summary_lines(self) -> list[str]:
        lines = []
        for check in self.checks:
            symbol = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL"}.get(check.status, "?")
            lines.append(f"{check.id}: {symbol}" + (f" — {check.message}" if check.message else ""))
        return lines


def measure_loudness(path: Path) -> tuple[float | None, float | None]:
    """(integrated LUFS, true peak dBFS) con ebur128. None si no medible."""
    cmd = [
        settings.ffmpeg_path,
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-af",
        "ebur128=peak=true",
        "-vn",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = run(cmd, timeout=300)
    except Exception:  # noqa: BLE001
        return None, None
    stderr = proc.stderr or ""
    integrated = _EBUR128_I_RE.findall(stderr)
    peaks = _EBUR128_PEAK_RE.findall(stderr)
    lufs = float(integrated[-1]) if integrated else None
    peak = float(peaks[-1]) if peaks else None
    return lufs, peak


def _mp4_has_faststart(path: Path) -> bool | None:
    """True si el átomo moov aparece antes que mdat. None si no es MP4."""
    try:
        with path.open("rb") as handle:
            header = handle.read(12)
            if len(header) < 8 or header[4:8] != b"ftyp":
                return None
            moov = mdat = -1
            offset = 0
            while moov == -1 or mdat == -1:
                handle.seek(offset)
                chunk = handle.read(4 * 1024 * 1024)
                if not chunk or len(chunk) <= 8:
                    break
                if moov == -1:
                    idx = chunk.find(b"moov")
                    if idx != -1:
                        moov = offset + idx
                if mdat == -1:
                    idx = chunk.find(b"mdat")
                    if idx != -1:
                        mdat = offset + idx
                if moov != -1 and mdat != -1:
                    break
                # Solapa 8 bytes para no partir un átomo entre chunks.
                offset += len(chunk) - 8
    except OSError:
        return None
    if moov == -1:
        return False
    if mdat == -1:
        return True
    return moov < mdat


def _subtitle_safe_zone_status(
    subs_path: Path | None,
    profile,
) -> QCCheck | None:
    """Verifica que los subtítulos respeten el margen seguro inferior."""
    if subs_path is None or not subs_path.exists():
        return None
    try:
        content = subs_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    playres = _ASS_PLAYRES_RE.findall(content)
    if len(playres) < 2:
        return None
    playres_y = int(playres[1])
    margin_v = None
    for line in content.splitlines():
        if line.startswith("Style:"):
            parts = line.split(",")
            if len(parts) >= 11:
                try:
                    # ..., MarginL, MarginR, MarginV, Encoding
                    margin_v = int(parts[-2])
                except ValueError:
                    margin_v = None
            break
    if margin_v is None or playres_y <= 0:
        return None
    required = profile.safe_bottom * playres_y
    if margin_v + 1 >= required:
        return QCCheck("safe_zone", PASS, margin_v, f">= {required:.0f}px")
    return QCCheck(
        "safe_zone",
        WARN,
        margin_v,
        f">= {required:.0f}px",
        (
            "Los subtítulos entran en el margen seguro inferior del destino "
            "(la UI de la plataforma puede taparlos)."
        ),
    )


def run_delivery_qc(
    path: Path,
    fmt: str,
    profile,
    *,
    expected_duration: float,
    expected_has_audio: bool,
    source_hdr_format: str | None = None,
    subs_path: Path | None = None,
    measure_audio: bool = True,
) -> DeliveryQCResult:
    """Ejecuta el delivery QC completo sobre un render ya producido."""
    checks: list[QCCheck] = []

    def add(check_id: str, status: str, actual, expected, message: str = "") -> None:
        checks.append(QCCheck(check_id, status, actual, expected, message))

    # 1) Archivo
    if not path.exists():
        add("file", FAIL, "missing", "exists", "El archivo renderizado no existe.")
        return DeliveryQCResult(
            passed=False,
            profile=profile.id,
            checks=checks,
            source_hdr_format=source_hdr_format,
        )
    size = path.stat().st_size
    add("file", PASS if size > 0 else FAIL, size, "> 0 bytes")

    info: VideoInfo = probe(path)

    # 2) Geometría (single source of truth: reframe.target_size para fmt)
    expected_w, expected_h = target_size(fmt)
    if (info.width, info.height) == (expected_w, expected_h):
        add("resolution", PASS, f"{info.width}x{info.height}", f"{expected_w}x{expected_h}")
    else:
        add(
            "resolution",
            FAIL,
            f"{info.width}x{info.height}",
            f"{expected_w}x{expected_h}",
            "Resolución de salida distinta a la del formato.",
        )
    expected_aspect = expected_w / expected_h
    actual_aspect = info.width / max(info.height, 1)
    if abs(actual_aspect - expected_aspect) <= 0.02 * expected_aspect:
        add("aspect", PASS, round(actual_aspect, 4), round(expected_aspect, 4))
    else:
        add("aspect", FAIL, round(actual_aspect, 4), round(expected_aspect, 4))

    # 3) Codec y pixel format
    codec = info.codec or "unknown"
    if codec in profile.allowed_codecs:
        add("codec", PASS, codec, "/".join(profile.allowed_codecs))
    else:
        add(
            "codec",
            FAIL,
            codec,
            "/".join(profile.allowed_codecs),
            f"Codec no aceptado por el destino {profile.display_name}.",
        )
    if info.pixel_format == profile.pixel_format:
        add("pixel_format", PASS, info.pixel_format, profile.pixel_format)
    else:
        add(
            "pixel_format",
            FAIL,
            info.pixel_format,
            profile.pixel_format,
            "Pixel format de salida inesperado.",
        )

    # 4) FPS (informativo/editorial: no re-encode por fps)
    fps = info.fps
    if fps is None:
        add("fps", WARN, None, f"<= {profile.fps_max:g}", "FPS no determinable.")
    elif fps <= profile.fps_max + profile.fps_tolerance:
        add("fps", PASS, round(fps, 3), f"<= {profile.fps_max:g}")
    else:
        add(
            "fps",
            WARN,
            round(fps, 3),
            f"<= {profile.fps_max:g}",
            f"FPS por encima del máximo recomendado de {profile.display_name}.",
        )

    # 5) VFR (conservador)
    if info.variable_frame_rate_suspected and not profile.allow_vfr:
        add(
            "vfr",
            WARN,
            True,
            False,
            "Frame rate variable sospechado en la salida.",
        )
    else:
        add("vfr", PASS, info.variable_frame_rate_suspected, False)

    # 6) Audio
    if expected_has_audio:
        if info.has_audio:
            if info.audio_codec == profile.audio_codec:
                add("audio_codec", PASS, info.audio_codec, profile.audio_codec)
            else:
                add(
                    "audio_codec",
                    FAIL,
                    info.audio_codec,
                    profile.audio_codec,
                    "Codec de audio distinto al del perfil de destino.",
                )
            if info.sample_rate and info.sample_rate != profile.sample_rate:
                add(
                    "sample_rate",
                    WARN,
                    info.sample_rate,
                    profile.sample_rate,
                    "Sample rate distinto al recomendado (reproducible, no bloqueante).",
                )
            else:
                add("sample_rate", PASS, info.sample_rate, profile.sample_rate)
            if info.channels and info.channels != profile.channels:
                add(
                    "channels",
                    WARN,
                    info.channels,
                    profile.channels,
                    "El audio no es estéreo como recomienda el destino.",
                )
            else:
                add("channels", PASS, info.channels, profile.channels)
        else:
            add("audio", FAIL, False, True, "Audio faltante en la salida.")
    else:
        add("audio", PASS, info.has_audio, False, "Sin audio esperado.")

    # 7) Loudness (integrado + true peak) solo con audio y clips no triviales
    if measure_audio and info.has_audio and info.duration >= 1.0:
        lufs, peak = measure_loudness(path)
        if lufs is None:
            add(
                "loudness",
                WARN,
                None,
                f"{profile.target_lufs:g} LUFS",
                "Loudness integrado no medible.",
            )
        else:
            deviation = abs(lufs - profile.target_lufs)
            if deviation <= profile.loudness_warn_lu:
                add("loudness", PASS, round(lufs, 1), f"{profile.target_lufs:g} LUFS")
            elif deviation <= profile.loudness_fail_lu:
                add(
                    "loudness",
                    WARN,
                    round(lufs, 1),
                    f"{profile.target_lufs:g} LUFS",
                    "Loudness fuera del objetivo (dentro de tolerancia amplia).",
                )
            else:
                add(
                    "loudness",
                    FAIL,
                    round(lufs, 1),
                    f"{profile.target_lufs:g} LUFS",
                    "Loudness integrado muy fuera del objetivo del destino.",
                )
        if peak is not None:
            overshoot = profile.true_peak - peak
            if overshoot >= -profile.true_peak_warn_db:
                add("true_peak", PASS, round(peak, 1), f"<= {profile.true_peak:g} dBFS")
            elif overshoot >= -profile.true_peak_fail_db:
                add(
                    "true_peak",
                    WARN,
                    round(peak, 1),
                    f"<= {profile.true_peak:g} dBFS",
                    "True peak por encima del objetivo (riesgo de limiting de plataforma).",
                )
            else:
                add(
                    "true_peak",
                    FAIL,
                    round(peak, 1),
                    f"<= {profile.true_peak:g} dBFS",
                    "True peak claramente por encima del límite del destino.",
                )
    elif info.has_audio:
        add(
            "loudness",
            PASS,
            None,
            f"{profile.target_lufs:g} LUFS",
            "Clip demasiado corto para medir.",
        )

    # 8) Duración
    duration_delta = abs(info.duration - expected_duration)
    max_duration_delta = max(0.75, expected_duration * 0.03)
    if duration_delta <= max_duration_delta:
        add("duration", PASS, round(info.duration, 2), round(expected_duration, 2))
    else:
        add(
            "duration",
            FAIL,
            round(info.duration, 2),
            round(expected_duration, 2),
            "Duración de la salida no coincide con la esperada.",
        )
    if profile.max_duration_seconds and info.duration > profile.max_duration_seconds:
        add(
            "duration_max",
            WARN,
            round(info.duration, 1),
            f"<= {profile.max_duration_seconds:g}s",
            f"El clip excede la duración máxima de {profile.display_name}.",
        )

    # 9) Color: HDR/SDR según profile
    output_hdr = bool(info.hdr)
    output_hdr_format = info.hdr_format if output_hdr else "sdr"
    if output_hdr and profile.prefer_sdr:
        add(
            "color",
            FAIL,
            output_hdr_format,
            "sdr",
            "Salida HDR en un destino que requiere SDR.",
        )
    elif output_hdr and not profile.allow_hdr:
        add(
            "color",
            FAIL,
            output_hdr_format,
            "sdr",
            f"{profile.display_name} no acepta HDR.",
        )
    elif output_hdr:
        add("color", PASS, output_hdr_format, "hdr permitido")
    else:
        if info.color_transfer in (None, "bt709") or not output_hdr:
            add("color", PASS, info.color_transfer or "sdr (sin tag)", "bt709/sdr")
        else:
            add(
                "color",
                WARN,
                info.color_transfer,
                "bt709/sdr",
                "Transferencia de color inusual para SDR.",
            )

    # 10) Contenedor: faststart para MP4 (streaming social)
    if path.suffix.lower() == ".mp4":
        faststart = _mp4_has_faststart(path)
        if faststart:
            add("faststart", PASS, True, True)
        else:
            add(
                "faststart",
                WARN,
                faststart,
                True,
                "moov no precede a mdat: la carga en streaming puede verse afectada.",
            )

    # 11) Safe zones (editorial: WARN)
    safe_zone = _subtitle_safe_zone_status(subs_path, profile)
    if safe_zone is not None:
        checks.append(safe_zone)

    passed = all(check.status != FAIL for check in checks)
    return DeliveryQCResult(
        passed=passed,
        profile=profile.id,
        checks=checks,
        source_hdr_format=source_hdr_format,
        output_hdr_format=output_hdr_format,
    )
