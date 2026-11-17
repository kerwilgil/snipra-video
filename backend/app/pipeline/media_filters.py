"""Composición de filtros multimedia para el render (media foundation 1.0).

Segmentos de filtergraph que se anteponen al reframe (tone map HDR,
transform de estabilización, denoise) y cadena de limpieza de audio que se
antepone al loudnorm FINAL. Todo se compone en el ÚNICO encode existente:
nunca decode→encode intermedios.

Capability-driven: cada segmento consulta al Capability Doctor y se omite
(o se reporta) cuando falta el filtro necesario.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.core.media import run
from app.core.media_capabilities import get_media_capabilities

# ---------------------------------------------------------------------------
# HDR / Dolby Vision → SDR
# ---------------------------------------------------------------------------


def hdr_tonemap_chain() -> str | None:
    """Pipeline de tone mapping HDR→SDR (zscale → tonemap → zscale → bt709).

    Objetivo SDR: BT.709, 8-bit, yuv420p. No es un retag: convierte la
    curva de transferencia real. Devuelve None si faltan capabilities
    (el llamador debe fallar con mensaje claro, nunca producir washed-out).
    """
    caps = get_media_capabilities()
    if not (
        caps["filters"].get("zscale") == "available"
        and caps["filters"].get("tonemap") == "available"
    ):
        return None
    return (
        "zscale=t=linear:npl=100,format=gbrpf32le,"
        "tonemap=hable:desat=0,"
        "zscale=t=bt709:p=bt709:m=bt709:r=tv,format=yuv420p"
    )


# ---------------------------------------------------------------------------
# Estabilización
# ---------------------------------------------------------------------------

# auto → medium cuando no existe métrica de movimiento confiable (Bloque F1).
STABILIZATION_PRESETS: dict[str, dict[str, int]] = {
    "light": {"shakiness": 3, "smoothing": 6},
    "medium": {"shakiness": 5, "smoothing": 12},
    "strong": {"shakiness": 8, "smoothing": 20},
}

STABILIZATION_TRANSFORM_NAME = "stab_transforms.trf"


def resolve_stabilization_mode(mode: str) -> str:
    """auto equivale a medium (sin métrica confiable no se inventa intensidad)."""
    if mode in STABILIZATION_PRESETS:
        return mode
    return "medium"


def stabilization_backend() -> str:
    """'vidstab' | 'deshake' | '' según capabilities reales."""
    caps = get_media_capabilities()
    return caps.get("backends", {}).get("stabilization", "")


def run_stabilization_analysis(
    base_name: str,
    work: Path,
    mode: str,
) -> Path:
    """Pase de análisis vidstabdetect sobre el clip base (sin reencode).

    Devuelve la ruta del archivo de transforms (dentro del temp del job).
    El llamador es responsable del cleanup tras completar o fallar.
    """
    preset = STABILIZATION_PRESETS[resolve_stabilization_mode(mode)]
    result = work / STABILIZATION_TRANSFORM_NAME
    if result.exists():
        result.unlink()
    cmd = [
        settings.ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        base_name,
        "-vf",
        (
            f"vidstabdetect=shakiness={preset['shakiness']}:accuracy=15:"
            f"stepsize=6:result={STABILIZATION_TRANSFORM_NAME}"
        ),
        "-an",
        "-f",
        "null",
        "-",
    ]
    run(cmd, timeout=1800, cwd=work)
    if not result.exists():
        raise RuntimeError("vidstabdetect no produjo el archivo de transforms")
    return result


def vidstab_transform_filter(mode: str) -> str:
    """Filtro vidstabtransform para el filtergraph final.

    optzoom=1: zoom adaptativo que compensa los bordes revelados por la
    corrección → sin barras negras (estrategia zoom/crop, coherente con
    allow_black_bars=False).
    """
    preset = STABILIZATION_PRESETS[resolve_stabilization_mode(mode)]
    return (
        f"vidstabtransform=input={STABILIZATION_TRANSFORM_NAME}:"
        f"smoothing={preset['smoothing']}:optzoom=1"
    )


def deshake_fallback_filter() -> str:
    """Fallback deshake (no equivalente a vidstab; queda documentado)."""
    return "deshake=rx=16:ry=16"


# ---------------------------------------------------------------------------
# Denoise de video
# ---------------------------------------------------------------------------

DENOISE_PRESETS: dict[str, str] = {
    "light": "hqdn3d=1.5:1.5:4:3",
    "medium": "hqdn3d=3:2:6:4.5",
    "strong": "hqdn3d=6:3:9:6",
}


def denoise_filter(mode: str) -> str | None:
    """Filtro de denoise para el preset pedido. None si falta la capability.

    hqdn3d es el filtro preferido (ampliamente disponible y rápido);
    nlmeans solo como alternativa cuando hqdn3d no existe.
    """
    caps = get_media_capabilities()
    filters = caps["filters"]
    if filters.get("hqdn3d") == "available":
        return DENOISE_PRESETS.get(mode, DENOISE_PRESETS["medium"])
    if filters.get("nlmeans") == "available":
        # nlmeans es mucho más lento: presets aún más conservadores.
        strengths = {"light": "s=2:p=3", "medium": "s=3:p=4", "strong": "s=4:p=5"}
        return f"nlmeans={strengths.get(mode, strengths['medium'])}:r=15"
    return None


# ---------------------------------------------------------------------------
# Limpieza de audio (antes del loudnorm FINAL)
# ---------------------------------------------------------------------------


def audio_cleanup_chain(mode: str) -> str | None:
    """Cadena de limpieza de voz para ANTES del loudnorm.

    Objetivos: rumble grave fuera, ruido constante reducido, dinámica
    controlada — sin separación IA de fuentes. Capability-driven: se omiten
    los filtros que no existan. None si el modo es off o no queda nada
    aplicable.
    """
    if mode not in ("voice_light", "voice", "voice_strong"):
        return None

    caps = get_media_capabilities()["filters"]

    def available(name: str) -> bool:
        return caps.get(name) == "available"

    filters: list[str] = []
    highpass_f = {"voice_light": 70, "voice": 80, "voice_strong": 90}[mode]
    if available("highpass"):
        filters.append(f"highpass=f={highpass_f}")

    if mode == "voice_strong" and available("lowpass"):
        filters.append("lowpass=f=12000")

    # Reducción de ruido espectral: afftdn preferente, anlmdn como alternativa.
    nr_params = {
        "voice_light": "afftdn=nr=10:nf=-30",
        "voice": "afftdn=nr=14:nf=-35",
        "voice_strong": "afftdn=nr=20:nf=-40",
    }[mode]
    if available("afftdn"):
        filters.append(nr_params)
    elif available("anlmdn"):
        strengths = {"voice_light": "s=10", "voice": "s=15", "voice_strong": "s=20"}
        filters.append(f"anlmdn={strengths[mode]}:p=3:m=3")

    if mode in ("voice", "voice_strong") and available("acompressor"):
        threshold = "-20dB" if mode == "voice" else "-22dB"
        ratio = "2.5" if mode == "voice" else "3"
        filters.append(
            f"acompressor=threshold={threshold}:ratio={ratio}:attack=10:release=150"
        )
    if mode == "voice_strong" and available("alimiter"):
        filters.append("alimiter=limit=0.95:level=false")

    if not filters:
        return None
    return ",".join(filters)


# ---------------------------------------------------------------------------
# Composición del pre-chain de video
# ---------------------------------------------------------------------------


def compose_video_prefilters(
    *,
    tonemap_needed: bool,
    stabilization_mode: str,
    denoise_mode: str,
    stab_transform_ready: bool,
) -> tuple[list[str], list[str]]:
    """Segmentos de filtro que se aplican a [0:v] ANTES del reframe.

    Orden (Bloque I1): tone map → stabilization transform → denoise →
    (reframe/crop/resize → sharpening → captions, ya existentes).

    Devuelve (filtros, notas). Las notas describen omisiones por
    capabilities faltantes para que el llamador pueda reportarlas.
    """
    filters: list[str] = []
    notes: list[str] = []

    if tonemap_needed:
        chain = hdr_tonemap_chain()
        if chain is None:
            raise RuntimeError(
                "HDR detectado pero este FFmpeg no dispone del pipeline "
                "requerido (zscale + tonemap) para convertirlo correctamente "
                "a SDR."
            )
        filters.append(chain)

    if stabilization_mode != "off":
        backend = stabilization_backend()
        if backend == "vidstab" and stab_transform_ready:
            filters.append(vidstab_transform_filter(stabilization_mode))
        elif backend == "deshake":
            filters.append(deshake_fallback_filter())
            notes.append(
                "Estabilización con fallback deshake (vidstab no disponible): "
                "resultado no equivalente a vidstab."
            )
        else:
            notes.append(
                "Estabilización omitida: este FFmpeg no tiene vidstab ni deshake."
            )

    if denoise_mode != "off":
        video_filter = denoise_filter(denoise_mode)
        if video_filter is None:
            notes.append("Denoise omitido: sin hqdn3d ni nlmeans.")
        else:
            filters.append(video_filter)

    return filters, notes


def loudnorm_chain(profile) -> str:
    """Loudnorm FINAL de audio con objetivo del platform profile."""
    return f"loudnorm=I={profile.target_lufs:g}:TP={profile.true_peak:g}:LRA=11"
