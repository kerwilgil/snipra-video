"""Reframe de un clip a un formato destino mediante filtros FFmpeg.

Modos:
- auto:   decide por aspecto del video fuente y formato destino.
- preserve: conserva encuadre completo y solo escala/rellena si hace falta.
- center: recorte centrado al aspecto destino (cover).
- blur:   video ajustado (contain) sobre fondo difuminado del mismo video.
- track:  seguimiento de sujeto (YOLO+MediaPipe). Requiere extra [vision];
          degrada a center si no está disponible.
- split:  2 personas, una arriba y otra abajo, cada una con su propio
          seguimiento. Solo por seleccion explicita (nunca via auto). Requiere
          extra [vision]; degrada a center si no está disponible o si el clip
          nunca tiene 2 caras a la vez.
"""

from app.core.config import settings

# resolución destino por formato
TARGETS: dict[str, tuple[int, int]] = {
    "9x16": (1080, 1920),
    "1x1": (1080, 1080),
    "16x9": (1920, 1080),
}

ASPECT_TOLERANCE = 0.04


def target_size(fmt: str) -> tuple[int, int]:
    return TARGETS.get(fmt, TARGETS["9x16"])


def _vision_available() -> bool:
    try:
        import importlib.util

        return (
            importlib.util.find_spec("ultralytics") is not None
            and importlib.util.find_spec("cv2") is not None
        )
    except ImportError:
        return False


def resolve_mode(requested: str | None = None) -> str:
    mode = (requested or settings.reframe_mode or "auto").lower()
    if mode in ("track", "split") and not settings.enable_tracking:
        return "center"
    return mode


def auto_reframe_mode(
    src_width: int,
    src_height: int,
    fmt: str,
    requested: str | None = None,
) -> str:
    """Elige el reframe efectivo por aspecto.

    Auto conserva videos ya compatibles con el destino. Si hay cambio fuerte de aspecto
    (p.ej. 16:9 -> 9:16) sigue al sujeto con tracking cuando el extra [vision] esta
    disponible; si no, cae a center crop. Para 9:16 no se usa blur/contain porque deja
    el horizontal dentro de un canvas vertical.
    """
    mode = resolve_mode(requested)
    if mode != "auto":
        return mode

    if src_width <= 0 or src_height <= 0:
        return "blur"

    dst_width, dst_height = target_size(fmt)
    src_aspect = src_width / src_height
    dst_aspect = dst_width / dst_height
    if abs(src_aspect - dst_aspect) <= ASPECT_TOLERANCE:
        return "preserve"
    if settings.enable_tracking:
        return "track"
    return "center" if fmt == "9x16" else "blur"


def build_filtergraph(fmt: str, mode: str, in_label: str = "0:v", out_label: str = "v") -> str:
    """Devuelve un filtergraph que toma [in_label] y produce [out_label] al tamaño destino."""
    w, h = target_size(fmt)
    mode = resolve_mode(mode)
    if fmt == "9x16" and mode == "blur":
        mode = "center"

    if mode == "preserve":
        if fmt == "9x16":
            return (
                f"[{in_label}]scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={w}:{h},setsar=1[{out_label}]"
            )
        return (
            f"[{in_label}]scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1[{out_label}]"
        )

    if mode == "blur":
        return (
            f"[{in_label}]scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={w}:{h},boxblur=20:2[bg];"
            f"[{in_label}]scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
            f"[bg][fg]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2,"
            f"setsar=1[{out_label}]"
        )

    # center (y fallback de track en F3)
    return (
        f"[{in_label}]scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={w}:{h},setsar=1[{out_label}]"
    )
