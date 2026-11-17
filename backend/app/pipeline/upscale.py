"""Super-resolución IA opcional con Real-ESRGAN (ncnn-vulkan).

Reconstruye detalle real en clips que se ven blandos por el upscale del crop
16:9→9:16. Usa el binario `realesrgan-ncnn-vulkan` (portable, GPU vía Vulkan:
NVIDIA/AMD/Intel/Apple) — no requiere torch ni CUDA instalados.

GATED y A PRUEBA DE FALLOS: si el binario no está o algo falla, `upscale_video`
devuelve None y el pipeline conserva el clip original. Nunca rompe el job.

Flujo: extrae frames → Real-ESRGAN sobre la carpeta → recompone con el audio
original y reescala de vuelta a la resolución original (nitidez reconstruida sin
inflar el peso del archivo).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from app.core.config import settings
from app.core.media import probe, run

logger = logging.getLogger(__name__)


def upscale_available() -> bool:
    """True si el binario Real-ESRGAN es invocable (PATH o ruta absoluta)."""
    path = settings.realesrgan_path
    return bool(shutil.which(path)) or Path(path).exists()


def upscale_video(in_path: str | Path, out_path: str | Path) -> Path | None:
    """Mejora la nitidez del video con Real-ESRGAN. None si no aplica o falla."""
    if not upscale_available():
        return None

    in_path = Path(in_path)
    out_path = Path(out_path)
    try:
        info = probe(in_path)
        if info.width <= 0 or info.height <= 0:
            return None

        with tempfile.TemporaryDirectory(prefix="snipra_sr_") as tmp:
            tmp_dir = Path(tmp)
            frames_in = tmp_dir / "in"
            frames_out = tmp_dir / "out"
            frames_in.mkdir()
            frames_out.mkdir()

            # 1) extraer frames PNG
            run(
                [
                    settings.ffmpeg_path,
                    "-y",
                    "-i",
                    str(in_path),
                    str(frames_in / "frame_%06d.png"),
                ],
                timeout=1800,
            )
            if not any(frames_in.iterdir()):
                return None

            # 2) Real-ESRGAN sobre la carpeta (GPU vía Vulkan)
            proc = subprocess.run(
                [
                    settings.realesrgan_path,
                    "-i",
                    str(frames_in),
                    "-o",
                    str(frames_out),
                    "-n",
                    settings.realesrgan_model,
                ],
                capture_output=True,
                timeout=3600,
            )
            if proc.returncode != 0 or not any(frames_out.iterdir()):
                logger.warning("Real-ESRGAN no produjo frames; se conserva el original.")
                return None

            # 3) recomponer: frames mejorados + audio original, de vuelta a WxH original
            fps = _fps(in_path)
            run(
                [
                    settings.ffmpeg_path,
                    "-y",
                    "-framerate",
                    fps,
                    "-i",
                    str(frames_out / "frame_%06d.png"),
                    "-i",
                    str(in_path),
                    "-map",
                    "0:v",
                    "-map",
                    "1:a?",
                    "-vf",
                    f"scale={info.width}:{info.height}:flags=lanczos",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(out_path),
                ],
                timeout=1800,
            )
        return out_path if out_path.exists() else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Super-resolución falló (%s). Se conserva el clip original.", exc)
        return None


def _fps(path: Path) -> str:
    """Frame rate del video como string 'num/den' (default 30 si falla)."""
    try:
        proc = subprocess.run(
            [
                settings.ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        value = (proc.stdout or "").strip()
        return value if "/" in value and not value.startswith("0") else "30"
    except (OSError, subprocess.SubprocessError):
        return "30"
