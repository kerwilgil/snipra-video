"""Compositor: une varios ClipAsset ya renderizados en un video nuevo.

No pasa por el pipeline de clips (ingest/transcribe/detect): toma los mp4 que ya
existen en la biblioteca y los concatena, con transicion y musica de fondo
opcionales. Un solo comando de FFmpeg hace todo el trabajo.
"""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.core.media import VideoInfo, make_thumbnail, probe, run

# Los assets pueden venir con fps distinto segun la fuente original, y tanto
# concat como xfade exigen que todos los inputs coincidan. Normalizamos a 30.
TARGET_FPS = 30
TARGET_SAMPLE_RATE = 48000
_AUDIO_FORMAT = "aformat=sample_fmts=fltp:channel_layouts=stereo"

# Mismo perfil de encode que render.py::_encode_format, para que una compilacion
# no se vea distinta a los clips que la componen. -pix_fmt yuv420p: sin esto una
# fuente AV1/VP9 10-bit puede colar un libx264 en yuv420p10le, que WhatsApp y
# varios reproductores rechazan igual que rechazarian AV1.
_VIDEO_CODEC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
_AUDIO_CODEC = ["-c:a", "aac", "-b:a", "128k"]

TRANSITIONS = {"none", "fade"}


class CompileError(RuntimeError):
    pass


@dataclass
class CompiledOutput:
    path: Path
    thumb_path: Path | None
    duration: float
    width: int
    height: int
    size_bytes: int
    audio_report: dict


def compilation_dir(compilation_id: str) -> Path:
    return settings.compositions_dir / compilation_id


def normalize_transition(value: str | None) -> str:
    candidate = (value or "none").strip().lower()
    return candidate if candidate in TRANSITIONS else "none"


def _safe_transition_seconds(seconds: float, infos: list[VideoInfo]) -> float:
    """xfade falla si la transicion dura mas que el clip mas corto.

    La acotamos a la mitad del clip mas corto para que siempre quede material
    a ambos lados del cruce.
    """
    shortest = min((info.duration for info in infos if info.duration > 0), default=0.0)
    if shortest <= 0:
        return 0.0
    return max(0.1, min(max(0.0, seconds), shortest / 2))


def _normalized_inputs(infos: list[VideoInfo], width: int, height: int) -> list[str]:
    """Deja cada input en el mismo tamaño, fps y formato de audio."""
    parts: list[str] = []
    for index, info in enumerate(infos):
        parts.append(
            f"[{index}:v]fps={TARGET_FPS},"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{index}]"
        )
        if info.has_audio:
            parts.append(f"[{index}:a]aresample={TARGET_SAMPLE_RATE},{_AUDIO_FORMAT}[a{index}]")
        else:
            # Con concat/xfade en modo a=1 un input sin pista de audio rompe el
            # graph entero: lo rellenamos con silencio del largo del clip.
            parts.append(
                f"anullsrc=channel_layout=stereo:sample_rate={TARGET_SAMPLE_RATE},"
                f"atrim=duration={max(0.1, info.duration):.3f}[a{index}]"
            )
    return parts


def _join_clips(
    infos: list[VideoInfo], transition: str, transition_seconds: float
) -> tuple[list[str], str, str]:
    """Une los inputs ya normalizados. Devuelve (filtros, label video, label audio)."""
    count = len(infos)
    if count == 1:
        return [], "v0", "a0"

    if transition == "fade":
        seconds = _safe_transition_seconds(transition_seconds, infos)
        if seconds > 0:
            parts: list[str] = []
            video_label, audio_label = "v0", "a0"
            # xfade pide el offset absoluto donde empieza el cruce. Cada fusion
            # solapa `seconds`, asi que el largo acumulado no es la suma simple.
            elapsed = infos[0].duration
            for index in range(1, count):
                offset = max(0.0, elapsed - seconds)
                parts.append(
                    f"[{video_label}][v{index}]xfade=transition=fade:"
                    f"duration={seconds:.3f}:offset={offset:.3f}[vx{index}]"
                )
                parts.append(f"[{audio_label}][a{index}]acrossfade=d={seconds:.3f}[ax{index}]")
                video_label, audio_label = f"vx{index}", f"ax{index}"
                elapsed = elapsed + infos[index].duration - seconds
            return parts, video_label, audio_label

    streams = "".join(f"[v{index}][a{index}]" for index in range(count))
    return [f"{streams}concat=n={count}:v=1:a=1[vcat][acat]"], "vcat", "acat"


def _mix_bgm(
    bgm_index: int, audio_label: str, volume: float, ducking: bool
) -> tuple[list[str], str]:
    parts = [
        f"[{bgm_index}:a]volume={max(0.0, min(1.0, volume)):.3f},"
        f"aresample={TARGET_SAMPLE_RATE},{_AUDIO_FORMAT}[bgmraw]"
    ]
    # normalize=0 es clave: por defecto amix divide cada entrada entre n y la voz
    # se iria a la mitad de volumen. Con normalize=0 la voz queda a ganancia 1 y
    # la musica solo al nivel que fijo el filtro volume.
    if ducking:
        # sidechaincompress usa la voz como disparador para bajar la musica.
        # asplit porque un mismo label no puede consumirse dos veces en el graph.
        parts.append(f"[{audio_label}]asplit=2[avoice][atrigger]")
        parts.append(
            "[bgmraw][atrigger]sidechaincompress="
            "threshold=0.05:ratio=8:attack=5:release=250[bgmduck]"
        )
        parts.append("[avoice][bgmduck]amix=inputs=2:duration=first:normalize=0[aout]")
    else:
        parts.append(f"[{audio_label}][bgmraw]amix=inputs=2:duration=first:normalize=0[aout]")
    return parts, "aout"


def integrated_lufs(path: str | Path) -> float | None:
    """Mide LUFS integrados con FFmpeg/EBU R128; None si no hay lectura fiable."""
    cmd = [
        settings.ffmpeg_path,
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-filter_complex",
        "ebur128=framelog=verbose",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    matches = re.findall(r"\bI:\s*(-?\d+(?:\.\d+)?)\s+LUFS", proc.stderr or "")
    if not matches:
        return None
    value = float(matches[-1])
    return value if math.isfinite(value) and value > -70 else None


def automatic_bgm_volume(
    voice_paths: list[str | Path],
    bgm_path: str | Path,
    duck_lu: float = 12.0,
) -> tuple[float, dict]:
    """Calcula ganancia musical relativa a la voz usando sonoridad integrada."""
    measurements: list[tuple[float, float]] = []
    for path in voice_paths:
        lufs = integrated_lufs(path)
        if lufs is None:
            continue
        duration = max(0.1, probe(path).duration)
        measurements.append((lufs, duration))
    bgm_lufs = integrated_lufs(bgm_path)
    if not measurements or bgm_lufs is None:
        return 0.18, {
            "mode": "fallback",
            "reason": "No fue posible medir LUFS en todas las fuentes",
            "applied_volume": 0.18,
        }
    total_duration = sum(duration for _, duration in measurements)
    energy = sum((10 ** (lufs / 10)) * duration for lufs, duration in measurements)
    voice_lufs = 10 * math.log10(energy / total_duration)
    target_bgm_lufs = voice_lufs - max(6.0, min(24.0, duck_lu))
    gain_db = target_bgm_lufs - bgm_lufs
    volume = max(0.01, min(1.0, 10 ** (gain_db / 20)))
    return volume, {
        "mode": "automatic_lufs",
        "voice_lufs": round(voice_lufs, 2),
        "bgm_lufs": round(bgm_lufs, 2),
        "target_bgm_lufs": round(target_bgm_lufs, 2),
        "gain_db": round(20 * math.log10(volume), 2),
        "applied_volume": round(volume, 4),
        "duck_lu": round(max(6.0, min(24.0, duck_lu)), 1),
    }


def build_filtergraph(
    infos: list[VideoInfo],
    width: int,
    height: int,
    *,
    transition: str = "none",
    transition_seconds: float = 0.4,
    bgm_index: int | None = None,
    bgm_volume: float = 0.18,
    bgm_ducking: bool = True,
) -> tuple[str, str, str]:
    """Arma el filter_complex completo. Devuelve (graph, map video, map audio)."""
    parts = _normalized_inputs(infos, width, height)
    join_parts, video_label, audio_label = _join_clips(infos, transition, transition_seconds)
    parts += join_parts
    if bgm_index is not None:
        bgm_parts, audio_label = _mix_bgm(bgm_index, audio_label, bgm_volume, bgm_ducking)
        parts += bgm_parts
    return ";".join(parts), f"[{video_label}]", f"[{audio_label}]"


def render_compilation(
    compilation_id: str,
    asset_paths: list[str | Path],
    *,
    transition: str = "none",
    transition_seconds: float = 0.4,
    bgm_path: str | Path | None = None,
    bgm_volume: float = 0.18,
    bgm_ducking: bool = True,
    bgm_auto_level: bool = True,
    bgm_duck_lu: float = 12.0,
) -> CompiledOutput:
    paths = [Path(p) for p in asset_paths]
    if not paths:
        raise CompileError("La compilacion no tiene clips")
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise CompileError("Faltan archivos de clip (¿se borro el job?): " + ", ".join(missing[:3]))

    infos = [probe(path) for path in paths]
    width = next((info.width for info in infos if info.width), 0)
    height = next((info.height for info in infos if info.height), 0)
    if not width or not height:
        raise CompileError("No se pudo determinar la resolucion de los clips")

    out_dir = compilation_dir(compilation_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "output.mp4"

    cmd = [settings.ffmpeg_path, "-y"]
    for path in paths:
        cmd += ["-i", str(path)]

    bgm_index: int | None = None
    audio_report: dict = {"mode": "no_music"}
    if bgm_path:
        bgm = Path(bgm_path)
        if not bgm.exists():
            raise CompileError(f"No existe el audio de fondo: {bgm.name}")
        bgm_index = len(paths)
        if bgm_auto_level:
            bgm_volume, audio_report = automatic_bgm_volume(paths, bgm, bgm_duck_lu)
        else:
            audio_report = {
                "mode": "manual",
                "applied_volume": round(max(0.0, min(1.0, bgm_volume)), 4),
            }
        # -stream_loop -1 repite la musica indefinidamente; el amix con
        # duration=first la recorta al largo del video, sin dejar cola.
        cmd += ["-stream_loop", "-1", "-i", str(bgm)]

    graph, video_map, audio_map = build_filtergraph(
        infos,
        width,
        height,
        transition=normalize_transition(transition),
        transition_seconds=transition_seconds,
        bgm_index=bgm_index,
        bgm_volume=bgm_volume,
        bgm_ducking=bgm_ducking,
    )
    cmd += ["-filter_complex", graph, "-map", video_map, "-map", audio_map]
    cmd += _VIDEO_CODEC + _AUDIO_CODEC + ["-movflags", "+faststart", str(out_path)]
    run(cmd, timeout=3600)

    if not out_path.exists():
        raise CompileError("FFmpeg no produjo el archivo de la compilacion")

    final = probe(out_path)
    thumb: Path | None = out_dir / "thumb.jpg"
    try:
        make_thumbnail(out_path, thumb, at_seconds=min(1.0, max(0.0, final.duration / 2)))
    except Exception:  # noqa: BLE001 - el thumbnail es cosmetico, no aborta la compilacion
        thumb = None

    return CompiledOutput(
        path=out_path,
        thumb_path=thumb,
        duration=final.duration,
        width=final.width,
        height=final.height,
        size_bytes=out_path.stat().st_size,
        audio_report=audio_report,
    )
