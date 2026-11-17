"""Render de clips: corte → reframe por formato → subtítulos quemados → thumbnail."""

import array
import contextlib
import json
import re
import shutil
import subprocess
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.config import settings
from app.core.media import make_thumbnail, probe, run
from app.core.media_options import normalize_media_options
from app.core.platform_profiles import resolve_profile
from app.core.processing_modes import (
    PODCAST_VIRAL_CONFIG,
    is_podcast_viral_mode,
    reframe_request_for_processing_mode,
)
from app.pipeline import editorial, media_filters, menu_filter, reedit, tracking
from app.pipeline.delivery_qc import run_delivery_qc
from app.pipeline.extract import extract_clip
from app.pipeline.ingest import job_dir
from app.pipeline.reframe import auto_reframe_mode, build_filtergraph, resolve_mode, target_size
from app.pipeline.subtitles import build_ass, normalize_style_config, write_ass
from app.providers.base import ClipCandidate

MAX_QUALITY_ATTEMPTS = 3


@dataclass
class RenderedAsset:
    format: str
    path: str
    thumb_path: str
    width: int
    height: int
    duration: float
    size_bytes: int


@dataclass
class QualityReport:
    passed: bool
    attempts: int
    issues: list[str]
    checks: dict[str, bool | float | int | str]


@dataclass
class TimelineFrame:
    time: float
    filename: str


def _legacy_slug(idx: int, title: str) -> str:
    base = "".join(c if c.isalnum() else "-" for c in title.lower()).strip("-")
    base = "-".join(filter(None, base.split("-")))[:40] or "clip"
    return f"{idx + 1:02d}-{base}"


def _slug(idx: int, title: str) -> str:
    normalized = unicodedata.normalize("NFKD", title)
    ascii_title = normalized.encode("ascii", "ignore").decode("ascii").lower()
    base = "".join(c if c.isalnum() else "-" for c in ascii_title).strip("-")
    base = "-".join(filter(None, base.split("-")))[:40] or "clip"
    return f"{idx + 1:02d}-{base}"


@lru_cache(maxsize=1)
def _has_subtitles_filter() -> bool:
    result = subprocess.run(
        [settings.ffmpeg_path, "-hide_banner", "-filters"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return " subtitles " in result.stdout or "\n ... subtitles " in result.stdout


def _encode_format(
    work: Path,
    base_name: str,
    graph_base: str,
    out_name: str,
    subs_name: str | None,
    cards_name: str | None = None,
    privacy_masks: list[dict] | None = None,
    audio_filter: str = "loudnorm=I=-14:TP=-1.5:LRA=11",
) -> None:
    """Corre FFmpeg con cwd=work: aplica reframe (+nitidez +subtítulos) y codifica el formato.

    graph_base toma [0:v] y produce [v] ya en el tamaño destino.
    audio_filter es la cadena de audio completa (cleanup + loudnorm FINAL).
    """
    # Sharpening post-crop: disimula el upscale del recorte 16:9 -> 9:16.
    # Solo luma (croma=0) para no introducir ruido de color ni halos. Sutil.
    graph, vmap = _final_filtergraph(
        graph_base,
        subs_name,
        cards_name=cards_name,
        privacy_masks=privacy_masks,
    )

    cmd = [
        settings.ffmpeg_path,
        "-y",
        "-i",
        base_name,
        "-filter_complex",
        graph,
        "-map",
        vmap,
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        # Fuerza 8-bit 4:2:0: una fuente AV1/VP9 10-bit (YouTube/Facebook la sirven
        # cada vez mas seguido) puede producir un libx264 de salida en yuv420p10le
        # -- H.264 valido, pero WhatsApp y varios reproductores lo rechazan igual
        # que rechazarian AV1.
        "-pix_fmt",
        "yuv420p",
        # Nivela cada clip al estandar de loudness social (-14 LUFS integrado con
        # el true peak del perfil de destino). Sin esto un clip mas bajo que el
        # resto del feed pierde retencion aunque el contenido sea igual de bueno.
        # Un solo pase (no two-pass): el ahorro de tiempo importa mas que el
        # ultimo decimal de precision para clips cortos. Simple filter sobre el
        # audio mapeado; si el clip no tiene pista de audio, no hay stream sobre
        # el que aplicar (cleanup = no-op válido).
        "-af",
        audio_filter,
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        out_name,
    ]
    run(cmd, timeout=1800, cwd=work)


def _final_filtergraph(
    graph_base: str,
    subs_name: str | None,
    *,
    cards_name: str | None = None,
    privacy_masks: list[dict] | None = None,
) -> tuple[str, str]:
    """Construye el graph final: reframe/overlay -> nitidez -> captions."""
    # Sharpening post-crop: disimula el upscale del recorte 16:9 -> 9:16.
    # Solo luma (croma=0) para no introducir ruido de color ni halos. Sutil.
    graph = graph_base
    current = "v"
    for index, mask in enumerate(privacy_masks or []):
        width = max(2, int(mask.get("width", 0)))
        height = max(2, int(mask.get("height", 0)))
        x = max(0, int(mask.get("x", 0)))
        y = max(0, int(mask.get("y", 0)))
        start = max(0.0, float(mask.get("start", 0.0)))
        end = max(start, float(mask.get("end", start)))
        graph += (
            f";[{current}]split=2[maskbase{index}][maskcrop{index}]"
            f";[maskcrop{index}]crop={width}:{height}:{x}:{y},"
            f"boxblur=12:2:6:1[maskblur{index}]"
            f";[maskbase{index}][maskblur{index}]overlay={x}:{y}:"
            f"enable='between(t,{start:.3f},{end:.3f})'[masked{index}]"
        )
        current = f"masked{index}"
    graph += f";[{current}]unsharp=5:5:0.8:5:5:0.0[vsharp]"
    current = "vsharp"
    if cards_name and _has_subtitles_filter():
        graph += f";[{current}]subtitles=filename={cards_name}[vcards]"
        current = "vcards"
    if subs_name and _has_subtitles_filter():
        graph += f";[{current}]subtitles=filename={subs_name}[vout]"
        current = "vout"
    return graph, f"[{current}]"


def _artifact_dir(job_id: str, index: int, title: str) -> Path:
    return job_dir(job_id) / "clips" / _slug(index, title)


def _write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path, fallback):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def load_clip_privacy_suggestions(job_id: str, index: int, title: str) -> list[dict]:
    return _load_json(_artifact_dir(job_id, index, title) / "privacy_suggestions.json", [])


def load_clip_protected_zones(job_id: str, index: int, title: str) -> list[dict]:
    return _load_json(_artifact_dir(job_id, index, title) / "protected_zones.json", [])


def load_clip_editorial_qa(job_id: str, index: int, title: str) -> dict:
    return _load_json(_artifact_dir(job_id, index, title) / "editorial_qa.json", {})


def _segments_to_subtitle_blocks(
    segments: list[dict] | None,
    clip_start: float,
    clip_end: float,
) -> list[dict]:
    blocks: list[dict] = []
    if not segments:
        return blocks
    for seg in segments:
        seg_start = float(seg.get("start", clip_start))
        seg_end = float(seg.get("end", clip_end))
        if seg_end <= clip_start or seg_start >= clip_end:
            continue
        words = []
        for word in seg.get("words") or []:
            ws, we = word.get("start"), word.get("end")
            if ws is None or we is None or we <= clip_start or ws >= clip_end:
                continue
            words.append(
                {
                    "word": str(word.get("word", "")).strip(),
                    "start": round(max(0.0, float(ws) - clip_start), 3),
                    "end": round(max(0.0, min(float(we), clip_end) - clip_start), 3),
                }
            )
        text = " ".join(w["word"] for w in words).strip() or str(seg.get("text", "")).strip()
        if text:
            start = round(max(0.0, seg_start - clip_start), 3)
            end = round(
                max(start + 0.8, min(seg_end, clip_end) - clip_start),
                3,
            )
            blocks.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "highlight_words": [],
                    "words": words,
                }
            )
    return blocks


def subtitle_blocks_to_segments(blocks: list[dict]) -> list[dict]:
    segments: list[dict] = []
    for block in blocks:
        start = max(0.0, float(block.get("start", 0.0)))
        end = max(start + 0.8, min(start + 3.5, float(block.get("end", start + 0.8))))
        words = block.get("words") or []
        if not words:
            parts = [p for p in str(block.get("text", "")).split() if p]
            step = (end - start) / max(1, len(parts))
            words = [
                {
                    "word": part,
                    "start": round(start + i * step, 3),
                    "end": round(start + (i + 1) * step, 3),
                }
                for i, part in enumerate(parts)
            ]
        segments.append(
            {
                "start": start,
                "end": end,
                "text": str(block.get("text", "")).strip(),
                "words": words,
            }
        )
    return segments


def _validate_export(path: Path, fmt: str) -> None:
    info = probe(path)
    expected_w, expected_h = target_size(fmt)
    errors = []
    if (info.width, info.height) != (expected_w, expected_h):
        errors.append(
            f"El render {fmt} debe ser {expected_w}x{expected_h}, no {info.width}x{info.height}."
        )
    if fmt == "9x16" and abs((info.width / max(info.height, 1)) - (9 / 16)) > 0.01:
        errors.append("El aspect ratio final no es 9:16 real.")
    if errors:
        raise RuntimeError(" ".join(errors))


def _write_quality_report(path: Path, report: QualityReport) -> None:
    _write_json(
        path,
        {
            "passed": report.passed,
            "attempts": report.attempts,
            "issues": report.issues,
            "checks": report.checks,
        },
    )


def _timeline_frame_times(duration: float, extra_times: list[float] | None = None) -> list[float]:
    if duration <= 0:
        return [0.0]
    points = [0.12, 0.35, 0.58, 0.82]
    times = [duration * point for point in points]
    for value in extra_times or []:
        times.extend((value - 0.08, value + 0.08))
    return sorted({round(min(max(value, 0.0), max(0.0, duration - 0.1)), 3) for value in times})


def _write_timeline_frames(
    clip_dir: Path,
    video_path: Path,
    fmt: str,
    duration: float,
    extra_times: list[float] | None = None,
) -> list[TimelineFrame]:
    frames: list[TimelineFrame] = []
    for idx, t in enumerate(_timeline_frame_times(duration, extra_times), start=1):
        frame = clip_dir / f"timeline-{fmt}-frame-{idx}.jpg"
        make_thumbnail(video_path, frame, at_seconds=t)
        frames.append(TimelineFrame(time=t, filename=frame.name))
    return frames


def _audio_energy_samples(video_path: Path, sample_count: int = 48) -> list[dict]:
    """Return compact RMS samples for Timeline QA waveform rendering."""
    try:
        info = probe(video_path)
    except Exception:  # noqa: BLE001
        return []
    if not info.has_audio or info.duration <= 0:
        return []

    sample_rate = 8000
    cmd = [
        settings.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=180)
    except Exception:  # noqa: BLE001
        return []
    if proc.returncode != 0 or not proc.stdout:
        return []

    samples = array.array("h")
    samples.frombytes(proc.stdout)
    if not samples:
        return []
    if samples.itemsize != 2:
        return []

    bucket_size = max(1, len(samples) // sample_count)
    out: list[dict] = []
    for idx in range(0, len(samples), bucket_size):
        chunk = samples[idx : idx + bucket_size]
        if not chunk:
            continue
        rms = (sum(int(value) * int(value) for value in chunk) / len(chunk)) ** 0.5
        out.append(
            {
                "time": round((idx / max(1, len(samples))) * info.duration, 3),
                "rms": round(min(1.0, rms / 32768.0), 4),
            }
        )
        if len(out) >= sample_count:
            break
    return out


def _normalized_block_words(block: dict) -> list[dict]:
    """Palabras de un bloque de subtítulo en shape {word, start, end}.

    Los bloques traen la palabra bajo "word" cuando vienen de la detección automática
    (`_segments_to_subtitle_blocks`) o bajo "text" cuando vienen de subtitles.json /
    `pipeline/subtitles.py::_join()` (edición manual, rewrite de captions). Se soportan
    ambas claves para no perder palabras según el origen del bloque.
    """
    words: list[dict] = []
    for word in block.get("words") or []:
        text = str(word.get("word") or word.get("text") or "").strip()
        if not text:
            continue
        words.append(
            {
                "word": text,
                "start": round(float(word.get("start", 0.0)), 3),
                "end": round(float(word.get("end", 0.0)), 3),
            }
        )
    return words


def _timeline_words(blocks: list[dict]) -> list[dict]:
    words: list[dict] = []
    for block in blocks:
        words.extend(_normalized_block_words(block))
    return words


def _write_timeline_artifact(
    clip_dir: Path,
    *,
    candidate: ClipCandidate,
    formats: list[dict],
    blocks: list[dict],
    style_cfg: dict,
    processing_mode: str,
    requested_mode: str,
    effective_start: float,
    trim_offset: float,
    cards: list[dict] | None = None,
    protected_zones: list[dict] | None = None,
    privacy_suggestions: list[dict] | None = None,
    privacy_masks: list[dict] | None = None,
    editorial_qa: dict | None = None,
) -> None:
    source_ranges = [item.model_dump() for item in candidate.source_ranges]
    source_start = (
        min(item["start"] for item in source_ranges) if source_ranges else effective_start
    )
    source_end = max(item["end"] for item in source_ranges) if source_ranges else candidate.end
    _write_json(
        clip_dir / "timeline.json",
        {
            "version": 1,
            "clip": {
                "title": candidate.title,
                "hook": candidate.hook,
                "score": candidate.score,
                "reason": candidate.reason,
                "score_breakdown": [
                    item.model_dump(exclude_none=True, exclude_defaults=True)
                    for item in candidate.score_breakdown
                ],
                "source_start": round(source_start, 3),
                "source_end": round(source_end, 3),
                "source_ranges": source_ranges,
                "duration": round(candidate.duration, 3),
                "trim_offset": round(trim_offset, 3),
            },
            "processing": {
                "mode": processing_mode,
                "requested_reframe": requested_mode,
                "caption_style": style_cfg.get("style", ""),
            },
            "formats": formats,
            "subtitles": [
                {
                    "start": round(float(block.get("start", 0.0)), 3),
                    "end": round(float(block.get("end", 0.0)), 3),
                    "text": str(block.get("text", "")).strip(),
                    "word_count": len(block.get("words") or []),
                    "words": _normalized_block_words(block),
                }
                for block in blocks
            ],
            "words": _timeline_words(blocks),
            "editorial": {
                "cards": cards or [],
                "protected_zones": protected_zones or [],
                "privacy_suggestions": privacy_suggestions or [],
                "privacy_masks": privacy_masks or [],
                "qa": editorial_qa or {},
            },
        },
    )


def _ffmpeg_stderr(cmd: list[str], timeout: int = 180) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.stderr or ""


def _black_segments(path: Path) -> list[tuple[float, float, float]]:
    stderr = _ffmpeg_stderr(
        [
            settings.ffmpeg_path,
            "-hide_banner",
            "-i",
            str(path),
            "-vf",
            "blackdetect=d=0.6:pix_th=0.10",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    segments = []
    pattern = re.compile(
        r"black_start:(?P<start>\d+(?:\.\d+)?) "
        r"black_end:(?P<end>\d+(?:\.\d+)?) "
        r"black_duration:(?P<duration>\d+(?:\.\d+)?)"
    )
    for match in pattern.finditer(stderr):
        segments.append(
            (
                float(match.group("start")),
                float(match.group("end")),
                float(match.group("duration")),
            )
        )
    return segments


def _mostly_silent(path: Path, duration: float) -> bool:
    if duration <= 0:
        return False
    stderr = _ffmpeg_stderr(
        [
            settings.ffmpeg_path,
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            "silencedetect=n=-50dB:d=0.5",
            "-vn",
            "-f",
            "null",
            "-",
        ]
    )
    silence_durations = [
        float(match.group(1)) for match in re.finditer(r"silence_duration: (\d+(?:\.\d+)?)", stderr)
    ]
    return bool(silence_durations) and max(silence_durations) >= duration * 0.95


def _has_dialogue_events(path: Path) -> bool:
    if not path.exists():
        return False
    with contextlib.suppress(OSError):
        return "Dialogue:" in path.read_text(encoding="utf-8", errors="ignore")
    return False


def _evaluate_render_quality(
    path: Path,
    fmt: str,
    *,
    expected_duration: float,
    expected_has_audio: bool,
    subtitles_file: Path | None,
    attempt: int,
) -> QualityReport:
    issues: list[str] = []
    info = probe(path)
    expected_w, expected_h = target_size(fmt)
    duration_delta = abs(info.duration - expected_duration)
    max_duration_delta = max(0.75, expected_duration * 0.03)
    checks: dict[str, bool | float | int | str] = {
        "format": fmt,
        "attempt": attempt,
        "width": info.width,
        "height": info.height,
        "duration": round(info.duration, 3),
        "expected_duration": round(expected_duration, 3),
        "duration_delta": round(duration_delta, 3),
        "has_audio": info.has_audio,
        "expected_audio": expected_has_audio,
        "captions_expected": subtitles_file is not None,
    }

    if (info.width, info.height) != (expected_w, expected_h):
        issues.append(f"resolucion inesperada: {info.width}x{info.height}")
    if duration_delta > max_duration_delta:
        issues.append(f"duracion inesperada: {info.duration:.2f}s vs {expected_duration:.2f}s")
    if expected_has_audio and not info.has_audio:
        issues.append("audio faltante")
    if expected_has_audio and info.has_audio:
        silent = _mostly_silent(path, info.duration)
        checks["mostly_silent"] = silent
        if silent:
            issues.append("audio casi totalmente silencioso")
    if subtitles_file is not None:
        captions_present = _has_dialogue_events(subtitles_file)
        checks["captions_file_has_dialogue"] = captions_present
        if not captions_present:
            issues.append("subtitulos esperados sin eventos Dialogue")

    black_segments = _black_segments(path)
    longest_black = max((duration for _, _, duration in black_segments), default=0.0)
    checks["black_segments"] = len(black_segments)
    checks["longest_black_seconds"] = round(longest_black, 3)
    if longest_black >= max(1.0, info.duration * 0.25):
        issues.append(f"segmento negro largo: {longest_black:.2f}s")

    return QualityReport(
        passed=not issues,
        attempts=attempt,
        issues=issues,
        checks=checks,
    )


def _encode_with_quality_gate(
    work: Path,
    base_name: str,
    graph_base: str,
    out_name: str,
    subs_name: str | None,
    cards_name: str | None = None,
    privacy_masks: list[dict] | None = None,
    audio_filter: str = "loudnorm=I=-14:TP=-1.5:LRA=11",
    *,
    fmt: str,
    expected_duration: float,
    expected_has_audio: bool,
    event_callback: Callable[[str, str], None] | None = None,
) -> QualityReport:
    out_path = work / out_name
    subs_file = work / subs_name if subs_name else None
    last_report: QualityReport | None = None
    for attempt in range(1, MAX_QUALITY_ATTEMPTS + 1):
        _encode_format(
            work,
            base_name,
            graph_base,
            out_name,
            subs_name,
            cards_name=cards_name,
            privacy_masks=privacy_masks,
            audio_filter=audio_filter,
        )
        _validate_export(out_path, fmt)
        report = _evaluate_render_quality(
            out_path,
            fmt,
            expected_duration=expected_duration,
            expected_has_audio=expected_has_audio,
            subtitles_file=subs_file,
            attempt=attempt,
        )
        last_report = report
        if report.passed:
            _write_quality_report(work / f"quality-{fmt}.json", report)
            if attempt > 1 and event_callback:
                event_callback("info", f"Quality gate OK en intento {attempt} ({fmt})")
            return report
        if event_callback:
            event_callback(
                "warning",
                f"Quality gate intento {attempt}/{MAX_QUALITY_ATTEMPTS} "
                f"fallo ({fmt}): {', '.join(report.issues)}",
            )

    assert last_report is not None
    _write_quality_report(work / f"quality-{fmt}.json", last_report)
    raise RuntimeError(
        f"Quality gate fallo tras {MAX_QUALITY_ATTEMPTS} intentos ({fmt}): "
        + "; ".join(last_report.issues)
    )


def render_clip(
    job_id: str,
    src: str | Path,
    index: int,
    candidate: ClipCandidate,
    formats: list[str],
    segments: list[dict] | None = None,
    processing_mode: str = "auto",
    caption_style: str = "viral_bold",
    captions_enabled: bool = True,
    style_config: dict | None = None,
    event_callback: Callable[[str, str], None] | None = None,
    editorial_options: dict | None = None,
    privacy_masks: list[dict] | None = None,
    media_options: dict | None = None,
) -> list[RenderedAsset]:
    clips_dir = job_dir(job_id) / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_dir = _artifact_dir(job_id, index, candidate.title)
    clip_dir.mkdir(parents=True, exist_ok=True)

    # 0) probe avanzado de la fuente (probe first): decide si el corte base
    # debe preservar 10-bit para que el tone map HDR tenga precisión. Para
    # SDR se mantiene el comportamiento histórico (yuv420p 8-bit).
    src_info = probe(src)
    extract_pix_fmt = "yuv420p"
    if src_info.hdr and (src_info.bit_depth or 8) > 8:
        extract_pix_fmt = "yuv420p10le"

    # 1) corte base (aspecto original, con audio). Un edit gaming puede estar
    # formado por varios tramos no contiguos de la fuente.
    base = clip_dir / "clip_base.mp4"
    render_candidate = candidate
    render_segments = segments
    if candidate.source_ranges:
        ranges = [(item.start, item.end) for item in candidate.source_ranges]
        reedit.build_concat_base(src, ranges, base)
        render_segments, total = reedit.remap_segments(segments or [], ranges)
        render_candidate = candidate.model_copy(update={"start": 0.0, "end": max(0.1, total)})
        _write_json(
            clip_dir / "source_ranges.json",
            [item.model_dump() for item in candidate.source_ranges],
        )
    else:
        extract_clip(src, candidate.start, candidate.end, base, pix_fmt=extract_pix_fmt)

    return render_from_base(
        job_id,
        base,
        index,
        render_candidate,
        formats,
        segments=render_segments,
        processing_mode=processing_mode,
        caption_style=caption_style,
        captions_enabled=captions_enabled,
        style_config=style_config,
        event_callback=event_callback,
        editorial_options=editorial_options,
        privacy_masks=privacy_masks,
        media_options=media_options,
    )


def render_from_base(
    job_id: str,
    base: Path,
    index: int,
    candidate: ClipCandidate,
    formats: list[str],
    segments: list[dict] | None = None,
    processing_mode: str = "auto",
    caption_style: str = "viral_bold",
    captions_enabled: bool = True,
    style_config: dict | None = None,
    subtitle_blocks: list[dict] | None = None,
    event_callback: Callable[[str, str], None] | None = None,
    editorial_options: dict | None = None,
    privacy_masks: list[dict] | None = None,
    media_options: dict | None = None,
) -> list[RenderedAsset]:
    """Renderiza por formato a partir de un corte base ya existente.

    `segments` y `candidate` deben estar en el timeline del base (clip empieza en 0 si el
    base ya viene recortado/concatenado, como en el re-edit por transcript).

    `media_options` activa las mejoras de medios (destination_profile,
    color_handling, stabilization, video_denoise, audio_cleanup). Con todo
    en OFF el resultado es idéntico al pipeline clásico.
    """
    clip_dir = _artifact_dir(job_id, index, candidate.title)
    clip_dir.mkdir(parents=True, exist_ok=True)
    base = Path(base)
    canonical_base = clip_dir / "clip_base.mp4"
    if base.resolve() != canonical_base.resolve():
        shutil.copyfile(base, canonical_base)
        base = canonical_base
    base_info = probe(base)
    editorial_options = editorial_options or {}

    # --- Media foundation: probe first + capability-driven ---
    media_options = normalize_media_options(media_options)

    # HDR: detección precisa desde el probe avanzado. Con Dolby Vision
    # registramos el formato de origen; el pipeline convierte el base layer
    # a SDR con tone mapping (no promete preservación de metadatos dinámicos).
    source_hdr_format: str | None = None
    needs_tonemap = False
    if base_info.hdr:
        source_hdr_format = (
            "dolby_vision" if base_info.dolby_vision else base_info.hdr_format
        )
        needs_tonemap = True
        if media_options["color_handling"] == "preserve" and event_callback:
            event_callback(
                "warning",
                "HDR preserve solicitado pero el pipeline de entrega (H.264 8-bit "
                "SDR) no puede preservar HDR: se aplicará tone map a SDR.",
            )

    # Estabilización: pase de análisis (vidstabdetect) una sola vez por clip.
    # El archivo de transforms vive en el temp del job y se limpia siempre.
    stabilization_mode = media_options["stabilization"]
    stab_transform_ready = False
    if stabilization_mode != "off":
        backend = media_filters.stabilization_backend()
        if backend == "vidstab":
            try:
                media_filters.run_stabilization_analysis(
                    base.name, clip_dir, stabilization_mode
                )
                stab_transform_ready = True
            except Exception as exc:  # noqa: BLE001
                if event_callback:
                    event_callback(
                        "warning",
                        f"Estabilización omitida: falló el análisis vidstab ({exc}).",
                    )
        elif backend == "deshake" and event_callback:
            event_callback(
                "warning",
                "Estabilización con fallback deshake: vidstab no disponible en "
                "este FFmpeg (resultado no equivalente).",
            )
        elif event_callback:
            event_callback(
                "warning",
                "Estabilización omitida: este FFmpeg no dispone de vidstab ni deshake.",
            )

    # Pre-chain de video (formato-independiente): tonemap → stab → denoise.
    pre_filters, pre_notes = media_filters.compose_video_prefilters(
        tonemap_needed=needs_tonemap,
        stabilization_mode=stabilization_mode,
        denoise_mode=media_options["video_denoise"],
        stab_transform_ready=stab_transform_ready,
    )
    for note in pre_notes:
        if event_callback:
            event_callback("warning", note)

    # Cadena de audio: cleanup (opcional) + loudnorm FINAL (siempre).
    audio_prefix: list[str] = []
    cleanup_chain = media_filters.audio_cleanup_chain(media_options["audio_cleanup"])
    if cleanup_chain:
        audio_prefix.append(cleanup_chain)
    elif (
        media_options["audio_cleanup"] != "off"
        and event_callback
        and base_info.has_audio
    ):
        event_callback(
            "warning",
            "Limpieza de audio omitida: este FFmpeg no dispone de los filtros.",
        )

    # 1b) Tracking primero — necesario antes del cálculo de subtítulos porque
    # puede señalar que hay que recortar el inicio del clip (plano general / intro).
    # - track/auto → paneo animado siguiendo al sujeto (analyze_subject)
    # - center      → crop ESTÁTICO centrado en la cara (analyze_face_anchor)
    # Sin extra [vision], ambas devuelven None → fallback a center geométrico.
    # Optimización: si todos los formatos destino preservan el aspecto del source,
    # YOLO no aportará nada → skipeamos el análisis por completo.
    reframe_request = reframe_request_for_processing_mode(processing_mode, settings.reframe_mode)
    requested_mode = resolve_mode(reframe_request)
    dual_track: tuple[tracking.SubjectTrack, tracking.SubjectTrack] | None = None
    if all(
        auto_reframe_mode(base_info.width, base_info.height, fmt, requested_mode) == "preserve"
        for fmt in (formats or ["9x16"])
    ):
        track = None
    elif requested_mode == "split":
        dual_track = tracking.analyze_dual_subjects(str(base))
        # Sin 2 caras a la vez en ningun momento del clip: no hay nada que
        # partir en 2 -- cae a seguir a la unica persona dominante.
        track = tracking.analyze_subject(str(base)) if dual_track is None else None
    elif requested_mode in ("track", "auto"):
        track = tracking.analyze_subject(str(base))
    elif requested_mode == "center":
        track = tracking.analyze_face_anchor(str(base))
    else:
        track = None

    # Si el tracking detectó un inicio sin cara (plano general, intro animada),
    # recortar el clip base hasta la primera detección real. Solo en renders frescos
    # (sin bloques de subtítulos ya calculados) para no alterar ediciones manuales.
    # Tras recortar, re-ejecutar el tracking sobre el clip ya limpio (el track
    # original sigue teniendo tiempos del clip sin recortar).
    existing_blocks = _load_json(clip_dir / "subtitles.json", [])
    trim_offset = getattr(track, "trim_start", 0.0)
    should_trim = (
        trim_offset > 0
        and subtitle_blocks is None
        and not existing_blocks
        and not candidate.source_ranges
    )
    if should_trim:
        _trim_tmp = clip_dir / "_trim_tmp.mp4"
        extract_clip(base, trim_offset, base_info.duration, _trim_tmp)
        _trim_tmp.replace(base)
        base_info = probe(base)
        # Re-ejecutar tracking sobre el clip recortado (empieza con cara → track limpio)
        if requested_mode in ("track", "auto"):
            track = tracking.analyze_subject(str(base))
        elif requested_mode == "center":
            track = tracking.analyze_face_anchor(str(base))

    # Inicio efectivo en el timeline de la fuente (ajustado por trim si aplica)
    effective_start = candidate.start + (trim_offset if should_trim else 0.0)

    style_cfg = normalize_style_config(style_config, caption_style)
    if is_podcast_viral_mode(processing_mode) and not style_cfg.get("style"):
        style_cfg["style"] = "viral_bold"
    blocks = [] if not captions_enabled else subtitle_blocks
    if captions_enabled and blocks is None:
        blocks = existing_blocks or _segments_to_subtitle_blocks(
            segments,
            effective_start,
            candidate.end,
        )
    render_segments = (
        subtitle_blocks_to_segments(blocks)
        if blocks
        else ((segments or []) if captions_enabled else [])
    )
    clip_start = 0.0 if blocks else effective_start
    clip_end = candidate.duration if blocks else candidate.end

    _write_json(clip_dir / "transcript.json", {"segments": segments or []})
    _write_json(clip_dir / "subtitles.json", blocks or [])
    _write_json(clip_dir / "style_config.json", style_cfg)
    _write_json(
        clip_dir / "processing_config.json",
        {
            "processing_mode": processing_mode,
            "caption_style": style_cfg["style"],
            "captions_enabled": captions_enabled,
            "format": "9:16",
            "resolution": "1080x1920",
            "resize_mode": "cover",
            "allow_black_bars": False,
            "allow_letterbox": False,
            "podcast_viral": (
                PODCAST_VIRAL_CONFIG if is_podcast_viral_mode(processing_mode) else None
            ),
            "media": {
                "destination_profile": media_options["destination_profile"],
                "color_handling": media_options["color_handling"],
                "stabilization": media_options["stabilization"],
                "video_denoise": media_options["video_denoise"],
                "audio_cleanup": media_options["audio_cleanup"],
                "source_hdr_format": source_hdr_format,
                "tonemapped_to_sdr": needs_tonemap,
            },
        },
    )

    assets: list[RenderedAsset] = []
    timeline_formats: list[dict] = []
    all_protected_zones: list[dict] = []
    all_privacy_suggestions: list[dict] = []
    cards = editorial.remap_cards(
        candidate.editorial_cards,
        candidate.source_ranges,
        effective_start,
    )
    try:
        for fmt in formats or ["9x16"]:
            w, h = target_size(fmt)
            profile = resolve_profile(media_options["destination_profile"], fmt)

            zones = (
                editorial.protected_zones(
                    w,
                    h,
                    captions=bool(render_segments),
                    subject=track is not None or dual_track is not None,
                    profile=profile,
                )
                if editorial_options.get("protected_zones", True)
                else []
            )
            zones = [{**zone, "format": fmt} for zone in zones]

            cards_name = None
            if cards:
                cards_file = clip_dir / f"editorial-cards-{fmt}.ass"
                cards_file.write_text(
                    editorial.build_cards_ass(
                        cards,
                        w,
                        h,
                        editorial.card_position(zones, h),
                    ),
                    encoding="utf-8",
                )
                cards_name = cards_file.name

            # 2) subtítulos para este formato (tamaño destino)
            subs_name = None
            if render_segments:
                ass = build_ass(
                    render_segments,
                    clip_start,
                    clip_end,
                    w,
                    h,
                    style=style_cfg["style"],
                    style_config=style_cfg,
                )
                subs_file = clip_dir / f"subtitles-{fmt}.ass"
                if write_ass(ass, subs_file):
                    subs_name = subs_file.name

            # 3) reframe + burn
            # split → 2 crops apilados, uno por persona; track → paneo animado;
            # center con cara detectada → crop fijo en la cara; center sin cara /
            # preserve / blur → filtergraph estático. split sin 2 caras nunca
            # detectadas cae al track de la persona dominante (ver dual_track más
            # arriba), y de ahi al mismo fallback estatico que track.
            # Las mejoras de medios (tonemap/estabilización/denoise) se anteponen
            # como pre-chain [0:v]→[vpre]: ANTES del reframe/crop final.
            in_label = "vpre" if pre_filters else "0:v"
            graph_base = None
            mode = auto_reframe_mode(base_info.width, base_info.height, fmt, requested_mode)
            if mode == "split" and dual_track is not None:
                graph_base = tracking.build_split_graph(
                    dual_track[0], dual_track[1], fmt, in_label=in_label, out_label="v"
                )
            elif mode in ("track", "center", "split") and track is not None:
                graph_base = tracking.build_track_graph(
                    track, fmt, in_label=in_label, out_label="v"
                )
            if graph_base is None:
                static_mode = "center" if mode in ("track", "split") else mode
                graph_base = build_filtergraph(fmt, static_mode, in_label=in_label, out_label="v")
            if pre_filters:
                graph_base = f"[0:v]{','.join(pre_filters)}[{in_label}];{graph_base}"

            # Loudness objetivo del perfil de destino (single source of truth).
            # Con perfil genérico produce exactamente loudnorm=I=-14:TP=-1.5:LRA=11.
            audio_filter = ",".join([*audio_prefix, media_filters.loudnorm_chain(profile)])

            out_name = "render_final.mp4" if fmt == "9x16" else f"render_final_{fmt}.mp4"
            quality_report = _encode_with_quality_gate(
                clip_dir,
                base.name,
                graph_base,
                out_name,
                subs_name,
                cards_name=cards_name,
                privacy_masks=[
                    mask
                    for mask in (privacy_masks or [])
                    if not mask.get("format") or mask.get("format") == fmt
                ],
                audio_filter=audio_filter,
                fmt=fmt,
                expected_duration=base_info.duration,
                expected_has_audio=base_info.has_audio,
                event_callback=event_callback,
            )
            out_path = clip_dir / out_name

            # 3b) Super-resolución IA opcional (gated por GPU + binario; no-op si falta)
            if getattr(settings, "enable_ai_upscale", False):
                from app.pipeline.upscale import upscale_video

                sr_tmp = clip_dir / f"sr_{out_name}"
                if upscale_video(out_path, sr_tmp):
                    sr_tmp.replace(out_path)
                    _validate_export(out_path, fmt)
                    quality_report = _evaluate_render_quality(
                        out_path,
                        fmt,
                        expected_duration=base_info.duration,
                        expected_has_audio=base_info.has_audio,
                        subtitles_file=(clip_dir / subs_name) if subs_name else None,
                        attempt=quality_report.attempts,
                    )
                    _write_quality_report(clip_dir / f"quality-{fmt}.json", quality_report)
                    if not quality_report.passed:
                        raise RuntimeError(
                            f"Quality gate fallo tras upscale ({fmt}): "
                            + "; ".join(quality_report.issues)
                        )

            # 3c) Delivery QC: verify last. El render solo es correcto si el
            # archivo cumple las reglas del perfil de destino (codec, color,
            # loudness, faststart...). FAIL = fallo mecánico → se reporta claro.
            delivery_result = run_delivery_qc(
                out_path,
                fmt,
                profile,
                expected_duration=base_info.duration,
                expected_has_audio=base_info.has_audio,
                source_hdr_format=source_hdr_format,
                subs_path=(clip_dir / subs_name) if subs_name else None,
            )
            _write_json(clip_dir / f"delivery_qc-{fmt}.json", delivery_result.to_dict())
            if not delivery_result.passed:
                raise RuntimeError(
                    f"Delivery QC fallo ({fmt}, perfil {profile.id}): "
                    + "; ".join(
                        f"{check.id}: {check.message or check.status}"
                        for check in delivery_result.failures
                    )
                )
            qc_warnings = [
                check for check in delivery_result.checks if check.status == "warn"
            ]
            if qc_warnings and event_callback:
                event_callback(
                    "warning",
                    f"Delivery QC avisos ({fmt}): "
                    + "; ".join(
                        f"{check.id} {check.message or ''}".strip()
                        for check in qc_warnings
                    ),
                )

            # 4) thumbnail
            thumb = clip_dir / f"thumbnail-{fmt}.jpg"
            make_thumbnail(out_path, thumb, at_seconds=min(1.0, candidate.duration / 2))

            info = probe(out_path)
            format_masks = [
                mask
                for mask in (privacy_masks or [])
                if not mask.get("format") or mask.get("format") == fmt
            ]
            boundary_times = [
                float(value)
                for overlay in [*cards, *format_masks]
                for value in (
                    overlay.get("at", overlay.get("start", 0.0)),
                    overlay.get(
                        "end",
                        float(overlay.get("at", 0.0)) + float(overlay.get("duration", 0.0)),
                    ),
                )
            ]
            timeline_frames = _write_timeline_frames(
                clip_dir,
                out_path,
                fmt,
                info.duration,
                boundary_times if editorial_options.get("overlay_qa", True) else None,
            )
            if editorial_options.get("privacy_assist") or editorial_options.get(
                "protected_zones", True
            ):
                for frame in timeline_frames:
                    regions = menu_filter.frame_regions(str(clip_dir / frame.filename))
                    if editorial_options.get("protected_zones", True):
                        zones.extend(
                            {
                                "kind": "ocr",
                                "format": fmt,
                                "time": frame.time,
                                "x": region["x"],
                                "y": region["y"],
                                "width": region["width"],
                                "height": region["height"],
                                "text": region["text"][:80],
                            }
                            for region in regions
                        )
                    if editorial_options.get("privacy_assist"):
                        all_privacy_suggestions.extend(
                            {**suggestion, "format": fmt}
                            for suggestion in editorial.privacy_suggestions(regions, at=frame.time)
                        )
            all_protected_zones.extend(zones)
            audio_energy = _audio_energy_samples(out_path)
            timeline_formats.append(
                {
                    "format": fmt,
                    "mode": mode,
                    "width": info.width,
                    "height": info.height,
                    "duration": round(info.duration, 3),
                    "size_bytes": out_path.stat().st_size,
                    "video": out_name,
                    "thumbnail": thumb.name,
                    "quality": {
                        "passed": quality_report.passed,
                        "attempts": quality_report.attempts,
                        "issues": quality_report.issues,
                        "checks": quality_report.checks,
                    },
                    "delivery_qc": delivery_result.to_dict(),
                    "destination_profile": profile.id,
                    "frames": [
                        {"time": frame.time, "filename": frame.filename}
                        for frame in timeline_frames
                    ],
                    "audio_energy": audio_energy,
                    "protected_zones": zones,
                }
            )
            assets.append(
                RenderedAsset(
                    format=fmt,
                    path=str(out_path),
                    thumb_path=str(thumb),
                    width=info.width,
                    height=info.height,
                    duration=info.duration,
                    size_bytes=out_path.stat().st_size,
                )
            )

    finally:
        # Cleanup de transforms de vidstab tras completar o fallar (nunca queda en disco).
        (clip_dir / media_filters.STABILIZATION_TRANSFORM_NAME).unlink(missing_ok=True)
    qa = editorial.overlay_qa(cards, privacy_masks or [], base_info.duration)
    qa["boundary_frames"] = sum(
        max(0, len(item.get("frames", [])) - 4) for item in timeline_formats
    )
    if not editorial_options.get("overlay_qa", True):
        qa = {"passed": True, "issues": [], "checked_overlays": 0, "disabled": True}
    _write_json(clip_dir / "editorial_cards.json", cards)
    _write_json(clip_dir / "privacy_suggestions.json", all_privacy_suggestions)
    _write_json(clip_dir / "privacy_masks.json", privacy_masks or [])
    _write_json(clip_dir / "protected_zones.json", all_protected_zones)
    _write_json(clip_dir / "editorial_qa.json", qa)

    _write_timeline_artifact(
        clip_dir,
        candidate=candidate,
        formats=timeline_formats,
        blocks=blocks or [],
        style_cfg=style_cfg,
        processing_mode=processing_mode,
        requested_mode=requested_mode,
        effective_start=effective_start,
        trim_offset=trim_offset if should_trim else 0.0,
        cards=cards,
        protected_zones=all_protected_zones,
        privacy_suggestions=all_privacy_suggestions,
        privacy_masks=privacy_masks or [],
        editorial_qa=qa,
    )

    return assets


def reframe_mode_effective() -> str:
    return resolve_mode(settings.reframe_mode)
