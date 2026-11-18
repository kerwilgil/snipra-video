"""Transcripción local con faster-whisper → segmentos con timestamps por palabra."""

from __future__ import annotations

import contextlib
import json
import wave
from collections.abc import Callable
from pathlib import Path

from app.core.config import settings
from app.core.media import extract_audio, extract_audio_segment
from app.pipeline.ingest import find_source, job_dir

_model_cache: dict[str, object] = {}
ProgressCallback = Callable[[float, str], None]


def _get_model(model_name: str | None = None):
    selected_model = model_name or settings.whisper_model
    key = f"{selected_model}:{settings.whisper_device}:{settings.whisper_compute_type}"
    if key not in _model_cache:
        from faster_whisper import WhisperModel

        _model_cache[key] = WhisperModel(
            selected_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
    return _model_cache[key]


def load_wav_16k_mono(path: str | Path):
    """Lee el WAV mono 16 kHz que genera ffmpeg y lo devuelve como float32 en [-1, 1].

    Se pasa a faster-whisper como array para que no decodifique audio con PyAV; asi el
    backend no carga en su proceso las bibliotecas FFmpeg/x264/x265 (GPL) de PyAV.
    """
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        fmt = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
        if fmt != (1, 2, 16000):
            raise ValueError("Se esperaba un WAV PCM s16le mono de 16 kHz")
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def segments_path(job_id: str) -> Path:
    return job_dir(job_id) / "segments.json"


def transcribe_job(
    job_id: str,
    progress_callback: ProgressCallback | None = None,
    intervals: list[tuple[float, float]] | None = None,
    model_name: str | None = None,
) -> dict:
    """Transcribe el source del job y guarda segments.json. Devuelve el dict resultante."""
    src = find_source(job_id)
    wav = job_dir(job_id) / "audio.wav"
    if progress_callback is not None:
        progress_callback(0.02, "Preparando audio")

    selected_intervals = intervals or []
    if selected_intervals:
        wav_paths = []
        for idx, (start, end) in enumerate(selected_intervals):
            wav_part = job_dir(job_id) / f"audio-{idx:02d}.wav"
            extract_audio_segment(src, wav_part, start=start, end=end)
            wav_paths.append((wav_part, start, end))
    else:
        extract_audio(src, wav)
        wav_paths = [(wav, 0.0, 0.0)]

    if progress_callback is not None:
        progress_callback(0.08, "Cargando modelo de transcripcion")

    model = _get_model(model_name)
    language = ""
    total_duration = 0.0
    total_audio = (
        sum(max(0.1, end - start) for _, start, end in wav_paths) if selected_intervals else 0.0
    )
    completed_audio = 0.0
    segments = []
    for wav_path, offset, interval_end in wav_paths:
        seg_iter, info = model.transcribe(
            load_wav_16k_mono(wav_path), word_timestamps=True, vad_filter=True
        )
        language = language or info.language
        total_duration = max(total_duration, interval_end or info.duration)
        interval_duration = max(0.1, interval_end - offset) if interval_end else info.duration
        for s in seg_iter:
            progress_total = total_audio or info.duration or 1.0
            if progress_callback is not None and progress_total:
                local = min(interval_duration, s.end)
                pct = max(0.1, min(0.98, (completed_audio + local) / progress_total))
                progress_callback(pct, f"Transcribiendo {round(pct * 100)}%")
            words = [
                {
                    "start": round(w.start + offset, 3),
                    "end": round(w.end + offset, 3),
                    "word": w.word,
                }
                for w in (s.words or [])
            ]
            segments.append(
                {
                    "start": round(s.start + offset, 3),
                    "end": round(s.end + offset, 3),
                    "text": s.text.strip(),
                    "words": words,
                }
            )
        completed_audio += interval_duration

    result = {
        "language": language,
        "duration": round(total_duration, 3),
        "segments": segments,
        "transcribed_intervals": selected_intervals,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    segments_path(job_id).write_text(payload, encoding="utf-8")
    if progress_callback is not None:
        progress_callback(1.0, "Transcripcion completada")

    # limpia el wav temporal
    with contextlib.suppress(OSError):
        wav.unlink(missing_ok=True)
        for wav_path, _, _ in wav_paths:
            wav_path.unlink(missing_ok=True)

    return result


def load_segments(job_id: str) -> dict:
    p = segments_path(job_id)
    if not p.exists():
        raise FileNotFoundError(f"No hay segments.json para el job {job_id}")
    return json.loads(p.read_text(encoding="utf-8"))
