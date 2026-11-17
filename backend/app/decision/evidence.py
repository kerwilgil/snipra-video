"""Normalize already extracted evidence. No media analysis, inference or scoring."""

from __future__ import annotations

import math

from app.decision.models import AudioEvidence, ClipEvidence, SpeechEvidence, VisualEvidence
from app.providers.base import ClipCandidate


def _coverage(ranges: list[tuple[float, float]], intervals: list) -> float:
    """Union coverage within selected source ranges (also supports montage)."""
    total = sum(end - start for start, end in ranges)
    covered = 0.0
    for start, end in ranges:
        spans = sorted(
            (max(start, float(a)), min(end, float(b)))
            for a, b in intervals
            if float(b) > start and float(a) < end
        )
        cursor = start
        for a, b in spans:
            covered += max(0.0, b - max(cursor, a))
            cursor = max(cursor, b)
    return min(1.0, covered / total)


def normalize_candidate(
    candidate: ClipCandidate,
    transcript_doc: dict,
    signals_doc: dict,
    *,
    video_id: str,
) -> ClipEvidence:
    ranges = [(span.start, span.end) for span in candidate.source_ranges] or [
        (candidate.start, candidate.end)
    ]
    segments = [
        segment
        for segment in transcript_doc.get("segments", [])
        if any(segment["end"] > start and segment["start"] < end for start, end in ranges)
    ]
    refs = [f"candidate:{video_id}:{candidate.start}:{candidate.end}"]
    audio = AudioEvidence()
    visual = VisualEvidence()
    speech = SpeechEvidence()
    if "silence_intervals" in signals_doc:
        audio.silence_ratio = _coverage(ranges, signals_doc["silence_intervals"])
        refs.append(f"signals:{video_id}:silence_intervals")
    if "loudness" in signals_doc:
        values = [
            float(db)
            for at, db in signals_doc["loudness"]
            if any(start <= at < end for start, end in ranges) and math.isfinite(float(db))
        ]
        if values:
            audio.energy = sum(values) / len(values)
            refs.append(f"signals:{video_id}:loudness")
    if "scene_times" in signals_doc:
        visual.scene_changes = sum(
            any(start <= at < end for start, end in ranges) for at in signals_doc["scene_times"]
        )
        refs.append(f"signals:{video_id}:scene_times")
    if segments:
        # Word timestamps provide exact membership; segment text is a coarse fallback.
        words = [
            word
            for segment in segments
            for word in segment.get("words", [])
            if any(start <= word.get("start", -1) < end for start, end in ranges)
        ]
        speech.transcript = (
            " ".join(str(word.get("word", "")).strip() for word in words)
            if words
            else " ".join(segment.get("text", "").strip() for segment in segments)
        )
        if words:
            speech.words_per_second = len(words) / candidate.duration
        audio.speech_ratio = _coverage(
            ranges,
            [(segment["start"], segment["end"]) for segment in segments],
        )
        refs.append(f"transcript:{video_id}:segments")
    return ClipEvidence(
        start_ms=round(candidate.start * 1000),
        end_ms=round(candidate.end * 1000),
        duration_ms=(
            round(candidate.duration * 1000)
            if candidate.source_ranges
            else round(candidate.end * 1000) - round(candidate.start * 1000)
        ),
        source_ranges=candidate.source_ranges,
        existing_score=candidate.score,
        audio=audio,
        visual=visual,
        speech=speech,
        signals={"score_breakdown": [item.model_dump() for item in candidate.score_breakdown]},
        evidence_refs=refs,
    )
