"""Targeted transcript retrieval for Editor IA; never prompt instructions."""

from __future__ import annotations

from app.editor.director import untrusted_evidence
from app.pipeline.transcribe import load_segments


def get_range(job_id: str | None, start_us: int, end_us: int) -> list[dict]:
    if not job_id or end_us <= start_us:
        return []
    try:
        segments = load_segments(job_id).get("segments", [])
    except FileNotFoundError:
        return []
    start, end = start_us / 1_000_000, end_us / 1_000_000
    return [
        {
            "start_us": round(float(seg["start"]) * 1_000_000),
            "end_us": round(float(seg["end"]) * 1_000_000),
            "evidence": untrusted_evidence(seg.get("text", "")),
        }
        for seg in segments
        if float(seg.get("end", 0)) > start and float(seg.get("start", 0)) < end
    ]
