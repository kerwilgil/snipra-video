"""Tests de SponsorBlock: extracción de video ID, filtrado puro y fetch fail-open."""

from __future__ import annotations

import httpx
import pytest

from app.pipeline import sponsorblock
from app.pipeline.detect import detect_clips, detect_music_clips
from app.providers.base import ClipCandidate

# --- extract_youtube_video_id -------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/watch?v=dQw4w9WgXcQ&t=10s", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?si=abc", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_extract_youtube_video_id_supported_formats(url, expected):
    assert sponsorblock.extract_youtube_video_id(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://www.tiktok.com/@user/video/12345",
        "https://vimeo.com/12345",
        "not-a-url",
        "https://www.youtube.com/watch?x=1",  # sin v=
        "",
    ],
)
def test_extract_youtube_video_id_non_youtube_returns_none(url):
    assert sponsorblock.extract_youtube_video_id(url) is None


# --- fetch_sponsor_segments (fail-open) ---------------------------------------


def test_fetch_sponsor_segments_network_error_returns_empty(monkeypatch):
    def _raise(*args, **kwargs):
        raise httpx.ConnectTimeout("timeout")

    monkeypatch.setattr(sponsorblock.httpx, "get", _raise)
    assert sponsorblock.fetch_sponsor_segments("dQw4w9WgXcQ") == []


def test_fetch_sponsor_segments_404_returns_empty(monkeypatch):
    class _Resp:
        status_code = 404

    monkeypatch.setattr(sponsorblock.httpx, "get", lambda *a, **k: _Resp())
    assert sponsorblock.fetch_sponsor_segments("dQw4w9WgXcQ") == []


def test_fetch_sponsor_segments_parses_valid_payload(monkeypatch):
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"segment": [10.0, 40.0], "category": "sponsor"},
                {"segment": [100.0, 110.0], "category": "selfpromo"},
                {"segment": "not-a-list", "category": "sponsor"},  # malformado: se ignora
            ]

    monkeypatch.setattr(sponsorblock.httpx, "get", lambda *a, **k: _Resp())
    segments = sponsorblock.fetch_sponsor_segments("dQw4w9WgXcQ")
    assert segments == [
        {"start": 10.0, "end": 40.0, "category": "sponsor"},
        {"start": 100.0, "end": 110.0, "category": "selfpromo"},
    ]


def test_fetch_segments_for_url_skips_non_youtube():
    assert sponsorblock.fetch_segments_for_url("https://vimeo.com/12345") == []


# --- filter_candidates_by_sponsor_segments (pura, sin red) --------------------


def _cand(start: float, end: float, score: float = 0.5) -> ClipCandidate:
    return ClipCandidate(start=start, end=end, score=score)


def test_filter_no_segments_is_noop():
    candidates = [_cand(0, 60), _cand(100, 160)]
    out = sponsorblock.filter_candidates_by_sponsor_segments(candidates, [], min_seconds=45)
    assert out == candidates


def test_filter_discards_candidate_fully_inside_sponsor_segment():
    candidates = [_cand(0, 60)]
    segments = [{"start": 0, "end": 60, "category": "sponsor"}]
    out = sponsorblock.filter_candidates_by_sponsor_segments(candidates, segments, min_seconds=45)
    assert out == []


def test_filter_discards_when_overlap_exceeds_40_percent():
    # 60s candidato, 30s de solape con sponsor => 50% > 40% => descartado
    candidates = [_cand(0, 60)]
    segments = [{"start": 0, "end": 30, "category": "sponsor"}]
    out = sponsorblock.filter_candidates_by_sponsor_segments(candidates, segments, min_seconds=45)
    assert out == []


def test_filter_trims_start_when_sponsor_overlaps_beginning():
    # 90s candidato, sponsor cubre los primeros 20s (22% overlap) => recorta el inicio
    candidates = [_cand(0, 90)]
    segments = [{"start": 0, "end": 20, "category": "sponsor"}]
    out = sponsorblock.filter_candidates_by_sponsor_segments(candidates, segments, min_seconds=45)
    assert len(out) == 1
    assert out[0].start == 20
    assert out[0].end == 90


def test_filter_trims_end_when_sponsor_overlaps_tail():
    candidates = [_cand(0, 90)]
    segments = [{"start": 70, "end": 90, "category": "sponsor"}]
    out = sponsorblock.filter_candidates_by_sponsor_segments(candidates, segments, min_seconds=45)
    assert len(out) == 1
    assert out[0].start == 0
    assert out[0].end == 70


def test_filter_drops_when_trim_leaves_clip_too_short():
    # overlap 23/60 = 38% (bajo el umbral de descarte de 40%) => intenta recortar,
    # pero el resultado (37s) queda debajo de min_seconds*0.85 (38.25s) => descarta
    candidates = [_cand(0, 60)]
    segments = [{"start": 0, "end": 23, "category": "sponsor"}]
    out = sponsorblock.filter_candidates_by_sponsor_segments(candidates, segments, min_seconds=45)
    assert out == []


def test_filter_leaves_untouched_candidates_outside_sponsor_zones():
    candidates = [_cand(0, 60), _cand(200, 260)]
    segments = [{"start": 100, "end": 160, "category": "sponsor"}]
    out = sponsorblock.filter_candidates_by_sponsor_segments(candidates, segments, min_seconds=45)
    assert [c.start for c in out] == [0, 200]
    assert [c.end for c in out] == [60, 260]


def test_filter_merges_overlapping_sponsor_segments_before_evaluating():
    # dos tramos adyacentes/solapados que juntos cubren 0-50s (>40% de un candidato de 60s)
    candidates = [_cand(0, 60)]
    segments = [
        {"start": 0, "end": 30, "category": "sponsor"},
        {"start": 25, "end": 50, "category": "selfpromo"},
    ]
    out = sponsorblock.filter_candidates_by_sponsor_segments(candidates, segments, min_seconds=45)
    assert out == []


# --- integración con detect_clips / detect_music_clips ------------------------


def test_detect_clips_without_sponsorblock_segments_is_unchanged(fake_segments):
    """Un job sin sponsorblock_enabled (segments=None) no cambia el comportamiento actual."""
    baseline = detect_clips(
        fake_segments, provider="naive", model="", num_clips=3, formats=["9x16"]
    )
    with_none = detect_clips(
        fake_segments,
        provider="naive",
        model="",
        num_clips=3,
        formats=["9x16"],
        sponsorblock_segments=None,
    )
    with_empty = detect_clips(
        fake_segments,
        provider="naive",
        model="",
        num_clips=3,
        formats=["9x16"],
        sponsorblock_segments=[],
    )
    baseline_bounds = [(c.start, c.end) for c in baseline]
    assert baseline_bounds == [(c.start, c.end) for c in with_none]
    assert baseline_bounds == [(c.start, c.end) for c in with_empty]


def test_detect_clips_never_empty_when_sponsorblock_covers_everything(fake_segments):
    """Nunca deja el pool en 0 por el filtro: si el filtrado vacía el pool, usa el sin filtrar."""
    total = fake_segments["duration"]
    events: list[tuple[str, str]] = []
    cands = detect_clips(
        fake_segments,
        provider="naive",
        model="",
        num_clips=3,
        formats=["9x16"],
        sponsorblock_segments=[{"start": 0, "end": total, "category": "sponsor"}],
        event_callback=lambda level, msg: events.append((level, msg)),
    )
    assert len(cands) >= 1
    assert any("sin filtrar" in msg for _, msg in events)


def test_detect_music_clips_never_empty_when_sponsorblock_covers_everything():
    clips = detect_music_clips(
        300.0,
        {"scene_times": []},
        num_clips=5,
        min_seconds=45,
        max_seconds=90,
        sponsorblock_segments=[{"start": 0, "end": 300, "category": "sponsor"}],
    )
    assert len(clips) == 5
