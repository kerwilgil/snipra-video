"""Tests de la detección por LLM: parseo, flujo con mock y fallback a naive."""

import pytest

from app.pipeline.detect import detect_clips
from app.providers._llm import run_detection
from app.providers.base import ClipCandidate, DetectParams
from app.providers.prompt import chunk_segments, parse_candidates

VALID_JSON = """\
[
  {"start": 0.0, "end": 15.0, "title": "Gancho", "hook": "No vas a creer esto", "score": 0.9, "reason": "fuerte"},
  {"start": 20.0, "end": 38.0, "title": "Dato", "hook": "El 90% no sabe", "score": 0.7, "reason": "sorprende"}
]"""


def test_parse_plain_json():
    cands = parse_candidates(VALID_JSON)
    assert len(cands) == 2
    assert cands[0].title == "Gancho"
    assert cands[1].end == 38.0


def test_parse_with_markdown_fences():
    fenced = f"```json\n{VALID_JSON}\n```"
    assert len(parse_candidates(fenced)) == 2


def test_parse_with_surrounding_text():
    noisy = f"Aquí tienes los clips:\n{VALID_JSON}\nEspero que sirvan."
    assert len(parse_candidates(noisy)) == 2


def test_run_detection_with_mock(fake_segments):
    def fake_complete(system: str, user: str) -> str:
        assert "Transcripción" in user
        return VALID_JSON

    cands = run_detection(fake_complete, fake_segments["segments"], DetectParams(num_clips=5))
    assert len(cands) == 2


def test_run_detection_retries_then_fails(fake_segments):
    calls = {"n": 0}

    def bad_complete(system: str, user: str) -> str:
        calls["n"] += 1
        return "no soy json"

    with pytest.raises(RuntimeError):
        run_detection(bad_complete, fake_segments["segments"], DetectParams(), retries=1)
    assert calls["n"] == 2  # intento + 1 reintento


def _long_segments(n: int = 300) -> list[dict]:
    # cada segmento ~150 chars => fuerza varios chunks (evita truncar transcripciones largas)
    return [{"start": i * 10.0, "end": i * 10.0 + 9.0, "text": "palabra " * 18} for i in range(n)]


def test_chunk_segments_splits_long_transcript():
    chunks = chunk_segments(_long_segments(300))
    assert len(chunks) > 1
    # todos los segmentos quedan repartidos, sin perder ninguno
    assert sum(len(c) for c in chunks) == 300


def test_run_detection_analyzes_all_chunks(fake_segments):
    long_segments = _long_segments(300)
    expected_chunks = len(chunk_segments(long_segments))
    calls = {"n": 0}

    def fake_complete(system: str, user: str) -> str:
        calls["n"] += 1
        return '[{"start": 0.0, "end": 12.0, "title": "x", "hook": "y", "score": 0.8}]'

    cands = run_detection(fake_complete, long_segments, DetectParams(num_clips=3))
    assert calls["n"] == expected_chunks  # llama al LLM por cada bloque, no solo el inicio
    assert len(cands) == expected_chunks


def test_detect_clips_falls_back_to_naive(fake_segments):
    # provider claude sin API key => debe degradar a naive y devolver clips igual
    cands = detect_clips(
        fake_segments,
        provider="claude",
        model="",
        num_clips=3,
        formats=["9x16"],
    )
    assert len(cands) >= 1


def test_codex_text_failure_uses_claude_code_before_naive(monkeypatch, fake_segments):
    from app.pipeline import detect as detect_mod

    calls: list[str] = []

    class BrokenCodex:
        def detect(self, segments, params):
            calls.append("codex")
            raise RuntimeError("plugin de Codex inválido")

    class WorkingClaude:
        def detect(self, segments, params):
            calls.append("claude_code")
            return [
                ClipCandidate(
                    start=0,
                    end=50,
                    title="Elegido por Claude",
                    score=0.9,
                )
            ]

    monkeypatch.setattr(
        detect_mod,
        "get_detector",
        lambda provider, model: WorkingClaude() if provider == "claude_code" else BrokenCodex(),
    )
    monkeypatch.setattr(
        "app.providers.cli_subscription.resolve_provider_cli",
        lambda provider: "claude.exe" if provider == "claude_code" else None,
    )
    events: list[tuple[str, str]] = []

    cands = detect_mod.detect_clips(
        fake_segments,
        provider="codex",
        model="gpt-5.6-terra",
        num_clips=1,
        formats=["9x16"],
        event_callback=lambda level, message: events.append((level, message)),
    )

    assert calls == ["codex", "claude_code"]
    assert cands[0].title == "Elegido por Claude"
    assert any("Detección Codex falló" in message for _, message in events)
