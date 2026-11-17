import pytest

from app.providers.captions import (
    TranslationError,
    _parse_rewrite_response,
    rewrite_subtitle_blocks,
    translate_subtitle_blocks,
)


def test_parse_rewrite_response_accepts_json_array():
    raw = '["Gancho claro", "Idea directa"]'

    assert _parse_rewrite_response(raw, 2) == ["Gancho claro", "Idea directa"]


def test_rewrite_subtitle_blocks_naive_preserves_timing_with_new_words():
    blocks = [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "Eh este mira esto es importante",
            "words": [{"word": "Eh", "start": 0.0, "end": 0.2}],
        }
    ]

    rewritten = rewrite_subtitle_blocks(blocks, provider="naive", model="")

    assert rewritten[0]["start"] == 0.0
    assert rewritten[0]["end"] == 2.0
    assert "este" not in rewritten[0]["text"].lower()
    assert rewritten[0]["words"][0]["start"] == 0.0
    assert rewritten[0]["words"][-1]["end"] == 2.0


def test_translate_subtitle_blocks_naive_raises_clear_error():
    blocks = [{"start": 0.0, "end": 2.0, "text": "Hola mundo", "words": []}]

    with pytest.raises(TranslationError):
        translate_subtitle_blocks(blocks, target_language="ingles", provider="naive", model="")


def test_translate_subtitle_blocks_empty_language_raises():
    blocks = [{"start": 0.0, "end": 2.0, "text": "Hola mundo", "words": []}]

    with pytest.raises(TranslationError):
        translate_subtitle_blocks(blocks, target_language="  ", provider="claude", model="")


def test_translate_subtitle_blocks_applies_llm_translation_preserving_timing(monkeypatch):
    blocks = [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "Hola mundo",
            "words": [{"word": "Hola", "start": 0.0, "end": 1.0}],
        }
    ]

    class FakeDetector:
        def _complete(self, system_prompt, user_prompt):
            return '["Hello world"]'

    monkeypatch.setattr(
        "app.providers.captions.get_detector", lambda provider, model: FakeDetector()
    )

    translated = translate_subtitle_blocks(
        blocks, target_language="ingles", provider="claude", model="claude-haiku-4-5"
    )

    assert translated[0]["text"] == "Hello world"
    assert translated[0]["start"] == 0.0
    assert translated[0]["end"] == 2.0
    assert translated[0]["words"][0]["start"] == 0.0
    assert translated[0]["words"][-1]["end"] == 2.0


def test_translate_subtitle_blocks_llm_failure_raises_translation_error(monkeypatch):
    blocks = [{"start": 0.0, "end": 2.0, "text": "Hola mundo", "words": []}]

    class BrokenDetector:
        def _complete(self, system_prompt, user_prompt):
            raise RuntimeError("proveedor caido")

    monkeypatch.setattr(
        "app.providers.captions.get_detector", lambda provider, model: BrokenDetector()
    )

    with pytest.raises(TranslationError):
        translate_subtitle_blocks(blocks, target_language="ingles", provider="claude", model="")
