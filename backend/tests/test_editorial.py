from app.pipeline import editorial
from app.providers.base import ClipCandidate, EditorialCard, SourceRange


def _candidate(**updates) -> ClipCandidate:
    values = {
        "start": 10.0,
        "end": 30.0,
        "title": "Idea",
        "reason": "unidad narrativa",
        "score": 0.8,
    }
    values.update(updates)
    return ClipCandidate(**values)


def test_compact_editing_removes_internal_silence_but_preserves_breathing_room():
    compact = editorial.compact_candidate(
        _candidate(),
        [],
        [(15.0, 17.0), (22.0, 23.0)],
    )
    assert len(compact.source_ranges) == 3
    assert compact.duration < 20.0
    assert compact.duration > 16.0
    assert all(item.kind == "compact" for item in compact.source_ranges)


def test_compact_editing_rejects_an_overaggressive_cut():
    original = _candidate()
    compact = editorial.compact_candidate(original, [], [(11.0, 29.0)])
    assert compact.source_ranges == []
    assert compact.duration == original.duration


def test_compact_editing_keeps_last_of_two_repeated_takes():
    segments = [
        {"start": 11.0, "end": 14.0, "text": "Esta es la frase que quiero explicar ahora"},
        {"start": 14.5, "end": 18.0, "text": "Esta es la frase que quiero explicar ahora"},
        {"start": 18.2, "end": 28.0, "text": "y esta es la conclusión definitiva"},
    ]
    compact = editorial.compact_candidate(_candidate(), segments, [])
    assert compact.source_ranges[0].end <= 11.0
    assert compact.source_ranges[1].start >= 14.0


def test_semantic_card_requires_literal_evidence_and_no_invented_numbers():
    valid = EditorialCard(
        kind="consequence",
        at=14.0,
        text="El costo se duplica",
        evidence="sube de 10 a 20",
    )
    invented = EditorialCard(
        kind="derived",
        at=15.0,
        text="Aumenta 300 por ciento",
        evidence="sube de 10 a 20",
    )
    unsupported = EditorialCard(
        kind="contrast",
        at=16.0,
        text="Dos caminos distintos",
        evidence="una frase que no existe",
    )
    candidate = _candidate(editorial_cards=[valid, invented, unsupported])
    cards = editorial.sanitize_cards(
        candidate,
        [{"start": 10.0, "end": 30.0, "text": "El precio sube de 10 a 20 al final."}],
    )
    assert cards == [valid]


def test_cards_are_remapped_to_a_compact_timeline():
    card = EditorialCard(kind="derived", at=22.0, text="Resultado", evidence="base")
    ranges = [
        SourceRange(start=10.0, end=15.0),
        SourceRange(start=20.0, end=30.0),
    ]
    assert editorial.remap_cards([card], ranges, 0.0)[0]["at"] == 7.0


def test_privacy_assist_suggests_but_does_not_confirm_sensitive_text():
    suggestions = editorial.privacy_suggestions(
        [
            {
                "text": "API key: demo-secret-value",
                "confidence": 0.94,
                "x": 10,
                "y": 20,
                "width": 200,
                "height": 30,
            }
        ],
        at=4.0,
    )
    assert len(suggestions) == 1
    assert suggestions[0]["confirmed"] is False
    assert suggestions[0]["start"] == 3.0


def test_overlay_qa_rejects_an_overlay_outside_the_clip():
    report = editorial.overlay_qa([{"at": 9.0, "duration": 4.0}], [], 10.0)
    assert report["passed"] is False
    assert report["issues"]


def test_cards_ass_contains_semantic_copy_and_timing():
    ass = editorial.build_cards_ass(
        [{"at": 1.0, "duration": 3.0, "kicker": "Contraste", "text": "Antes / después"}],
        1080,
        1920,
    )
    assert "Dialogue: 1,0:00:01.00,0:00:04.00" in ass
    assert "CONTRASTE" in ass
