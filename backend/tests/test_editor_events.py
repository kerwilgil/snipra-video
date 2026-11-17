from app.editor.events import best_event_pool


def test_highlight_pool_marks_multi_range_auto_clip_for_rework():
    pool = best_event_pool(
        [
            {"id": "montage", "source_ranges": [{"start": 0}, {"start": 30}], "legacy_score": 0.9},
            {"id": "event", "source_ranges": [{"start": 10}], "legacy_score": 0.6},
        ],
        "highlight",
    )
    by_id = {event["id"]: event for event in pool}
    assert by_id["montage"]["quality_gate"] == "REWORK"
    assert by_id["event"]["output_suitable"] is True
