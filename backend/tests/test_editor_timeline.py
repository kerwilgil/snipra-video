from app.editor import timeline as tl


def seeded():
    timeline = tl.empty_timeline()
    track = tl.new_track("video", 0)
    timeline["tracks"] = [track]
    clip = tl.new_clip("asset", 0, 10_000_000, 0)
    return tl.add_clip(timeline, track["id"], clip), track, clip


def test_split_trim_move_duplicate_delete_and_mapping_are_exact():
    timeline, track, clip = seeded()
    split = tl.split_clip(timeline, clip["id"], 4_000_000)
    left, right = split["tracks"][0]["clips"]
    assert left["source_out_us"] == right["source_in_us"] == 4_000_000
    trimmed = tl.trim_clip(split, right["id"], source_in_us=5_000_000)
    assert trimmed["tracks"][0]["clips"][1]["timeline_start_us"] == 5_000_000
    moved = tl.move_clip(trimmed, right["id"], track["id"], 12_000_000)
    assert tl.resolve_at(moved, 12_100_000)[0]["source_us"] == 5_100_000
    duplicated = tl.duplicate_clip(moved, right["id"], track["id"], 20_000_000)
    assert len(duplicated["tracks"][0]["clips"]) == 3
    ripple = tl.delete_clip(duplicated, right["id"], ripple=True)
    assert len(ripple["tracks"][0]["clips"]) == 2


def test_invalid_split_and_locked_track_fail_closed():
    timeline, track, clip = seeded()
    try:
        tl.split_clip(timeline, clip["id"], 0)
    except tl.InvalidRange:
        pass
    else:
        raise AssertionError("invalid split accepted")
    locked = {**timeline, "tracks": [{**timeline["tracks"][0], "locked": True}]}
    try:
        tl.delete_clip(locked, clip["id"])
    except tl.LockedTrack:
        pass
    else:
        raise AssertionError("locked track mutation accepted")


def test_preview_segments_share_exact_mapping_and_clip_properties():
    timeline, track, clip = seeded()
    updated = tl.set_clip_properties(
        timeline,
        clip["id"],
        playback_rate=2.0,
        transform={"x": 0.1, "y": -0.2, "scale": 1.25},
        audio={"volume": 0.6, "muted": True},
    )
    segment = tl.preview_segments(updated)[0]
    assert segment["timeline_end_us"] == 5_000_000
    assert segment["source_in_us"] == 0
    assert segment["source_out_us"] == 10_000_000
    assert segment["playback_rate"] == 2.0
    assert segment["transform"] == {"x": 0.1, "y": -0.2, "scale": 1.25}
    assert segment["audio"] == {"volume": 0.6, "muted": True}
    assert tl.resolve_at(updated, 1_000_000)[0]["source_us"] == 2_000_000


def test_empty_audio_track_controls_linked_video_audio_preview():
    timeline, _, clip = seeded()
    audio_track = tl.new_track("audio", 1)
    timeline["tracks"].append(audio_track)
    assert tl.preview_segments(timeline)[0]["audio"]["muted"] is False

    muted = tl.set_track_properties(timeline, audio_track["id"], muted=True)
    assert muted["tracks"][1]["muted"] is True
    assert tl.preview_segments(muted)[0]["audio"]["muted"] is True
    assert muted["tracks"][0]["clips"][0]["id"] == clip["id"]

    restored = tl.set_track_properties(muted, audio_track["id"], muted=False)
    assert tl.preview_segments(restored)[0]["audio"]["muted"] is False


def test_creative_semantics_validate_and_preview_content():
    timeline, track, clip = seeded()
    text_track = tl.new_track("text", 1)
    timeline["tracks"].append(text_track)
    creative = tl.set_clip_properties(
        timeline,
        clip["id"],
        playback_rate=1.25,
        transform={
            "crop": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8},
            "x": 0.2,
            "y": -0.1,
            "scale": 1.15,
        },
        audio={"volume": 0.7, "fade_in_us": 500_000, "fade_out_us": 500_000},
    )
    creative = tl.add_timed_content(
        creative,
        text_track["id"],
        tl.new_caption("Payoff", 1_000_000, 2_000_000, provenance="TRANSCRIPT"),
    )
    assert tl.preview_segments(creative)[0]["transform"]["crop"]["width"] == 0.8
    assert tl.preview_segments(creative)[0]["audio"]["fade_in_us"] == 500_000
    assert tl.preview_content(creative)[0]["text"] == "Payoff"
    try:
        tl.set_clip_properties(creative, clip["id"], audio={"fade_in_us": 99_000_000})
    except tl.InvalidRange:
        pass
    else:
        raise AssertionError("invalid fade accepted")


def test_creative_validation_fails_before_rendering():
    timeline, _, clip = seeded()
    invalid = (
        {"transform": {"crop": {"x": 0.9, "y": 0, "width": 0.2, "height": 1}}},
        {"transform": {"scale": 0}},
        {"audio": {"fade_out_us": 11_000_000}},
        {"playback_rate": 0},
    )
    for values in invalid:
        try:
            tl.set_clip_properties(timeline, clip["id"], **values)
        except tl.InvalidRange:
            pass
        else:
            raise AssertionError(f"invalid creative value accepted: {values}")
    track = tl.new_track("text", 1)
    timeline["tracks"].append(track)
    try:
        tl.new_caption("bad", 1_000_000, 1_000_000)
    except tl.InvalidRange:
        pass
    else:
        raise AssertionError("empty caption range accepted")
