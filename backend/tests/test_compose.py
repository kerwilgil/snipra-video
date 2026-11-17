from app.core.media import VideoInfo
from app.pipeline import compose


def _info(duration: float, has_audio: bool = True) -> VideoInfo:
    return VideoInfo(duration=duration, width=1080, height=1920, has_audio=has_audio)


def test_normalize_transition_falls_back_to_none():
    assert compose.normalize_transition("fade") == "fade"
    assert compose.normalize_transition("FADE") == "fade"
    assert compose.normalize_transition("wipe") == "none"
    assert compose.normalize_transition(None) == "none"
    assert compose.normalize_transition("") == "none"


def test_hard_cut_uses_concat_with_one_stream_pair_per_clip():
    graph, video_map, audio_map = compose.build_filtergraph(
        [_info(30.0), _info(20.0), _info(25.0)], 1080, 1920
    )
    assert "concat=n=3:v=1:a=1[vcat][acat]" in graph
    assert "[v0][a0][v1][a1][v2][a2]concat" in graph
    assert video_map == "[vcat]"
    assert audio_map == "[acat]"
    # sin transicion no debe aparecer xfade
    assert "xfade" not in graph


def test_clip_without_audio_gets_silent_track_so_concat_does_not_break():
    graph, _, _ = compose.build_filtergraph(
        [_info(10.0, has_audio=True), _info(8.0, has_audio=False)], 1080, 1920
    )
    assert "[0:a]aresample=48000" in graph
    assert "anullsrc=channel_layout=stereo:sample_rate=48000" in graph
    assert "atrim=duration=8.000[a1]" in graph


def test_fade_offsets_account_for_accumulated_overlap():
    # 3 clips de 10s con cruce de 0.4s: el 2do cruce arranca en
    # (10 + 10 - 0.4) - 0.4 = 19.2, no en 20.
    graph, video_map, audio_map = compose.build_filtergraph(
        [_info(10.0), _info(10.0), _info(10.0)],
        1080,
        1920,
        transition="fade",
        transition_seconds=0.4,
    )
    assert "offset=9.600[vx1]" in graph
    assert "offset=19.200[vx2]" in graph
    assert "acrossfade=d=0.400" in graph
    assert video_map == "[vx2]"
    assert audio_map == "[ax2]"
    assert "concat=" not in graph


def test_transition_is_clamped_to_half_of_shortest_clip():
    # El clip mas corto es 1s: una transicion de 2s reventaria xfade.
    graph, _, _ = compose.build_filtergraph(
        [_info(10.0), _info(1.0)], 1080, 1920, transition="fade", transition_seconds=2.0
    )
    assert "duration=0.500" in graph


def test_single_clip_needs_neither_concat_nor_xfade():
    graph, video_map, audio_map = compose.build_filtergraph([_info(12.0)], 1080, 1920)
    assert "concat=" not in graph
    assert "xfade" not in graph
    assert video_map == "[v0]"
    assert audio_map == "[a0]"


def test_bgm_ducking_splits_voice_and_keeps_it_at_unity_gain():
    graph, _, audio_map = compose.build_filtergraph(
        [_info(10.0), _info(10.0)],
        1080,
        1920,
        bgm_index=2,
        bgm_volume=0.2,
        bgm_ducking=True,
    )
    assert "[2:a]volume=0.200" in graph
    # asplit: el label de la voz se consume dos veces (mezcla + disparador)
    assert "[acat]asplit=2[avoice][atrigger]" in graph
    assert "[bgmraw][atrigger]sidechaincompress=" in graph
    # normalize=0 evita que amix divida la voz entre 2
    assert "amix=inputs=2:duration=first:normalize=0[aout]" in graph
    assert audio_map == "[aout]"


def test_bgm_without_ducking_mixes_directly():
    graph, _, audio_map = compose.build_filtergraph(
        [_info(10.0), _info(10.0)], 1080, 1920, bgm_index=2, bgm_ducking=False
    )
    assert "sidechaincompress" not in graph
    assert "asplit" not in graph
    assert "[acat][bgmraw]amix=inputs=2:duration=first:normalize=0[aout]" in graph
    assert audio_map == "[aout]"


def test_bgm_volume_is_clamped_to_valid_range():
    graph, _, _ = compose.build_filtergraph([_info(5.0)], 1080, 1920, bgm_index=1, bgm_volume=9.0)
    assert "volume=1.000" in graph


def test_inputs_are_normalized_to_same_size_and_fps():
    graph, _, _ = compose.build_filtergraph([_info(5.0), _info(5.0)], 720, 1280)
    assert graph.count("fps=30") == 2
    assert graph.count("scale=720:1280:force_original_aspect_ratio=decrease") == 2
    assert graph.count("setsar=1") == 2


def test_automatic_bgm_volume_targets_music_below_voice(monkeypatch):
    measured = {"voice-a.mp4": -16.0, "voice-b.mp4": -20.0, "music.mp3": -12.0}
    monkeypatch.setattr(compose, "integrated_lufs", lambda path: measured[str(path)])
    monkeypatch.setattr(compose, "probe", lambda _path: _info(10.0))

    volume, report = compose.automatic_bgm_volume(
        ["voice-a.mp4", "voice-b.mp4"],
        "music.mp3",
        duck_lu=12.0,
    )

    assert 0.05 < volume < 0.2
    assert report["mode"] == "automatic_lufs"
    assert report["target_bgm_lufs"] < report["voice_lufs"]


def test_automatic_bgm_volume_falls_back_when_lufs_is_unavailable(monkeypatch):
    monkeypatch.setattr(compose, "integrated_lufs", lambda _path: None)
    volume, report = compose.automatic_bgm_volume(["voice.mp4"], "music.mp3")
    assert volume == 0.18
    assert report["mode"] == "fallback"
