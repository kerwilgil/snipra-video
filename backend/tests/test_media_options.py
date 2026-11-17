"""Tests del contrato de opciones de medios (media foundation 1.0)."""

from __future__ import annotations

from app.core.media_options import (
    AUDIO_CLEANUP_MODES,
    COLOR_HANDLING_MODES,
    STABILIZATION_MODES,
    VIDEO_DENOISE_MODES,
    all_options_off,
    normalize_media_options,
)


def test_defaults_are_safe(monkeypatch):
    monkeypatch.setattr("app.core.media_options.settings", type("S", (), {
        "color_handling": "auto",
        "stabilization": "off",
        "video_denoise": "off",
        "audio_cleanup": "off",
    }))
    options = normalize_media_options(None)
    assert options == {
        "destination_profile": "auto",
        "color_handling": "auto",
        "stabilization": "off",
        "video_denoise": "off",
        "audio_cleanup": "off",
    }
    assert all_options_off(options) is True


def test_job_options_win_over_settings(monkeypatch):
    monkeypatch.setattr("app.core.media_options.settings", type("S", (), {
        "color_handling": "sdr",
        "stabilization": "strong",
        "video_denoise": "light",
        "audio_cleanup": "voice",
    }))
    options = normalize_media_options({"color_handling": "auto"})
    assert options["color_handling"] == "auto"
    assert options["stabilization"] == "strong"
    assert options["video_denoise"] == "light"
    assert options["audio_cleanup"] == "voice"


def test_unknown_keys_and_values_are_dropped(monkeypatch):
    monkeypatch.setattr("app.core.media_options.settings", type("S", (), {
        "color_handling": "auto",
        "stabilization": "off",
        "video_denoise": "off",
        "audio_cleanup": "off",
    }))
    options = normalize_media_options(
        {
            "color_handling": "hdr2000",
            "stabilization": "extreme",
            "video_denoise": "ultra",
            "audio_cleanup": "ai_magic",
            "destination_profile": "netflix",
            "evil_key": True,
        }
    )
    assert options == {
        "destination_profile": "auto",
        "color_handling": "auto",
        "stabilization": "off",
        "video_denoise": "off",
        "audio_cleanup": "off",
    }


def test_valid_profile_ids_pass():
    options = normalize_media_options({"destination_profile": "instagram_reels"})
    assert options["destination_profile"] == "instagram_reels"
    options = normalize_media_options({"destination_profile": "TIKTOK"})
    assert options["destination_profile"] == "tiktok"


def test_all_valid_enum_values_are_accepted(monkeypatch):
    monkeypatch.setattr("app.core.media_options.settings", type("S", (), {
        "color_handling": "auto",
        "stabilization": "off",
        "video_denoise": "off",
        "audio_cleanup": "off",
    }))
    for mode in COLOR_HANDLING_MODES:
        assert normalize_media_options({"color_handling": mode})["color_handling"] == mode
    for mode in STABILIZATION_MODES:
        assert normalize_media_options({"stabilization": mode})["stabilization"] == mode
    for mode in VIDEO_DENOISE_MODES:
        assert normalize_media_options({"video_denoise": mode})["video_denoise"] == mode
    for mode in AUDIO_CLEANUP_MODES:
        assert normalize_media_options({"audio_cleanup": mode})["audio_cleanup"] == mode


def test_non_dict_input_is_safe():
    assert normalize_media_options("nope")["stabilization"] == "off"
    assert normalize_media_options(["list"])["color_handling"] == "auto"
