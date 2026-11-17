"""yt-dlp necesita saber dónde está ffmpeg en la app empaquetada (no está en el PATH del launcher)."""

from __future__ import annotations

from app.core.config import settings
from app.pipeline import ingest


def test_ffmpeg_location_follows_absolute_ffmpeg_path(tmp_path, monkeypatch):
    helpers = tmp_path / "Helpers"
    helpers.mkdir()
    (helpers / "ffmpeg").write_text("", encoding="utf-8")
    monkeypatch.setattr(settings, "ffmpeg_path", str(helpers / "ffmpeg"))

    opts = ingest._base_ydl_options(str(tmp_path / "%(id)s.%(ext)s"))

    assert opts["ffmpeg_location"] == str(helpers)


def test_ffmpeg_location_absent_for_bare_command(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ffmpeg_path", "ffmpeg")

    opts = ingest._base_ydl_options(str(tmp_path / "%(id)s.%(ext)s"))

    assert "ffmpeg_location" not in opts
