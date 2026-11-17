"""Tests de la super-resolución IA (gated, sin binario real)."""

from app.pipeline import upscale


def test_upscale_unavailable_without_binary(monkeypatch):
    monkeypatch.setattr(upscale.settings, "realesrgan_path", "binario-que-no-existe-xyz")
    assert upscale.upscale_available() is False


def test_upscale_video_noop_when_unavailable(monkeypatch):
    # sin binario → devuelve None (el pipeline conserva el clip original)
    monkeypatch.setattr(upscale, "upscale_available", lambda: False)
    assert upscale.upscale_video("in.mp4", "out.mp4") is None


def test_upscale_video_survives_probe_error(monkeypatch):
    monkeypatch.setattr(upscale, "upscale_available", lambda: True)

    def _boom(_p):
        raise RuntimeError("probe falló")

    monkeypatch.setattr(upscale, "probe", _boom)
    # error interno → None, nunca propaga (no rompe el job)
    assert upscale.upscale_video("in.mp4", "out.mp4") is None
