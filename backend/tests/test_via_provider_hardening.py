"""Tests de N.1: provider text/visual desacoplados, policy de costo y
disponibilidad fail-closed de VIA. 0 llamadas reales a APIs.
"""

from __future__ import annotations

import base64

import pytest

from app.core.config import settings
from app.providers import vision
from app.providers.costs import PROVIDER_COST_CLASS, cost_class, is_paid_api, is_subscription
from app.video_intelligence.models import (
    VideoIntelligenceError,
    VideoIntelligenceErrorCode,
    ExtractedFrame,
)
from app.video_intelligence.providers import ExistingSnipraVLMProvider


@pytest.fixture
def clean_settings(monkeypatch):
    # Capability/cost tests stay offline; catalog behavior has its own suite.
    monkeypatch.setattr("app.providers.nvidia_catalog.resolve_model",
                        lambda value, **kw: value or "catalog/current-vision")
    for key, value in {
        "llm_provider": "naive",
        "llm_model": "",
        "anthropic_api_key": "",
        "openai_api_key": "",
        "nvidia_api_key": "",
        "vision_provider": "auto",
        "vision_model": "",
        "nvidia_vision_model": "meta/llama-3.2-90b-vision-instruct",
    }.items():
        monkeypatch.setattr(settings, key, value)
    return monkeypatch


# --- cost policy --------------------------------------------------------------

def test_cost_class_matrix():
    assert cost_class("naive") == "local"
    assert cost_class("claude_code") == "subscription"
    assert cost_class("codex") == "subscription"
    assert cost_class("nvidia") == "free_tier"
    assert cost_class("claude") == "paid_api"
    assert cost_class("openai") == "paid_api"
    assert cost_class("desconocido") == "unknown"
    assert is_paid_api("claude") and not is_paid_api("codex")
    assert is_subscription("codex") and not is_subscription("openai")
    # la matriz no puede quedar vacia ni perder proveedores clave
    for p in ("claude", "openai", "claude_code", "codex", "nvidia", "naive"):
        assert p in PROVIDER_COST_CLASS


# --- resolve_vision (policy auto vs explicit) --------------------------------

def test_auto_never_falls_back_to_paid_from_subscription(clean_settings):
    monkey = clean_settings
    monkey.setattr(settings, "llm_provider", "claude_code")
    p, m = vision.resolve_vision("claude_code", "claude-sonnet-5", via=True)
    assert p == "claude_code"
    assert m == "claude-sonnet-5"
    assert not is_paid_api(p)


def test_text_and_vision_can_be_different(clean_settings):
    monkey = clean_settings
    monkey.setattr(settings, "vision_provider", "codex")
    p, m = vision.resolve_vision("claude_code", "claude-sonnet-5", via=True)
    assert (p, m) == ("codex", vision.VISION_DEFAULT_MODELS["codex"])


def test_auto_with_paid_text_respects_explicit_user_choice(clean_settings):
    monkey = clean_settings
    monkey.setattr(settings, "llm_provider", "claude")
    p, m = vision.resolve_vision("claude", "claude-haiku-4-5", via=True)
    # El usuario eligio texto pagado deliberadamente: auto puede seguirlo.
    assert (p, m) == ("claude", "claude-haiku-4-5")


def test_nvidia_text_auto_does_not_enable_via_vision(clean_settings):
    monkey = clean_settings
    monkey.setattr(settings, "llm_provider", "nvidia")
    p, m = vision.resolve_vision("nvidia", "nvidia/nemotron-3-ultra-550b-a55b", via=True)
    assert (p, m) == ("", ""), "auto nunca habilita VIA con nvidia; debe ser explicito"


def test_nvidia_via_explicit_is_real_and_free(clean_settings):
    monkey = clean_settings
    monkey.setattr(settings, "vision_provider", "nvidia")
    p, m = vision.resolve_vision("nvidia", "x", via=True)
    assert p == "nvidia"
    assert m == "meta/llama-3.2-90b-vision-instruct"
    assert cost_class(p) == "free_tier"


def test_nvidia_never_for_rerank(clean_settings):
    monkey = clean_settings
    monkey.setattr(settings, "vision_provider", "nvidia")
    assert vision.resolve_vision("nvidia", "x") == ("", ""), "rerank nunca usa nvidia"


def test_rerank_unchanged_by_via_param(clean_settings):
    monkey = clean_settings
    monkey.setattr(settings, "vision_provider", "codex")
    assert vision.resolve_vision("codex", "") == ("codex", "gpt-5.6-terra")


# --- VIA availability fail-closed --------------------------------------------

def _provider():
    return ExistingSnipraVLMProvider()


def test_availability_refreshes_after_settings_change(clean_settings):
    monkey = clean_settings
    monkey.setattr("app.providers.cli_subscription.resolve_provider_cli", lambda name: "/cli")
    prov = _provider()
    assert prov.get_availability()["available"] is False
    monkey.setattr(settings, "vision_provider", "codex")
    assert prov.get_availability()["available"] is True
    assert prov.resolve_visual_identity() == ("codex", "gpt-5.6-terra")
    monkey.setattr(settings, "vision_provider", "auto")
    assert prov.get_availability()["available"] is False


def test_no_vision_provider_fail_closed(clean_settings):
    monkey = clean_settings
    monkey.setattr(settings, "llm_provider", "naive")
    prov = _provider()
    avail = prov.get_availability()
    assert avail["available"] is False
    assert avail["code"] == VideoIntelligenceErrorCode.VISUAL_PROVIDER_NOT_CONFIGURED
    with pytest.raises(VideoIntelligenceError) as excinfo:
        prov._check_availability()
    assert excinfo.value.code == VideoIntelligenceErrorCode.VISUAL_PROVIDER_NOT_CONFIGURED
    assert "vision" in (excinfo.value.message or "").lower()


def test_explicit_nvidia_missing_key_fail_closed(clean_settings):
    monkey = clean_settings
    monkey.setattr(settings, "vision_provider", "nvidia")
    prov = _provider()
    avail = prov.get_availability()
    assert avail["available"] is False
    assert avail["code"] == VideoIntelligenceErrorCode.MISSING_PROVIDER_CREDENTIALS


def test_cli_subscription_without_cli_fail_closed(clean_settings, monkeypatch):
    monkey = clean_settings
    monkey.setattr(settings, "llm_provider", "codex")
    monkeypatch.setattr(
        "app.providers.cli_subscription.resolve_provider_cli", lambda provider: None
    )
    prov = _provider()
    avail = prov.get_availability()
    assert avail["available"] is False
    assert avail["code"] == VideoIntelligenceErrorCode.VISUAL_PROVIDER_UNSUPPORTED


def test_claude_no_key_fail_closed(clean_settings):
    monkey = clean_settings
    monkey.setattr(settings, "llm_provider", "claude")
    monkey.setattr(settings, "anthropic_api_key", "")
    prov = _provider()
    code = VideoIntelligenceErrorCode.MISSING_PROVIDER_CREDENTIALS
    assert prov.get_availability()["code"] == code


def test_subscription_text_does_not_trigger_anthropic(monkeypatch):
    """text=codex nunca consulta ni requiere Anthropic."""
    monkey = monkeypatch
    monkey.setattr(settings, "llm_provider", "codex")
    monkey.setattr(settings, "llm_model", "gpt-5.6-terra")
    monkey.setattr(settings, "anthropic_api_key", "")
    monkey.setattr(settings, "vision_provider", "auto")

    called = {"anthropic": False}

    def _boom(*a, **k):
        called["anthropic"] = True
        raise AssertionError("Anthropic llamado con vision codex")

    monkey.setattr("anthropic.Anthropic", _boom, raising=False)
    monkey.setattr(
        "app.providers.cli_subscription.resolve_provider_cli", lambda p: "codex.exe"
    )
    prov = _provider()
    assert prov.get_availability()["available"] is True
    assert called["anthropic"] is False


# --- dispatch per-provider ----------------------------------------------------

def test_vlm_dispatch_nvidia_sends_frames_and_model(clean_settings, monkeypatch):
    monkey = clean_settings
    monkey.setattr(settings, "vision_provider", "nvidia")
    monkey.setattr(settings, "nvidia_api_key", "nvapi-test-key-123")

    captured = {}

    class FakeCompletions:
        def create(self, model, messages, **_):
            captured["model"] = model
            captured["messages"] = messages

            class C:
                message = type("M", (), {"content": "ok-via"})()

            return type("R", (), {"choices": [C()]})()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkey.setattr("openai.OpenAI", FakeOpenAI, raising=False)

    prov = _provider()
    text = prov._call_vlm_with_frames("sys", "pregunta", [
        ExtractedFrame(frame_id="frame_0000", timestamp_us=0, data=b"jpegdata"),
    ])
    assert text == "ok-via"
    assert captured["model"] == "meta/llama-3.2-90b-vision-instruct"
    assert captured["init"]["base_url"] == settings.nvidia_base_url
    content = captured["messages"][0]["content"]
    b64img = content[1]["image_url"]["url"]
    assert base64.b64decode(b64img.split(",", 1)[1]) == b"jpegdata"
    assert content[0]["text"] == "sys\n\npregunta"


def test_vlm_dispatch_missing_key_fail_closed_no_network(clean_settings):
    monkey = clean_settings
    monkey.setattr(settings, "vision_provider", "nvidia")
    prov = _provider()
    with pytest.raises(VideoIntelligenceError) as excinfo:
        prov._call_vlm_with_frames("s", "p", [
            ExtractedFrame(frame_id="frame_0000", timestamp_us=0, data=b"x"),
        ])
    assert excinfo.value.code == VideoIntelligenceErrorCode.MISSING_PROVIDER_CREDENTIALS


def test_vlm_unsupported_provider_codename(clean_settings, monkeypatch):
    monkey = clean_settings
    monkey.setattr(settings, "vision_provider", "codex")
    monkey.setattr(
        "app.providers.cli_subscription.resolve_provider_cli", lambda p: None
    )
    prov = _provider()
    with pytest.raises(VideoIntelligenceError) as excinfo:
        prov._call_vlm_with_frames("s", "p", [
            ExtractedFrame(frame_id="frame_0000", timestamp_us=0, data=b"x"),
        ])
    assert excinfo.value.code == VideoIntelligenceErrorCode.VISUAL_PROVIDER_UNSUPPORTED


# --- error taxonomy -----------------------------------------------------------

def test_error_taxonomy_timeout(clean_settings):
    monkey = clean_settings
    monkey.setattr(settings, "vision_provider", "nvidia")
    monkey.setattr(settings, "nvidia_api_key", "k")
    prov = _provider()
    err = prov._map_vlm_error("nvidia", Exception("x"))
    assert err.code == VideoIntelligenceErrorCode.INVALID_OUTPUT

    class APITimeoutError(Exception):
        pass

    err = prov._map_vlm_error("nvidia", APITimeoutError("slow"))
    assert err.code == VideoIntelligenceErrorCode.TIMEOUT

    class APIStatusError(Exception):
        status_code = 503

    err = prov._map_vlm_error("nvidia", APIStatusError("overloaded"))
    assert err.code == VideoIntelligenceErrorCode.PROVIDER_UNAVAILABLE

    class AuthErr(Exception):
        status_code = 401

    err = prov._map_vlm_error("nvidia", AuthErr("bad key"))
    assert err.code == VideoIntelligenceErrorCode.MISSING_PROVIDER_CREDENTIALS


def test_error_never_leaks_key(clean_settings):
    monkey = clean_settings
    monkey.setattr(settings, "vision_provider", "nvidia")
    monkey.setattr(settings, "nvidia_api_key", "sk-super-secret-KEY-abc123")
    prov = _provider()
    for exc in (
        Exception("boom"),
        type("APITimeoutError", (Exception,), {} )("t"),
    ):
        mapped = prov._map_vlm_error("nvidia", exc)
        assert "sk-super-secret" not in mapped.message
        assert "abc123" not in mapped.message


# --- fresh install defaults ---------------------------------------------------

def test_fresh_install_defaults_never_paid():
    from pathlib import Path

    env_example = (Path(__file__).resolve().parent.parent / ".env.example").read_text(
        encoding="utf-8"
    )
    assert "LLM_PROVIDER=naive" in env_example
    # .env.example no incluye claves reales ni activa provider de pago por defecto
    assert "ANTHROPIC_API_KEY=" in env_example
    assert "OPENAI_API_KEY=" in env_example
    assert "NVIDIA_API_KEY=" in env_example
    assert not any(
        line.strip() and not line.startswith("#") and "=" in line and line.split("=", 1)[1].strip()
        for line in env_example.splitlines()
        if line.startswith(("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NVIDIA_API_KEY"))
    ), "fresh install no debe tener keys pagadas activas"


def test_rerank_and_settings_choices_match(clean_settings):
    # choices listados siguen siendo solo suscripcion + free-tier
    assert set(vision.VISION_PROVIDER_CHOICES) == {"auto", "claude_code", "codex", "nvidia"}
    assert "claude" not in vision.VISION_PROVIDER_CHOICES
    assert "openai" not in vision.VISION_PROVIDER_CHOICES
