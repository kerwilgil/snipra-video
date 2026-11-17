"""Tests del cierre final N.1: defaults residuales que podian caer en paid API.

Cubre: Job/TrackedChannel defaults, _normalize_provider_model fallback,
coercion runtime de llm_provider y compatibilidad de los 6 providers validos.
0 llamadas reales a APIs.
"""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.core.runtime_settings import _coerce_value, apply_runtime_settings
from app.models import Job, TrackedChannel


def test_job_default_provider_is_naive():
    """Job() sin provider explicito -> naive, NUNCA una API pagada."""
    job = Job()
    assert job.provider == "naive"


def test_tracked_channel_default_provider_is_naive():
    """TrackedChannel() sin provider explicito -> naive, NUNCA una API pagada."""
    channel = TrackedChannel()
    assert channel.provider == "naive"


def test_normalize_provider_model_empty_falls_to_naive(monkeypatch):
    from app.api import jobs

    monkeypatch.setattr(jobs.settings, "llm_provider", "")
    monkeypatch.setattr(jobs.settings, "llm_model", "")
    provider, _model = jobs._normalize_provider_model(None, None)
    assert provider == "naive"


def test_normalize_provider_model_invalid_settings_fail_safe(monkeypatch):
    from app.api import jobs

    monkeypatch.setattr(jobs.settings, "llm_provider", "proveedor-inventado")
    monkeypatch.setattr(jobs.settings, "llm_model", "")
    provider, _model = jobs._normalize_provider_model(None, None)
    assert provider == "naive"
    assert provider != "claude"


def test_normalize_provider_model_invalid_does_not_alias_to_claude(monkeypatch):
    """Un provider de request invalido no puede re-deducirse a Claude."""
    from app.api import jobs

    monkeypatch.setattr(jobs.settings, "llm_provider", "naive")
    provider, _model = jobs._normalize_provider_model("paywhaa", None)
    assert not provider.startswith("claude")


def test_runtime_llm_provider_empty_and_invalid_fail_to_naive():
    assert _coerce_value("llm_provider", "") == "naive"
    assert _coerce_value("llm_provider", None) == "naive"
    assert _coerce_value("llm_provider", "proveedor-inventado") == "naive"


@pytest.mark.parametrize(
    "provider", ["naive", "claude", "openai", "nvidia", "claude_code", "codex"]
)
def test_runtime_llm_provider_valid_values_preserved(provider):
    assert _coerce_value("llm_provider", provider) == provider


def test_anthropic_key_alone_never_selects_claude(monkeypatch):
    """Key presente pero provider vacio/invalido -> nunca Claude; la key no manda."""
    from app.api import jobs

    monkeypatch.setattr(jobs.settings, "anthropic_api_key", "sk-fake-solo-para-test")
    monkeypatch.setattr(jobs.settings, "llm_provider", "")
    monkeypatch.setattr(jobs.settings, "llm_model", "")
    provider, _ = jobs._normalize_provider_model(None, None)
    assert provider == "naive"


def test_apply_runtime_settings_invalid_provider_safe(monkeypatch, tmp_path):
    """Persistir llm_provider invalido no deja estado pagado activo."""
    monkeypatch.setattr(settings, "llm_provider", "naive")
    apply_runtime_settings({"llm_provider": "pagada-sorpresa"}, persist=False)
    assert settings.llm_provider == "naive"


# --- Compatibilidad explicita -------------------------------------------------


def test_explicit_claude_still_selects_claude(monkeypatch):
    from app.api import jobs

    monkeypatch.setattr(jobs.settings, "llm_provider", "claude")
    monkeypatch.setattr(jobs.settings, "llm_model", "claude-haiku-4-5")
    provider, model = jobs._normalize_provider_model(None, None)
    assert provider == "claude"
    assert model == "claude-haiku-4-5"


def test_explicit_request_provider_wins(monkeypatch):
    from app.api import jobs

    monkeypatch.setattr(jobs.settings, "llm_provider", "claude")
    provider, _ = jobs._normalize_provider_model("codex", None)
    assert provider == "codex"
