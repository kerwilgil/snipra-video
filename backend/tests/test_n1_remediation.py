"""Tests de remediacion N.1: defaults seguros, sanitizacion de secretos y
politica de costo exacta. Todo con claves FALSAS; 0 llamadas reales."""

from __future__ import annotations

from app.core import config as config_mod
from app.core.config import Settings
from app.core.sanitize import REDACTED, sanitize_error_text
from app.providers import vision
from app.providers.costs import is_paid_api


class TestSafeDefaults:
    def test_settings_default_is_naive(self, monkeypatch, tmp_path):
        # Sin .env y sin variables AI: provider = naive (local, gratis).
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        s = Settings(_env_file=str(tmp_path / "no-existe.env"))
        assert s.llm_provider == "naive"

    def test_anthropic_key_alone_does_not_select_claude(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-key-para-test")
        s = Settings(_env_file=str(tmp_path / "no-existe.env"))
        assert s.llm_provider == "naive"
        assert s.anthropic_api_key == "sk-fake-key-para-test"

    def test_explicit_llm_provider_claude_respected(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LLM_PROVIDER", "claude")
        s = Settings(_env_file=str(tmp_path / "no-existe.env"))
        assert s.llm_provider == "claude"

    def test_env_example_uses_naive(self):
        from pathlib import Path

        text = (
            Path(config_mod.__file__).parent.parent.parent / ".env.example"
        ).read_text(encoding="utf-8")
        assert "LLM_PROVIDER=naive" in text

    def test_no_paid_api_selected_by_defaults(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("VISION_PROVIDER", raising=False)
        s = Settings(_env_file=str(tmp_path / "no-existe.env"))
        assert not is_paid_api(s.llm_provider)
        # vision auto sobre texto naive nunca produce un proveedor pagado
        provider, _model = vision.resolve_vision(s.llm_provider, s.llm_model, via=True)
        assert provider == ""


class TestSecretSanitization:
    def test_redacts_each_configured_key(self, monkeypatch):
        from app.core.config import settings

        fakes = {
            "anthropic_api_key": "sk-test-anthropic-FAKEAAA111",
            "openai_api_key": "sk-test-openai-FAKEBBB222",
            "nvidia_api_key": "nvapi-test-nvidia-FAKECCC333",
        }
        for field, value in fakes.items():
            monkeypatch.setattr(settings, field, value)

        for value in fakes.values():
            raw = f"authorization failed for {value} upstream"
            cleaned = sanitize_error_text(raw)
            assert value not in cleaned
            assert REDACTED in cleaned

    def test_sdk_exception_message_sanitized(self, monkeypatch):
        from app.core.config import settings
        from app.video_intelligence.providers import ExistingSnipraVLMProvider

        monkeypatch.setattr(settings, "nvidia_api_key", "nvapi-fake-secret-zzz999")
        fake = Exception("upstream error: invalid key nvapi-fake-secret-zzz999")
        err = ExistingSnipraVLMProvider._map_vlm_error("nvidia", fake)
        assert "nvapi-fake-secret-zzz999" not in err.message
        assert REDACTED in err.message
        assert "nvidia" in err.message

    def test_cli_stderr_sanitized(self, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "openai_api_key", "sk-cli-stderr-FAKE000")
        cleaned = sanitize_error_text("401 Unauthorized: key sk-cli-stderr-FAKE000")
        assert "sk-cli-stderr-FAKE000" not in cleaned
        assert REDACTED in cleaned

    def test_agent_propagation_sanitized(self, monkeypatch):
        from app.core.config import settings
        from app.video_intelligence.models import (
            VideoIntelligenceError,
            VideoIntelligenceErrorCode,
        )

        monkeypatch.setattr(settings, "anthropic_api_key", "sk-agent-fake-QWE999")
        exc = VideoIntelligenceError(
            VideoIntelligenceErrorCode.PROVIDER_UNAVAILABLE,
            "upstream dijo: bad key sk-agent-fake-QWE999",
        )
        # simula la rama de propagacion del agent (agent.py -> RuntimeError)
        try:
            raise RuntimeError(
                f"AGENT_VIA_EVIDENCE_UNAVAILABLE: {exc.code.value}: "
                f"{sanitize_error_text(exc.message)}"
            ) from exc
        except RuntimeError as err:
            text = str(err)
            assert "sk-agent-fake-QWE999" not in text
            assert REDACTED in text
            assert REDACTED in text

    def test_no_secrets_to_sanitize_is_passthrough(self, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "anthropic_api_key", "")
        monkeypatch.setattr(settings, "openai_api_key", "")
        monkeypatch.setattr(settings, "nvidia_api_key", "")
        raw = "Codex CLI fallo (1): quota agotada"
        assert sanitize_error_text(raw) == raw


class TestCostPolicy:
    def test_auto_never_paid_from_subscription_codex(self, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "vision_provider", "auto")
        p, _ = vision.resolve_vision("codex", "gpt-5.6-terra", via=True)
        assert p == "codex" and not is_paid_api(p)

    def test_auto_never_paid_from_claude_code(self, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "vision_provider", "auto")
        p, _ = vision.resolve_vision("claude_code", "claude-sonnet-5", via=True)
        assert p == "claude_code" and not is_paid_api(p)

    def test_naive_auto_fail_closed(self, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "vision_provider", "auto")
        p, _ = vision.resolve_vision("naive", "", via=True)
        assert p == ""

    def test_paid_text_explicit_keeps_paid_vision(self, monkeypatch):
        from app.core.config import settings

        # el usuario eligio deliberadamente una API pagada como proveedor de texto
        monkeypatch.setattr(settings, "vision_provider", "auto")
        p, _ = vision.resolve_vision("claude", "claude-haiku-4-5", via=True)
        assert p == "claude"

    def test_paid_vision_only_when_explicit(self, monkeypatch):
        from app.core.config import settings

        # nunca sale "por auto": un usuario con texto gratis/suscripcion no paga
        monkeypatch.setattr(settings, "vision_provider", "auto")
        for text in ("codex", "claude_code", "naive", "nvidia"):
            p, _ = vision.resolve_vision(text, "", via=True)
            assert not is_paid_api(p), f"auto desde {text} no debe dar proveedor pagado"

    def test_no_guard_pass_left(self):
        import inspect

        import app.video_intelligence.providers as vp

        src = inspect.getsource(vp.ExistingSnipraVLMProvider._ensure_detector)
        assert "pass" not in [line.strip() for line in src.splitlines() if line.strip() == "pass"]
