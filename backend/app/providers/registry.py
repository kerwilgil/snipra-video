"""Selección del proveedor de detección y structured completion según configuración.

F1: solo `naive`. F2 añade claude / openai / nvidia.
"""

from __future__ import annotations

from app.providers.base import ClipDetector
from app.providers.naive import NaiveDetector
from app.providers.structured_completion import StructuredCompletionAdapter

DEFAULT_MODELS = {
    "claude": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
    "nvidia": "auto",
    # Suscripción (usan el CLI local, no API key): consumen la cuota Pro/Max o Plus/Pro.
    "claude_code": "claude-sonnet-5",
    "codex": "gpt-5.6-terra",
    "naive": "",
}

# Proveedores que usan la suscripción vía CLI local en vez de una API key.
SUBSCRIPTION_PROVIDERS = frozenset({"claude_code", "codex"})


def get_detector(provider: str | None = None, model: str | None = None) -> ClipDetector:
    name = (provider or "naive").lower()

    if name in ("naive", ""):
        return NaiveDetector()

    # F2: proveedores con IA (import perezoso para no exigir SDK/keys si no se usan)
    if name == "claude":
        from app.providers.claude import ClaudeDetector

        return ClaudeDetector(model=model or DEFAULT_MODELS["claude"])
    if name == "openai":
        from app.providers.openai import OpenAIDetector

        return OpenAIDetector(model=model or DEFAULT_MODELS["openai"])
    if name == "nvidia":
        from app.providers.nvidia import NvidiaDetector

        return NvidiaDetector(model=model or DEFAULT_MODELS["nvidia"])
    if name == "claude_code":
        from app.providers.cli_subscription import ClaudeCodeDetector

        return ClaudeCodeDetector(model=model or DEFAULT_MODELS["claude_code"])
    if name == "codex":
        from app.providers.cli_subscription import CodexDetector

        return CodexDetector(model=model or DEFAULT_MODELS["codex"])

    # desconocido → fallback seguro
    return NaiveDetector()


def get_structured_completion_adapter(
    provider: str | None = None, model: str | None = None
) -> StructuredCompletionAdapter | None:
    """Get a structured completion adapter for the given provider/model.

    Returns None for providers that don't support structured completion.
    Callers should check for None and fall back to legacy _complete path.
    """
    name = (provider or "naive").lower()

    if name in ("naive", ""):
        return None

    if name == "claude":
        # Claude doesn't implement native structured completion yet
        return None
    if name == "openai":
        from app.providers.openai import OpenAIDetector

        return OpenAIDetector(model=model or DEFAULT_MODELS["openai"])
    if name == "nvidia":
        from app.providers.nvidia import NvidiaDetector

        return NvidiaDetector(model=model or DEFAULT_MODELS["nvidia"])
    if name == "claude_code":
        # CLI subscription providers don't implement structured completion
        return None
    if name == "codex":
        return None

    # desconocido → no soporta structured completion
    return None


def get_provider_capabilities(provider: str | None = None, model: str | None = None):
    """Get capabilities for a provider/model from the registry."""
    from app.providers.capabilities import get_provider_capabilities

    return get_provider_capabilities(provider, model)


def get_cost_class(provider: str | None) -> str:
    """Get cost class for a provider (backward compatible)."""
    from app.providers.capabilities import cost_class

    return cost_class(provider)
