"""Clases de costo por proveedor de IA — compatibility facade.

La política de costo es explícita y testeable. La fuente canónica es
`app.providers.capabilities`. Este módulo reexporta para compatibilidad.
"""

from __future__ import annotations

from app.providers.capabilities import (
    cost_class as _cost_class,
)
from app.providers.capabilities import (
    is_free_tier,
    is_local,
)
from app.providers.capabilities import (
    is_paid_api as _is_paid_api,
)
from app.providers.capabilities import (
    is_subscription as _is_subscription,
)


# Backward compatibility: expose the same function names
def cost_class(provider: str | None) -> str:
    return _cost_class(provider).value


def is_paid_api(provider: str | None) -> bool:
    return _is_paid_api(provider)


def is_subscription(provider: str | None) -> bool:
    return _is_subscription(provider)


def is_free_tier_class(provider: str | None) -> bool:
    return is_free_tier(provider)


def is_local_class(provider: str | None) -> bool:
    return is_local(provider)


# For backward compatibility with code that checks string values
# DERIVED from canonical registry in capabilities.py
PROVIDER_COST_CLASS: dict[str, str] = {
    provider: _cost_class(provider).value
    for provider in [
        "naive",
        "claude_code",
        "codex",
        "nvidia",
        "claude",
        "openai",
    ]
}