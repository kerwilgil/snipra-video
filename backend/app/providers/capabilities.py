"""Provider capabilities and cost classification for structured completion."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass


class CostClass(StrEnum):
    """Economic cost class for a provider/model. Authoritative at runtime."""

    LOCAL = "local"  # No cost, runs locally (naive, Real-ESRGAN, FFmpeg)
    FREE_TIER = "free_tier"  # Free tier cloud (NVIDIA NIM free endpoints)
    SUBSCRIPTION = "subscription"  # User's existing CLI subscription (claude_code, codex)
    PAID_API = "paid_api"  # Paid per-call API (OpenAI, Anthropic, Runway, Luma)
    UNKNOWN = "unknown"  # Unknown provider


class ProviderCapability(StrEnum):
    """Capability flags for a provider/model."""

    NATIVE_JSON_SCHEMA = "native_json_schema"
    JSON_MODE = "json_mode"
    TOOLS = "tools"
    REASONING = "reasoning"


class ProviderCapabilities(BaseModel):
    """Capabilities of a specific provider/model combination."""

    provider: str
    model: str
    supports_native_json_schema: bool = False
    supports_json_mode: bool = False
    supports_tools: bool = False
    supports_reasoning: bool = False
    cost_class: CostClass = CostClass.UNKNOWN

    def has(self, capability: ProviderCapability | str) -> bool:
        """Check if a specific capability is supported."""
        if isinstance(capability, str):
            capability = ProviderCapability(capability)
        return getattr(self, f"supports_{capability.value}", False)


class CompletionResult(BaseModel):
    """Result of a structured completion call."""

    content: str
    provider: str
    model: str
    structured: bool = False  # True if native JSON schema was used
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    reasoning_present: bool = False
    metadata: dict = Field(default_factory=dict)


# --- Registry of known provider capabilities ---
# This is the SINGLE SOURCE OF TRUTH for provider capabilities.
# Update this when adding new providers/models.

PROVIDER_CAPABILITIES_REGISTRY: dict[str, dict[str, ProviderCapabilities]] = {
    "nvidia": {
        "nvidia/nemotron-3.5-lightning-30b-a3b": ProviderCapabilities(
            provider="nvidia",
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            supports_native_json_schema=False,  # NOT QUALIFIED: reasoning output in content
            supports_json_mode=True,
            supports_tools=False,
            supports_reasoning=True,
            cost_class=CostClass.FREE_TIER,
        ),
        "nvidia/nemotron-3-super-120b-a12b": ProviderCapabilities(
            provider="nvidia",
            model="nvidia/nemotron-3-super-120b-a12b",
            supports_native_json_schema=True,  # QUALIFIED: non-reasoning model works perfectly
            supports_json_mode=True,
            supports_tools=False,
            supports_reasoning=False,
            cost_class=CostClass.FREE_TIER,
        ),
        "nvidia/nemotron-3-ultra-550b-a55b": ProviderCapabilities(
            provider="nvidia",
            model="nvidia/nemotron-3-ultra-550b-a55b",
            supports_native_json_schema=False,  # NOT QUALIFIED: endpoint instability
            supports_json_mode=True,
            supports_tools=False,
            supports_reasoning=True,
            cost_class=CostClass.FREE_TIER,
        ),
        "moonshotai/kimi-k3": ProviderCapabilities(
            provider="nvidia",
            model="moonshotai/kimi-k3",
            supports_native_json_schema=False,
            supports_json_mode=False,
            supports_tools=False,
            supports_reasoning=True,
            cost_class=CostClass.FREE_TIER,
        ),
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": ProviderCapabilities(
            provider="nvidia",
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            supports_native_json_schema=False,
            supports_json_mode=False,
            supports_tools=False,
            supports_reasoning=True,
            cost_class=CostClass.FREE_TIER,
        ),
        "meta/llama-3.2-11b-vision-instruct": ProviderCapabilities(
            provider="nvidia",
            model="meta/llama-3.2-11b-vision-instruct",
            supports_native_json_schema=False,
            supports_json_mode=False,
            supports_tools=False,
            supports_reasoning=False,
            cost_class=CostClass.FREE_TIER,
        ),
        "meta/llama-3.2-90b-vision-instruct": ProviderCapabilities(
            provider="nvidia",
            model="meta/llama-3.2-90b-vision-instruct",
            supports_native_json_schema=False,
            supports_json_mode=False,
            supports_tools=False,
            supports_reasoning=False,
            cost_class=CostClass.FREE_TIER,
        ),
    },
    "openai": {
        "gpt-4o": ProviderCapabilities(
            provider="openai",
            model="gpt-4o",
            supports_native_json_schema=True,
            supports_json_mode=True,
            supports_tools=True,
            supports_reasoning=False,
            cost_class=CostClass.PAID_API,
        ),
        "gpt-4o-mini": ProviderCapabilities(
            provider="openai",
            model="gpt-4o-mini",
            supports_native_json_schema=True,
            supports_json_mode=True,
            supports_tools=True,
            supports_reasoning=False,
            cost_class=CostClass.PAID_API,
        ),
    },
    "claude": {
        "claude-haiku-4-5": ProviderCapabilities(
            provider="claude",
            model="claude-haiku-4-5",
            supports_native_json_schema=False,
            supports_json_mode=False,
            supports_tools=False,
            supports_reasoning=False,
            cost_class=CostClass.PAID_API,
        ),
    },
    "claude_code": {
        "claude-sonnet-5": ProviderCapabilities(
            provider="claude_code",
            model="claude-sonnet-5",
            supports_native_json_schema=False,
            supports_json_mode=False,
            supports_tools=False,
            supports_reasoning=False,
            cost_class=CostClass.SUBSCRIPTION,
        ),
    },
    "codex": {
        "gpt-5.6-terra": ProviderCapabilities(
            provider="codex",
            model="gpt-5.6-terra",
            supports_native_json_schema=False,
            supports_json_mode=False,
            supports_tools=False,
            supports_reasoning=False,
            cost_class=CostClass.SUBSCRIPTION,
        ),
    },
    "naive": {
        "": ProviderCapabilities(
            provider="naive",
            model="",
            supports_native_json_schema=False,
            supports_json_mode=False,
            supports_tools=False,
            supports_reasoning=False,
            cost_class=CostClass.LOCAL,
        ),
    },
}


def get_provider_capabilities(provider: str, model: str) -> ProviderCapabilities:
    """Get capabilities for a provider/model combination.

    Only exact registered model matches return registered capabilities.
    Unknown models get conservative defaults with provider's cost class.
    Unknown providers get fully conservative defaults.
    """
    provider = (provider or "").strip().lower()
    model = (model or "").strip().lower()

    # Handle empty/None provider as naive
    if not provider:
        provider = "naive"

    if provider in PROVIDER_CAPABILITIES_REGISTRY:
        registry = PROVIDER_CAPABILITIES_REGISTRY[provider]
        if model in registry:
            return registry[model]

        # Unknown model for known provider: conservative defaults with provider's cost class
        return ProviderCapabilities(
            provider=provider,
            model=model,
            supports_native_json_schema=False,
            supports_json_mode=False,
            supports_tools=False,
            supports_reasoning=False,
            cost_class=_provider_cost_class(provider),
        )

    # Unknown provider: fully conservative
    return ProviderCapabilities(
        provider=provider,
        model=model,
        supports_native_json_schema=False,
        supports_json_mode=False,
        supports_tools=False,
        supports_reasoning=False,
        cost_class=CostClass.UNKNOWN,
    )


def _provider_cost_class(provider: str) -> CostClass:
    """Get cost class for a provider (private helper)."""
    if provider in PROVIDER_CAPABILITIES_REGISTRY:
        caps = next(iter(PROVIDER_CAPABILITIES_REGISTRY[provider].values()))
        return caps.cost_class
    return CostClass.UNKNOWN


def cost_class(provider: str | None) -> CostClass:
    """Get cost class for a provider (backward compatible)."""
    return _provider_cost_class(provider)


def is_paid_api(provider: str | None) -> bool:
    return cost_class(provider) == CostClass.PAID_API


def is_subscription(provider: str | None) -> bool:
    return cost_class(provider) == CostClass.SUBSCRIPTION


def is_free_tier(provider: str | None) -> bool:
    return cost_class(provider) == CostClass.FREE_TIER


def is_local(provider: str | None) -> bool:
    return cost_class(provider) == CostClass.LOCAL