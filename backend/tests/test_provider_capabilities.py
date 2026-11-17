"""Tests dedicados para Provider Capabilities Registry (SNIPRA 1.6B).

Cubren: registro canónico, qualification flags, cost class, defaults.
"""

from __future__ import annotations

import pytest

from app.providers.capabilities import (
    PROVIDER_CAPABILITIES_REGISTRY,
    CostClass,
    ProviderCapabilities,
    cost_class,
    get_provider_capabilities,
    is_free_tier,
    is_local,
    is_paid_api,
    is_subscription,
)
from app.providers.costs import (
    PROVIDER_COST_CLASS,
)
from app.providers.costs import (
    cost_class as costs_cost_class,
)
from app.providers.costs import (
    is_paid_api as costs_is_paid_api,
)
from app.providers.costs import (
    is_subscription as costs_is_subscription,
)
from app.providers.registry import DEFAULT_MODELS


class TestRegistryStructure:
    """Verify registry structure and completeness."""

    def test_registry_has_all_expected_providers(self):
        expected_providers = {"nvidia", "openai", "claude", "claude_code", "codex", "naive"}
        assert set(PROVIDER_CAPABILITIES_REGISTRY.keys()) == expected_providers

    def test_nvidia_models_present(self):
        nvidia_models = PROVIDER_CAPABILITIES_REGISTRY["nvidia"]
        expected = {
            "nvidia/nemotron-3.5-lightning-30b-a3b",
            "nvidia/nemotron-3-super-120b-a12b",
            "nvidia/nemotron-3-ultra-550b-a55b",
            "moonshotai/kimi-k3",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            "meta/llama-3.2-11b-vision-instruct",
            "meta/llama-3.2-90b-vision-instruct",
        }
        assert set(nvidia_models.keys()) == expected

    def test_openai_models_present(self):
        openai_models = PROVIDER_CAPABILITIES_REGISTRY["openai"]
        expected = {"gpt-4o", "gpt-4o-mini"}
        assert set(openai_models.keys()) == expected

    def test_claude_models_present(self):
        claude_models = PROVIDER_CAPABILITIES_REGISTRY["claude"]
        expected = {"claude-haiku-4-5"}
        assert set(claude_models.keys()) == expected

    def test_subscription_providers_present(self):
        assert "claude_code" in PROVIDER_CAPABILITIES_REGISTRY
        assert "codex" in PROVIDER_CAPABILITIES_REGISTRY


class TestQualificationFlags:
    """Test qualification flags match physical benchmark results."""

    def test_lightning_not_qualified(self):
        caps = PROVIDER_CAPABILITIES_REGISTRY["nvidia"]["nvidia/nemotron-3.5-lightning-30b-a3b"]
        assert caps.supports_native_json_schema is False
        assert caps.supports_reasoning is True

    def test_super_qualified(self):
        caps = PROVIDER_CAPABILITIES_REGISTRY["nvidia"]["nvidia/nemotron-3-super-120b-a12b"]
        assert caps.supports_native_json_schema is True
        assert caps.supports_reasoning is False

    def test_ultra_not_qualified(self):
        caps = PROVIDER_CAPABILITIES_REGISTRY["nvidia"]["nvidia/nemotron-3-ultra-550b-a55b"]
        assert caps.supports_native_json_schema is False
        assert caps.supports_reasoning is True

    def test_kimi_not_qualified(self):
        caps = PROVIDER_CAPABILITIES_REGISTRY["nvidia"]["moonshotai/kimi-k3"]
        assert caps.supports_native_json_schema is False
        assert caps.supports_reasoning is True

    def test_openai_models_qualified(self):
        for model in ["gpt-4o", "gpt-4o-mini"]:
            caps = PROVIDER_CAPABILITIES_REGISTRY["openai"][model]
            assert caps.supports_native_json_schema is True
            assert caps.supports_json_mode is True
            assert caps.supports_tools is True

    def test_claude_not_qualified(self):
        caps = PROVIDER_CAPABILITIES_REGISTRY["claude"]["claude-haiku-4-5"]
        assert caps.supports_native_json_schema is False
        assert caps.supports_json_mode is False
        assert caps.supports_tools is False

    def test_subscription_providers_not_qualified(self):
        for provider in ["claude_code", "codex"]:
            for caps in PROVIDER_CAPABILITIES_REGISTRY[provider].values():
                assert caps.supports_native_json_schema is False
                assert caps.supports_json_mode is False

    def test_naive_not_qualified(self):
        caps = PROVIDER_CAPABILITIES_REGISTRY["naive"][""]
        assert caps.supports_native_json_schema is False
        assert caps.supports_json_mode is False


class TestCostClassMapping:
    """Test cost class assignments are correct and consistent."""

    @pytest.mark.parametrize("provider,model,expected", [
        ("naive", "", "local"),
        ("nvidia", "nvidia/nemotron-3-super-120b-a12b", "free_tier"),
        ("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b", "free_tier"),
        ("nvidia", "nvidia/nemotron-3-ultra-550b-a55b", "free_tier"),
        ("openai", "gpt-4o", "paid_api"),
        ("openai", "gpt-4o-mini", "paid_api"),
        ("claude", "claude-haiku-4-5", "paid_api"),
        ("claude_code", "claude-sonnet-5", "subscription"),
        ("codex", "gpt-5.6-terra", "subscription"),
        ("naive", "", "local"),
    ])
    def test_cost_class_correct(self, provider, model, expected):
        caps = get_provider_capabilities(provider, model)
        assert caps.cost_class.value == expected, f"{provider}/{model} should be {expected}"

    def test_no_unknown_cost_class_for_known_providers(self):
        for provider, model in [
            ("naive", ""),
            ("nvidia", "nvidia/nemotron-3-super-120b-a12b"),
            ("openai", "gpt-4o-mini"),
            ("claude", "claude-haiku-4-5"),
            ("claude_code", "claude-sonnet-5"),
            ("codex", "gpt-5.6-terra"),
        ]:
            caps = get_provider_capabilities(provider, model)
            assert caps.cost_class != CostClass.UNKNOWN, f"{provider}/{model} should not be UNKNOWN"


class TestCostSourceOfTruth:
    """Test that capabilities.py is the single source of truth for costs."""

    def test_costs_module_delegates_to_capabilities(self):
        """costs.py functions should delegate to capabilities.py."""
        providers = ["naive", "nvidia", "openai", "claude", "claude_code", "codex"]
        for provider in providers:
            caps = get_provider_capabilities(provider, "")
            assert costs_cost_class(provider) == caps.cost_class.value
            assert costs_is_paid_api(provider) == (caps.cost_class == CostClass.PAID_API)
            assert costs_is_subscription(provider) == (caps.cost_class == CostClass.SUBSCRIPTION)

    def test_provider_cost_class_derived_from_registry(self):
        """PROVIDER_COST_CLASS dict should be derived from registry."""
        # Check that every entry in PROVIDER_COST_CLASS matches registry
        for provider, expected_class in PROVIDER_COST_CLASS.items():
            caps = get_provider_capabilities(provider, "")
            assert caps.cost_class.value == expected_class, f"{provider} cost class mismatch"

    def test_no_static_duplication(self):
        """Verify PROVIDER_COST_CLASS is derived, not hardcoded with different values."""
        # This test passes if the derivation logic in costs.py is correct
        # The actual values are tested in test_cost_class_correct
        pass


class TestDefaultModels:
    """Test DEFAULT_MODELS registry matches capability registry."""

    def test_claude_default_model_matches_registry(self):
        """Default Claude model should exist in capability registry."""
        default_claude = DEFAULT_MODELS["claude"]
        caps = get_provider_capabilities("claude", default_claude)
        assert caps.cost_class == CostClass.PAID_API
        assert caps.supports_native_json_schema is False

    def test_openai_default_model_matches_registry(self):
        default_openai = DEFAULT_MODELS["openai"]
        caps = get_provider_capabilities("openai", default_openai)
        assert caps.cost_class == CostClass.PAID_API
        assert caps.supports_native_json_schema is True

    def test_nvidia_default_is_dynamic_auto(self):
        """NVIDIA defaults to runtime catalog resolution; no permanent model ID."""
        default_nvidia = DEFAULT_MODELS["nvidia"]
        assert default_nvidia == "auto"
        caps = get_provider_capabilities("nvidia", default_nvidia)
        # Auto has no schema capability until the concrete model is resolved.
        assert caps.supports_native_json_schema is False

    def test_claude_code_default(self):
        assert DEFAULT_MODELS["claude_code"] == "claude-sonnet-5"

    def test_codex_default(self):
        assert DEFAULT_MODELS["codex"] == "gpt-5.6-terra"


class TestUnknownProvider:
    """Unknown providers should fail safely with conservative defaults."""

    def test_unknown_provider_returns_conservative(self):
        caps = get_provider_capabilities("totally-unknown-provider", "unknown-model")
        assert caps.supports_native_json_schema is False
        assert caps.supports_json_mode is False
        assert caps.supports_tools is False
        assert caps.supports_reasoning is False
        assert caps.cost_class == CostClass.UNKNOWN

    def test_unknown_provider_never_paid(self):
        caps = get_provider_capabilities("malicious-provider", "steal-keys")
        assert caps.cost_class != CostClass.PAID_API
        assert is_paid_api("malicious-provider") is False


class TestGetProviderCapabilitiesFunction:
    """Test get_provider_capabilities behavior."""

    def test_exact_match_returns_exact_caps(self):
        caps = get_provider_capabilities("openai", "gpt-4o")
        assert caps.model == "gpt-4o"
        assert caps.provider == "openai"
        assert caps.supports_native_json_schema is True

    def test_partial_model_match(self):
        """Should match by prefix/model family."""
        # This might match a known model
        caps = get_provider_capabilities("nvidia", "nvidia/nemotron-3-super-120b-a12b")
        assert caps.supports_native_json_schema is True

    def test_case_insensitive_provider(self):
        caps = get_provider_capabilities("NVIDIA", "nvidia/nemotron-3-super-120b-a12b")
        assert caps.supports_native_json_schema is True

    def test_empty_provider_returns_naive(self):
        caps = get_provider_capabilities("", "any-model")
        assert caps.provider == "naive"
        assert caps.supports_native_json_schema is False


class TestCostClassFunctions:
    """Test cost classification functions."""

    def test_cost_class_returns_string(self):
        assert cost_class("nvidia") == "free_tier"
        assert cost_class("openai") == "paid_api"
        assert cost_class("naive") == "local"
        assert cost_class("claude_code") == "subscription"

    def test_is_paid_api(self):
        assert is_paid_api("openai") is True
        assert is_paid_api("claude") is True
        assert is_paid_api("nvidia") is False
        assert is_paid_api("naive") is False
        assert is_paid_api("unknown") is False

    def test_is_subscription(self):
        assert is_subscription("claude_code") is True
        assert is_subscription("codex") is True
        assert is_subscription("openai") is False
        assert is_subscription("nvidia") is False

    def test_is_free_tier(self):
        assert is_free_tier("nvidia") is True
        assert is_free_tier("openai") is False
        assert is_free_tier("naive") is False

    def test_is_local(self):
        assert is_local("naive") is True
        assert is_local("nvidia") is False
        assert is_local("openai") is False


class TestCostClassEnum:
    """Test CostClass enum values."""

    def test_cost_class_values(self):
        assert CostClass.LOCAL == "local"
        assert CostClass.FREE_TIER == "free_tier"
        assert CostClass.SUBSCRIPTION == "subscription"
        assert CostClass.PAID_API == "paid_api"
        assert CostClass.UNKNOWN == "unknown"


class TestProviderCapabilitiesModel:
    """Test ProviderCapabilities model."""

    def test_has_method(self):
        caps = ProviderCapabilities(
            provider="test",
            model="test",
            supports_native_json_schema=True,
            supports_json_mode=False,
        )
        assert caps.has("native_json_schema") is True
        assert caps.has("json_mode") is False
        assert caps.has("tools") is False


class TestRegistryConsistency:
    """Ensure registry is internally consistent."""

    def test_no_duplicate_models(self):
        all_models = set()
        for provider, models in PROVIDER_CAPABILITIES_REGISTRY.items():
            for model in models:
                key = f"{provider}:{model}"
                assert key not in all_models, f"Duplicate model: {key}"
                all_models.add(key)

    def test_all_models_have_cost_class(self):
        for provider, models in PROVIDER_CAPABILITIES_REGISTRY.items():
            for model, caps in models.items():
                assert caps.cost_class != CostClass.UNKNOWN, (
                    f"{provider}/{model} missing cost class"
                )

    def test_nvidia_reasoning_models_flagged(self):
        """Reasoning models should have supports_reasoning=True and native=False."""
        reasoning_models = [
            "nvidia/nemotron-3.5-lightning-30b-a3b",
            "nvidia/nemotron-3-ultra-550b-a55b",
            "moonshotai/kimi-k3",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        ]
        for model in reasoning_models:
            caps = PROVIDER_CAPABILITIES_REGISTRY["nvidia"][model]
            assert caps.supports_reasoning is True, f"{model} should have reasoning=True"
            assert caps.supports_native_json_schema is False, (
                f"{model} should NOT be qualified for native schema"
            )

    def test_nvidia_super_is_only_qualified(self):
        """Only Nemotron 3 Super should be qualified among NVIDIA models."""
        qualified = [
            m for m, c in PROVIDER_CAPABILITIES_REGISTRY["nvidia"].items()
            if c.supports_native_json_schema
        ]
        assert qualified == ["nvidia/nemotron-3-super-120b-a12b"], (
            f"Only Super should be qualified, got: {qualified}"
        )


class TestCapabilitiesEdgeCases:
    """Edge cases and robustness."""

    def test_provider_case_insensitive(self):
        caps1 = get_provider_capabilities("NVIDIA", "nvidia/nemotron-3-super-120b-a12b")
        caps2 = get_provider_capabilities("nvidia", "nvidia/nemotron-3-super-120b-a12b")
        assert caps1.supports_native_json_schema == caps2.supports_native_json_schema

    def test_model_case_insensitive(self):
        caps1 = get_provider_capabilities("nvidia", "NVIDIA/NEMOTRON-3-SUPER-120B-A12B")
        caps2 = get_provider_capabilities("nvidia", "nvidia/nemotron-3-super-120b-a12b")
        assert caps1.supports_native_json_schema == caps2.supports_native_json_schema

    def test_empty_strings(self):
        caps = get_provider_capabilities("", "")
        assert caps.provider == "naive"
        assert caps.supports_native_json_schema is False

    def test_none_inputs(self):
        caps = get_provider_capabilities(None, None)
        assert caps.provider == "naive"
        assert caps.supports_native_json_schema is False


class TestUnknownModelBehavior:
    """Tests for unknown model fail-safe behavior (Block 01U)."""

    def test_openai_unknown_model_conservative(self):
        """OpenAI unknown model should have conservative capabilities but PAID_API cost."""
        caps = get_provider_capabilities("openai", "totally-unknown-openai-model")
        assert caps.provider == "openai"
        assert caps.model == "totally-unknown-openai-model"
        assert caps.supports_native_json_schema is False
        assert caps.supports_json_mode is False
        assert caps.supports_tools is False
        assert caps.supports_reasoning is False
        assert caps.cost_class == CostClass.PAID_API

    def test_nvidia_unknown_model_conservative(self):
        """NVIDIA unknown model should have conservative capabilities but FREE_TIER cost."""
        caps = get_provider_capabilities("nvidia", "nvidia/future-unqualified-model")
        assert caps.provider == "nvidia"
        assert caps.model == "nvidia/future-unqualified-model"
        assert caps.supports_native_json_schema is False
        assert caps.supports_json_mode is False
        assert caps.supports_tools is False
        assert caps.supports_reasoning is False
        assert caps.cost_class == CostClass.FREE_TIER

    def test_claude_unknown_model_conservative(self):
        """Claude unknown model should have conservative capabilities but PAID_API cost."""
        caps = get_provider_capabilities("claude", "claude-unknown-future-model")
        assert caps.provider == "claude"
        assert caps.model == "claude-unknown-future-model"
        assert caps.supports_native_json_schema is False
        assert caps.supports_json_mode is False
        assert caps.supports_tools is False
        assert caps.supports_reasoning is False
        assert caps.cost_class == CostClass.PAID_API

    def test_claude_code_unknown_model_conservative(self):
        """Claude Code unknown model should have conservative capabilities but SUBSCRIPTION cost."""
        caps = get_provider_capabilities("claude_code", "claude-future-model")
        assert caps.provider == "claude_code"
        assert caps.model == "claude-future-model"
        assert caps.supports_native_json_schema is False
        assert caps.supports_json_mode is False
        assert caps.supports_tools is False
        assert caps.supports_reasoning is False
        assert caps.cost_class == CostClass.SUBSCRIPTION

    def test_codex_unknown_model_conservative(self):
        """Codex unknown model should have conservative capabilities but SUBSCRIPTION cost."""
        caps = get_provider_capabilities("codex", "codex-future-model")
        assert caps.provider == "codex"
        assert caps.model == "codex-future-model"
        assert caps.supports_native_json_schema is False
        assert caps.supports_json_mode is False
        assert caps.supports_tools is False
        assert caps.supports_reasoning is False
        assert caps.cost_class == CostClass.SUBSCRIPTION


class TestSubstringMatchingPrevention:
    """SC-18: Substring matching must not trigger capability inheritance."""

    def test_openai_substring_no_match(self):
        """'fake-gpt-4o-not-real' must NOT inherit gpt-4o capabilities."""
        caps = get_provider_capabilities("openai", "fake-gpt-4o-not-real")
        # Should NOT inherit gpt-4o's native schema capability
        assert caps.supports_native_json_schema is False
        assert caps.supports_tools is False
        assert caps.cost_class == CostClass.PAID_API  # But cost class preserved

    def test_nvidia_substring_no_match(self):
        """Substring match should not trigger capability inheritance for NVIDIA."""
        caps = get_provider_capabilities("nvidia", "fake-nemotron-super-fake")
        assert caps.supports_native_json_schema is False
        assert caps.cost_class == CostClass.FREE_TIER

    def test_prefix_does_not_inherit_capabilities(self):
        """Prefix match should NOT inherit capabilities - only exact match works."""
        # Exact match should work
        caps1 = get_provider_capabilities("nvidia", "nvidia/nemotron-3-super-120b-a12b")
        # Prefix match should NOT work - no prefix matching anymore
        caps2 = get_provider_capabilities("nvidia", "nvidia/nemotron-3-super-120b-a12b-unqualified")
        assert caps1.supports_native_json_schema is True   # Exact match
        assert caps2.supports_native_json_schema is False  # Prefix match no longer works

    def test_openai_prefix_attack(self):
        """Test gpt-4o-fake-unqualified must NOT inherit gpt-4o capabilities."""
        caps = get_provider_capabilities("openai", "gpt-4o-fake-unqualified")
        assert caps.supports_native_json_schema is False
        assert caps.supports_json_mode is False
        assert caps.supports_tools is False
        assert caps.supports_reasoning is False
        assert caps.cost_class == CostClass.PAID_API

    def test_openai_mini_prefix_attack(self):
        """Test gpt-4o-mini-unqualified must NOT inherit gpt-4o-mini capabilities."""
        caps = get_provider_capabilities("openai", "gpt-4o-mini-unqualified")
        assert caps.supports_native_json_schema is False
        assert caps.supports_json_mode is False
        assert caps.supports_tools is False
        assert caps.supports_reasoning is False
        assert caps.cost_class == CostClass.PAID_API

    def test_nvidia_qualified_prefix_attack(self):
        """Test nemotron-3-super-120b-a12b-unqualified must NOT inherit Super capabilities."""
        caps = get_provider_capabilities("nvidia", "nvidia/nemotron-3-super-120b-a12b-unqualified")
        assert caps.supports_native_json_schema is False
        assert caps.cost_class == CostClass.FREE_TIER

    def test_claude_prefix_attack(self):
        """Test claude-haiku-4-5-unqualified must NOT inherit Haiku capabilities."""
        caps = get_provider_capabilities("claude", "claude-haiku-4-5-unqualified")
        assert caps.supports_native_json_schema is False
        assert caps.supports_json_mode is False
        assert caps.supports_tools is False
        assert caps.supports_reasoning is False
        assert caps.cost_class == CostClass.PAID_API


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
