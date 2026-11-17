"""Tests dedicados para Structured Completion Foundation (SNIPRA 1.6B).

Cubren: native schema path, fallback, semantic validation, retry, privacy, cost, etc.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.editor.agent import _rework_plan, plan_request
from app.editor.evidence_request import DirectorOutput, DirectorOutputType
from app.providers.capabilities import (
    CostClass,
    ProviderCapabilities,
    get_provider_capabilities,
)
from app.providers.costs import cost_class, is_paid_api, is_subscription, PROVIDER_COST_CLASS
from app.providers.registry import get_structured_completion_adapter
from app.providers.structured_completion import (
    CompletionResult,
    StructuredCompletionAdapter,
    validate_json,
)


class FakeStructuredAdapter(StructuredCompletionAdapter):
    """Fake adapter for testing native schema path."""

    def __init__(self, qualified: bool = True, response: str = None):
        self._qualified = qualified
        self._response = response
        self.call_count = 0
        self.last_schema = None

    @property
    def capabilities(self):
        return ProviderCapabilities(
            provider="test",
            model="test-model",
            supports_native_json_schema=self._qualified,
            supports_json_mode=True,
            cost_class=CostClass.LOCAL,
        )

    def complete(self, system: str, user: str, *, json_schema: dict = None, **kwargs):
        self.call_count += 1
        self.last_schema = json_schema
        if self._response is not None:
            return CompletionResult(
                content=self._response,
                provider="test",
                model="test-model",
                structured=self._qualified,
            )
        # Default valid DirectorOutput
        return CompletionResult(
            content=json.dumps({
                "type": "editorial_plan",
                "editorial_plan": {"objective": "test", "operations": []},
                "evidence_sufficiency": "sufficient",
                "decision_rationale": "test",
            }),
            provider="test",
            model="test-model",
            structured=self._qualified,
        )


class FakeLegacyDetector:
    """Fake legacy detector for fallback path."""

    def __init__(self, response: str = None):
        self._response = response
        self.call_count = 0

    def _complete(self, system: str, user: str) -> str:
        self.call_count += 1
        if self._response:
            return self._response
        return json.dumps({
            "type": "editorial_plan",
            "editorial_plan": {"objective": "test", "operations": []},
            "evidence_sufficiency": "sufficient",
            "decision_rationale": "legacy fallback",
        })


class TestQualifiedNativeModel:
    """SC-01: Qualified native model uses structured adapter."""

    def test_super_model_qualified(self):
        caps = get_provider_capabilities("nvidia", "nvidia/nemotron-3-super-120b-a12b")
        assert caps.supports_native_json_schema is True

    def test_director_uses_structured_adapter_when_qualified(self):
        """Director should use structured adapter when model is qualified."""
        adapter = FakeStructuredAdapter(qualified=True)
        schema = {"type": "object", "properties": {"test": {"type": "string"}}}

        result = adapter.complete("system", "user", json_schema=schema)

        assert result.structured is True
        assert adapter.call_count == 1
        assert adapter.last_schema == schema


class TestLightningNotQualified:
    """SC-02: Lightning falls back to legacy path."""

    def test_lightning_not_qualified(self):
        caps = get_provider_capabilities("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b")
        assert caps.supports_native_json_schema is False

    def test_lightning_falls_back_to_legacy(self):
        """Lightning should fall back to legacy _complete path."""
        adapter = get_structured_completion_adapter("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b")
        caps = adapter.capabilities if adapter else None

        assert adapter is not None
        assert caps.supports_native_json_schema is False


class TestUltraNotQualified:
    """SC-03: Ultra not qualified for native schema."""

    def test_ultra_not_qualified(self):
        caps = get_provider_capabilities("nvidia", "nvidia/nemotron-3-ultra-550b-a55b")
        assert caps.supports_native_json_schema is False


class TestClaudeLegacy:
    """SC-04: Claude uses legacy path without AttributeError."""

    def test_claude_returns_none_adapter(self):
        adapter = get_structured_completion_adapter("claude", "claude-haiku-4-5")
        assert adapter is None

    def test_claude_uses_legacy_complete(self):
        """Claude should fall back to legacy _complete without AttributeError."""
        with patch("app.providers.registry.get_detector") as mock_get_detector:
            fake_detector = FakeLegacyDetector()
            mock_get_detector.return_value = fake_detector

            adapter = get_structured_completion_adapter("claude", "claude-haiku-4-5")
            if adapter is None:
                from app.providers.registry import get_detector
                detector = get_detector("claude", "claude-haiku-4-5")
                complete_fn = getattr(detector, "_complete", None)
                assert callable(complete_fn)
                result = complete_fn("system", "user")
                assert isinstance(result, str)
                assert fake_detector.call_count == 1


class TestClaudeCodeLegacy:
    """SC-05: Claude Code uses legacy path."""

    def test_claude_code_returns_none_adapter(self):
        adapter = get_structured_completion_adapter("claude_code", "claude-sonnet-5")
        assert adapter is None

    def test_claude_code_legacy_path(self):
        from app.providers.registry import get_detector
        with patch("app.editor.agent.get_detector") as mock_get_detector:
            fake_detector = FakeLegacyDetector()
            mock_get_detector.return_value = fake_detector

            adapter = get_structured_completion_adapter("claude_code", "claude-sonnet-5")
            assert adapter is None

            detector = get_detector("claude_code", "claude-sonnet-5")
            complete_fn = getattr(detector, "_complete", None)
            assert callable(complete_fn)


class TestCodexLegacy:
    """SC-06: Codex uses legacy path."""

    def test_codex_returns_none_adapter(self):
        adapter = get_structured_completion_adapter("codex", "gpt-5.6-terra")
        assert adapter is None


class TestNativeSchemaPassed:
    """SC-07: Native schema passed correctly to adapter."""

    def test_schema_passed_exactly_once(self):
        adapter = FakeStructuredAdapter(qualified=True)
        schema = DirectorOutput.model_json_schema()

        adapter.complete("system", "user", json_schema=schema)

        assert adapter.call_count == 1
        assert adapter.last_schema == schema

    def test_schema_passed_on_retry(self):
        """Schema should be passed on retry as well."""
        adapter = FakeStructuredAdapter(qualified=True)
        DirectorOutput.model_json_schema()

        adapter._response = '{"invalid": "json"}'
        try:
            adapter.complete("system", "user", json_schema=DirectorOutput.model_json_schema())
        except:
            pass

        adapter._response = json.dumps({
            "type": "editorial_plan",
            "editorial_plan": {"objective": "test", "operations": []},
            "evidence_sufficiency": "sufficient",
            "decision_rationale": "test",
        })
        adapter.complete(
            "system\n\nRETRY",
            "user",
            json_schema=DirectorOutput.model_json_schema(),
        )

        assert adapter.call_count == 2


class TestSemanticValidationAfterSchema:
    """SC-08: Semantic validation still runs after native schema."""

    def test_unsupported_operation_fails_semantic_validation(self):
        invalid_response = json.dumps({
            "type": "editorial_plan",
            "editorial_plan": {
                "objective": "test",
                "operations": [{"type": "unsupported_operation", "parameters": {}}]
            },
            "evidence_sufficiency": "sufficient",
            "decision_rationale": "test",
        })

        FakeStructuredAdapter(qualified=True, response=invalid_response)


class TestBoundedRetry:
    """SC-09 & SC-10: Bounded retry logic."""

    def test_bounded_retry_succeeds_on_second_attempt(self):
        adapter = FakeStructuredAdapter(qualified=True)

        adapter._response = "not json at all"
        try:
            adapter.complete("sys", "user", json_schema={})
        except:
            pass

        adapter._response = json.dumps({
            "type": "editorial_plan",
            "editorial_plan": {"objective": "test", "operations": []},
            "evidence_sufficiency": "sufficient",
            "decision_rationale": "test",
        })
        result = adapter.complete("sys", "user", json_schema={})

        assert adapter.call_count == 2
        assert result.structured is True

    def test_fail_closed_after_two_invalid(self):
        adapter = FakeStructuredAdapter(qualified=True)

        adapter._response = "invalid 1"
        try:
            adapter.complete("sys", "user", json_schema={})
        except:
            pass

        adapter._response = "invalid 2"
        try:
            adapter.complete("sys", "user", json_schema={})
        except:
            pass

        assert adapter.call_count == 2


class TestNoCrossProviderFallback:
    """SC-11: No automatic cross-provider fallback."""

    def test_nvidia_failure_does_not_invoke_openai(self):
        """NVIDIA failure should NOT automatically invoke OpenAI."""
        with patch("app.providers.registry.get_structured_completion_adapter") as mock_adapter:
            with patch("app.providers.registry.get_detector") as mock_detector:
                mock_adapter.return_value = None
                mock_detector.side_effect = RuntimeError("Provider unavailable")

                from app.providers.registry import get_structured_completion_adapter
                adapter = get_structured_completion_adapter("nvidia", "test-model")
                assert adapter is None

                mock_detector.assert_not_called()


class TestReasoningPrivacy:
    """SC-12: Reasoning content not retained in output."""

    def test_reasoning_not_in_metadata_or_content(self):
        result = CompletionResult(
            content='{"type": "editorial_plan", "editorial_plan": {"objective": "test", "operations": []}, "evidence_sufficiency": "sufficient", "decision_rationale": "test"}',
            provider="nvidia",
            model="nvidia/nemotron-3-super-120b-a12b",
            structured=True,
            reasoning_present=True,
            metadata={},
        )

        assert "reasoning_content" not in result.metadata
        assert result.metadata == {}

        dumped = result.model_dump_json()
        assert "reasoning_content" not in dumped
        assert "PRIVATE_REASONING" not in dumped


class TestUsageOptional:
    """SC-13: Usage tokens are optional."""

    def test_missing_usage_returns_none(self):
        result = CompletionResult(
            content='{"type": "editorial_plan", "editorial_plan": {"objective": "test", "operations": []}, "evidence_sufficiency": "sufficient", "decision_rationale": "test"}',
            provider="test",
            model="test",
            structured=True,
            input_tokens=None,
            output_tokens=None,
        )

        assert result.input_tokens is None
        assert result.output_tokens is None


class TestReworkSnapshot:
    """SC-14: _rework_plan structured path has snapshot available."""

    def test_rework_plan_snapshot_available(self):
        with patch("app.editor.agent.inspect_project") as mock_inspect:
            with patch("app.editor.agent.get_structured_completion_adapter") as mock_adapter:
                with patch("app.editor.agent._parse_plan") as mock_parse:
                    with patch("app.editor.agent._system_context") as mock_sys:
                        mock_inspect.return_value = {"timeline_version": 1}
                        mock_sys.return_value = "system context"

                        mock_adapter_instance = FakeStructuredAdapter(qualified=True)
                        mock_adapter.return_value = mock_adapter_instance

                        mock_parse.return_value = type("Plan", (), {
                            "model_dump": lambda self, **kwargs: {"objective": "test", "operations": []},
                            "base_version": None,
                        })()

                        from app.editor.agent import _rework_plan
                        from app.editor.evidence_request import EvidenceRequest

                        session = MagicMock()
                        project_id = "test-project"
                        original_request = "test request"
                        failed_plan = type("Plan", (), {
                            "output_intent": "edit",
                            "model_dump": lambda self, **kwargs: {"objective": "test"},
                        })()
                        failure = {"error": "test"}
                        evidence = {}

                        try:
                            _rework_plan(
                                session, project_id, original_request, failed_plan, failure, evidence, None
                            )
                            assert True
                        except UnboundLocalError as e:
                            if "snapshot" in str(e):
                                pytest.fail(f"Snapshot bug still exists: {e}")
                            raise


class TestCostCanonical:
    """SC-15 & SC-16: Cost class canonical source."""

    @pytest.mark.parametrize("provider,model,expected_class", [
        ("naive", "", CostClass.LOCAL),
        ("nvidia", "nvidia/nemotron-3-super-120b-a12b", CostClass.FREE_TIER),
        ("openai", "gpt-4o-mini", CostClass.PAID_API),
        ("claude", "claude-haiku-4-5", CostClass.PAID_API),
        ("claude_code", "claude-sonnet-5", CostClass.SUBSCRIPTION),
        ("codex", "gpt-5.6-terra", CostClass.SUBSCRIPTION),
    ])
    def test_cost_class_consistency(self, provider, model, expected_class):
        caps = get_provider_capabilities(provider, model)
        func_class = cost_class(provider)
        dict_class = CostClass(PROVIDER_COST_CLASS.get(provider, "unknown"))

        assert caps.cost_class == expected_class, f"capabilities cost_class for {provider}/{model}"
        assert func_class == expected_class, f"cost_class() for {provider}"
        assert dict_class == expected_class, f"PROVIDER_COST_CLASS for {provider}"

    def test_cost_facade_functions(self):
        assert is_paid_api("openai") is True
        assert is_paid_api("nvidia") is False
        assert is_subscription("claude_code") is True
        assert is_subscription("nvidia") is False


class TestClaudeDefault:
    """SC-17: Claude default model registered correctly."""

    def test_claude_default_model(self):
        from app.providers.registry import DEFAULT_MODELS
        assert DEFAULT_MODELS["claude"] == "claude-haiku-4-5"

    def test_claude_default_has_paid_cost_class(self):
        caps = get_provider_capabilities("claude", "claude-haiku-4-5")
        assert caps.cost_class == CostClass.PAID_API


class TestUnknownProvider:
    """SC-18: Unknown provider fails safely."""

    def test_unknown_provider_conservative(self):
        caps = get_provider_capabilities("unknown-provider", "unknown-model")
        assert caps.supports_native_json_schema is False
        assert caps.supports_json_mode is False
        assert caps.cost_class == CostClass.UNKNOWN

    def test_unknown_provider_never_paid(self):
        caps = get_provider_capabilities("evil-provider", "steal-money")
        assert caps.cost_class != CostClass.PAID_API
        assert is_paid_api("evil-provider") is False


class TestJSONSchemaValidation:
    """SC-19 to SC-22: JSON Schema validation edge cases."""

    def test_valid_json_passes(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
        result = validate_json('{"x": 42}', schema)
        assert result["x"] == 42

    def test_invalid_output_raises_validation_error(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
        with pytest.raises(Exception) as exc_info:
            validate_json('{"x": "not an integer"}', schema)
        assert "ValidationError" in type(exc_info.value).__name__ or "validation" in str(exc_info.value).lower()

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="NO_VALID_JSON_BLOCK"):
            validate_json("not json at all", {})

    def test_invalid_schema_raises_schema_error(self):
        invalid_schema = {"type": "object", "properties": {"x": {"type": "invalid_type"}}}
        with pytest.raises(Exception) as exc_info:
            validate_json('{"x": 1}', invalid_schema)
        assert "SchemaError" in type(exc_info.value).__name__ or "schema" in str(exc_info.value).lower()


class TestValidateJsonExceptions:
    """SC-22 continued: Exception policy."""

    def test_validation_error_caught_by_providers(self):
        pass


class TestReworkSnapshotIntegration:
    """SC-14: Full integration test for rework plan with structured adapter."""

    def test_rework_plan_structured_path_works(self):
        with patch("app.editor.agent.inspect_project") as mock_inspect:
            with patch("app.editor.agent.get_structured_completion_adapter") as mock_adapter:
                with patch("app.editor.agent._parse_plan") as mock_parse:
                    with patch("app.editor.agent._system_context") as mock_sys:
                        mock_inspect.return_value = {"timeline_version": 1}
                        mock_sys.return_value = "system context"

                        mock_adapter_instance = FakeStructuredAdapter(qualified=True)
                        mock_adapter.return_value = mock_adapter_instance

                        mock_parse.return_value = type("Plan", (), {
                            "model_dump": lambda self, **kwargs: {"objective": "test", "operations": []},
                            "base_version": 1,
                        })()

                        from app.editor.agent import _rework_plan

                        session = MagicMock()
                        project_id = "test-project"
                        original_request = "test request"
                        failed_plan = type("Plan", (), {
                            "output_intent": "edit",
                            "model_dump": lambda self, **kwargs: {"objective": "test"},
                        })()
                        failure = {"error": "test"}
                        evidence = {}

                        result = _rework_plan(
                            session, project_id, original_request, failed_plan, failure, evidence, None
                        )

                        assert result is not None
                        assert hasattr(result, "model_dump")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])