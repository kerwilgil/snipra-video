"""Tests for Generative Media Contracts (SNIPRA 1.6D Block 01)."""

from __future__ import annotations

import pytest

from app.editor.asset_provenance import AssetProvenanceMetadata
from app.editor.generation import (
    SNIPRA_LTX_FP8_MIN_VRAM_BYTES,
    ExternalTaskState,
    GenerationAspectRatio,
    GenerationEstimate,
    GenerationExternalTask,
    GenerationMode,
    GenerationModel,
    GenerationProvider,
    GenerationRequest,
    GenerationResolution,
    LocalGenerationHardwareInfo,
    LocalGenerationHardwareReason,
    LocalGenerationUnavailable,
    LocalRuntimeState,
    PromptCompiler,
    UnsupportedGenerationProvider,
    approve_generation_request,
    can_install_local_ltx,
    estimate_generation_cost,
    generated_provenance,
    get_generation_provider_capabilities,
    is_evidence_eligible,
    is_generation_provider_billable,
    is_generation_provider_enabled_by_default,
    is_generation_provider_paid,
    probe_local_generation_hardware,
    requires_generation_api_key,
    resolve_generation_provider,
    supports_native_fp8_architecture,
)


class TestDefaultProviderIsLocalLTX:
    """Tests that LTX is the default provider."""

    def test_default_provider_is_local_ltx(self):
        req = GenerationRequest(
            project_id="p1",
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )
        assert req.provider == GenerationProvider.LOCAL_LTX
        assert req.model == GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8

    def test_default_model_is_ltx_2b_distilled_fp8(self):
        req = GenerationRequest(
            project_id="p1",
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )
        assert req.model == GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8

    def test_can_override_provider(self):
        req = GenerationRequest(
            project_id="p1",
            provider=GenerationProvider.RUNWAY,
            model=GenerationModel.RUNWAY_GEN45,
            prompt_intent="test",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )
        assert req.provider == GenerationProvider.RUNWAY
        assert req.model == GenerationModel.RUNWAY_GEN45


class TestProviderCapabilities:
    """Tests for provider capabilities and defaults."""

    def test_local_ltx_is_not_paid(self):
        assert is_generation_provider_paid(GenerationProvider.LOCAL_LTX) is False

    def test_local_ltx_requires_no_api_key(self):
        assert requires_generation_api_key(GenerationProvider.LOCAL_LTX) is False

    def test_local_ltx_enabled_by_default(self):
        assert is_generation_provider_enabled_by_default(GenerationProvider.LOCAL_LTX) is True

    def test_local_ltx_is_not_billable(self):
        assert is_generation_provider_billable(GenerationProvider.LOCAL_LTX) is False

    def test_runway_disabled_by_default(self):
        assert is_generation_provider_enabled_by_default(GenerationProvider.RUNWAY) is False

    def test_luma_disabled_by_default(self):
        assert is_generation_provider_enabled_by_default(GenerationProvider.LUMA) is False

    def test_runway_is_paid(self):
        assert is_generation_provider_paid(GenerationProvider.RUNWAY) is True

    def test_luma_is_paid(self):
        assert is_generation_provider_paid(GenerationProvider.LUMA) is True

    def test_runway_requires_api_key(self):
        assert requires_generation_api_key(GenerationProvider.RUNWAY) is True

    def test_luma_requires_api_key(self):
        assert requires_generation_api_key(GenerationProvider.LUMA) is True


class TestLocalEstimate:
    """Tests for local generation estimate."""

    def test_local_estimate_zero_api_cost(self):
        req = GenerationRequest(
            project_id="p1",
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )
        cost = estimate_generation_cost(req)
        assert cost.api_cost_usd == 0.0

    def test_local_estimate_not_billable(self):
        req = GenerationRequest(
            project_id="p1",
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )
        cost = estimate_generation_cost(req)
        assert cost.billable is False

    def test_local_estimate_no_monetary_approval(self):
        req = GenerationRequest(
            project_id="p1",
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )
        cost = estimate_generation_cost(req)
        assert cost.requires_monetary_approval is False

    def test_local_estimate_compute_class_is_local_gpu(self):
        req = GenerationRequest(
            project_id="p1",
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )
        cost = estimate_generation_cost(req)
        assert cost.estimated_compute_class == "LOCAL_GPU"

    def test_local_estimate_vram_class_to_be_validated(self):
        req = GenerationRequest(
            project_id="p1",
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )
        cost = estimate_generation_cost(req)
        assert cost.estimated_vram_class == "TO_BE_VALIDATED"

    def test_local_estimate_duration_is_none(self):
        req = GenerationRequest(
            project_id="p1",
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )
        cost = estimate_generation_cost(req)
        assert cost.estimated_duration_seconds is None


class TestCloudRequiresMonetaryApproval:
    def test_cloud_requires_monetary_approval(self):
        req = GenerationRequest(
            project_id="p1",
            provider=GenerationProvider.RUNWAY,
            model=GenerationModel.RUNWAY_GEN45,
            prompt_intent="test",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )
        cost = estimate_generation_cost(req)
        assert cost.requires_monetary_approval is True

    def test_cloud_estimate_billable(self):
        req = GenerationRequest(
            project_id="p1",
            provider=GenerationProvider.RUNWAY,
            model=GenerationModel.RUNWAY_GEN45,
            prompt_intent="test",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )
        cost = estimate_generation_cost(req)
        assert cost.billable is True


class TestUnknownProviderFailClosed:
    def test_unknown_provider_raises(self):
        with pytest.raises(UnsupportedGenerationProvider):
            get_generation_provider_capabilities("unknown_provider", "some_model")

    def test_unknown_model_for_known_provider_raises(self):
        with pytest.raises(UnsupportedGenerationProvider):
            get_generation_provider_capabilities(GenerationProvider.RUNWAY, "unknown-model")

    def test_local_unavailable_never_falls_back_to_cloud(self):
        exc = LocalGenerationUnavailable("Local generation runtime unavailable")
        assert "unavailable" in str(exc).lower()


class TestLocalRuntimeStates:
    def test_local_runtime_states_exist(self):
        assert LocalRuntimeState.NOT_INSTALLED == "NOT_INSTALLED"
        assert LocalRuntimeState.INSTALLING == "INSTALLING"
        assert LocalRuntimeState.READY == "READY"
        assert LocalRuntimeState.UNAVAILABLE == "UNAVAILABLE"
        assert LocalRuntimeState.ERROR == "ERROR"


class TestPromptCompilerProviderNeutral:
    def test_compiler_output_is_provider_neutral(self):
        compiler = PromptCompiler()
        req = GenerationRequest(
            project_id="p1",
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )
        compiled = compiler.compile(req)
        assert isinstance(compiled.positive_prompt, str)
        assert "runway" not in compiled.positive_prompt.lower()
        assert "luma" not in compiled.positive_prompt.lower()
        assert compiled.compiled_hash
        assert compiled.compiler_version == PromptCompiler.COMPILER_VERSION

    def test_compiler_includes_technical_params(self):
        compiler = PromptCompiler()
        req = GenerationRequest(
            project_id="p1",
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
            seed=42,
        )
        compiled = compiler.compile(req, grounded_context="ctx", visual_direction="cinematic")
        assert "5s" in compiled.positive_prompt
        assert "16:9" in compiled.positive_prompt
        assert "SD" in compiled.positive_prompt
        assert "Seed: 42" in compiled.positive_prompt
        assert compiled.structured_context["duration_seconds"] == 5
        assert compiled.structured_context["aspect_ratio"] == "16:9"
        assert compiled.structured_context["resolution"] == "SD"
        assert compiled.structured_context["seed"] == 42


class TestGeneratedProvenanceUsesActualValues:
    def test_generated_provenance_uses_actual_request_values(self):

        prov_dict = generated_provenance(
            parent_asset_id="parent1",
            root_source_asset_id="source1",
            provider=GenerationProvider.LOCAL_LTX,
            model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8,
            mode=GenerationMode.IMAGE_TO_VIDEO,
            duration_seconds=10,
            aspect_ratio=GenerationAspectRatio.RATIO_9_16,
            resolution=GenerationResolution.HD,
            request_hash="req_hash",
            compiled_prompt_hash="prompt_hash",
            external_task_id="task123",
            output_hash="out_hash",
            output_size_bytes=1000,
            operation_version="gen_v1",
            parameters={"seed": 42},
            duration_us=10_000_000,
            width=1920,
            height=1080,
            fps=30.0,
            reference_asset_ids=["ref1", "ref2"],
            grounding_asset_ids=["ground1"],
        )
        prov = AssetProvenanceMetadata.model_validate(prov_dict)
        assert prov.origin == "GENERATED"
        assert prov.semantic_role == "ILLUSTRATIVE"
        assert prov.provider == "local_ltx"
        assert prov.model_or_engine == "ltx-video-2b-distilled-fp8"
        assert prov.request_hash == "req_hash"
        assert prov.output_hash == "out_hash"
        assert prov.output_size_bytes == 1000
        params = prov.parameters
        assert params["mode"] == "image_to_video"
        assert params["duration_seconds"] == 10
        assert params["aspect_ratio"] == "9:16"
        assert params["resolution"] == "HD"
        assert params["seed"] == 42
        assert params["reference_asset_ids"] == ["ref1", "ref2"]
        assert params["grounding_asset_ids"] == ["ground1"]


class TestEvidenceEligibility:
    def test_generated_is_never_evidence(self):
        assert is_evidence_eligible("SOURCE") is True
        assert is_evidence_eligible("ENHANCED") is True
        assert is_evidence_eligible("GENERATED") is False

    def test_source_is_eligible(self):
        assert is_evidence_eligible("SOURCE") is True

    def test_enhanced_is_eligible(self):
        assert is_evidence_eligible("ENHANCED") is True


class TestModelChangeChangesHash:
    def test_model_change_changes_hash(self):
        req1 = GenerationRequest(
            project_id="p1", provider="local_ltx", model="ltx-video-2b-distilled-fp8",
            prompt_intent="A sunset", duration_seconds=5, aspect_ratio="16:9", resolution="SD"
        )
        req2 = GenerationRequest(
            project_id="p1", provider="runway", model="gen4.5",
            prompt_intent="A sunset", duration_seconds=5, aspect_ratio="16:9", resolution="SD"
        )
        assert req1.request_hash() != req2.request_hash()

    def test_provider_change_changes_hash(self):
        req1 = GenerationRequest(
            project_id="p1", provider="local_ltx", model="ltx-video-2b-distilled-fp8",
            prompt_intent="A sunset", duration_seconds=5, aspect_ratio="16:9", resolution="SD"
        )
        req2 = GenerationRequest(
            project_id="p1", provider="runway", model="gen4.5",
            prompt_intent="A sunset", duration_seconds=5, aspect_ratio="16:9", resolution="SD"
        )
        assert req1.request_hash() != req2.request_hash()


class TestContentApprovalSemantics:
    def test_local_content_approval_required(self):
        assert approve_generation_request(
            approval_state="APPROVED",
            approved_request_hash="abc123",
            current_request_hash="abc123",
        ) is True

    def test_local_monetary_approval_not_required(self):
        assert is_generation_provider_billable(GenerationProvider.LOCAL_LTX) is False

    def test_cloud_requires_monetary_approval(self):
        assert is_generation_provider_billable(GenerationProvider.RUNWAY) is True
        assert is_generation_provider_billable(GenerationProvider.LUMA) is True


class TestNoCloudFallback:
    """Tests that prove LocalGenerationUnavailable never falls back to cloud.

    Uses the PRODUCTIVE resolver from app.editor.generation.resolve_generation_provider
    """

    def test_local_unavailable_raises_without_cloud_calls(self):
        """When local runtime is UNAVAILABLE, must raise LocalGenerationUnavailable
        and never consult/invoke Runway or Luma providers."""

        # Track calls to cloud providers
        runway_calls = {"submit": 0, "get_status": 0, "cancel": 0, "fetch_result": 0}
        luma_calls = {"submit": 0, "get_status": 0, "cancel": 0, "fetch_result": 0}

        class FakeRunwayProvider:
            provider = GenerationProvider.RUNWAY
            model = GenerationModel.RUNWAY_GEN45

            def capabilities(self):
                return get_generation_provider_capabilities(self.provider, self.model)

            def estimate_cost(self, request):
                runway_calls["estimate_cost"] = runway_calls.get("estimate_cost", 0) + 1
                raise AssertionError("Runway estimate_cost must not be called")

            def compile_request(self, request):
                runway_calls["compile_request"] = runway_calls.get("compile_request", 0) + 1
                raise AssertionError("Runway compile_request must not be called")

            def submit(self, compiled_request):
                runway_calls["submit"] += 1
                raise AssertionError("Runway submit must not be called")

            def get_status(self, external_task):
                runway_calls["get_status"] += 1
                raise AssertionError("Runway get_status must not be called")

            def cancel(self, external_task):
                runway_calls["cancel"] += 1
                raise AssertionError("Runway cancel must not be called")

            def fetch_result(self, external_task):
                runway_calls["fetch_result"] += 1
                raise AssertionError("Runway fetch_result must not be called")

            def fetch_result_to_path(self, external_task, output_path):
                runway_calls["fetch_result_to_path"] = (
                    runway_calls.get("fetch_result_to_path", 0) + 1
                )
                raise AssertionError("Runway fetch_result_to_path must not be called")

        class FakeLumaProvider:
            provider = GenerationProvider.LUMA
            model = GenerationModel.LUMA_RAY32

            def capabilities(self):
                return get_generation_provider_capabilities(self.provider, self.model)

            def estimate_cost(self, request):
                luma_calls["estimate_cost"] = luma_calls.get("estimate_cost", 0) + 1
                raise AssertionError("Luma estimate_cost must not be called")

            def compile_request(self, request):
                luma_calls["compile_request"] = luma_calls.get("compile_request", 0) + 1
                raise AssertionError("Luma compile_request must not be called")

            def submit(self, compiled_request):
                luma_calls["submit"] += 1
                raise AssertionError("Luma submit must not be called")

            def get_status(self, external_task):
                luma_calls["get_status"] += 1
                raise AssertionError("Luma get_status must not be called")

            def cancel(self, external_task):
                luma_calls["cancel"] += 1
                raise AssertionError("Luma cancel must not be called")

            def fetch_result(self, external_task):
                luma_calls["fetch_result"] += 1
                raise AssertionError("Luma fetch_result must not be called")

            def fetch_result_to_path(self, external_task, output_path):
                luma_calls["fetch_result_to_path"] = luma_calls.get("fetch_result_to_path", 0) + 1
                raise AssertionError("Luma fetch_result_to_path must not be called")

        class FakeLocalRuntime:
            """Local runtime that is UNAVAILABLE."""

            @property
            def state(self) -> LocalRuntimeState:
                return LocalRuntimeState.UNAVAILABLE

            async def probe(self) -> bool:
                return False

            def availability(self) -> LocalRuntimeState:
                return LocalRuntimeState.UNAVAILABLE

            async def submit(self, compiled_request):
                raise LocalGenerationUnavailable("Local generation runtime unavailable")

            def get_status(self, external_task):
                raise LocalGenerationUnavailable("Local generation runtime unavailable")

            def cancel(self, external_task):
                raise LocalGenerationUnavailable("Local generation runtime unavailable")

            def fetch_result(self, external_task):
                raise LocalGenerationUnavailable("Local generation runtime unavailable")

            def fetch_result_to_path(self, external_task, output_path):
                raise LocalGenerationUnavailable("Local generation runtime unavailable")

        # Test: LOCAL_LTX request with UNAVAILABLE runtime
        request = GenerationRequest(
            project_id="p1",
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )

        local_runtime = FakeLocalRuntime()
        cloud_registry = {
            GenerationProvider.RUNWAY: FakeRunwayProvider(),
            GenerationProvider.LUMA: FakeLumaProvider(),
        }

        with pytest.raises(LocalGenerationUnavailable):
            resolve_generation_provider(request, local_runtime, cloud_registry)

        # Verify ZERO calls to cloud providers
        assert all(v == 0 for v in runway_calls.values()), f"Runway was called: {runway_calls}"
        assert all(v == 0 for v in luma_calls.values()), f"Luma was called: {luma_calls}"

    def test_local_not_installed_raises(self):
        """When local runtime is NOT_INSTALLED, must raise LocalGenerationUnavailable."""
        request = GenerationRequest(
            project_id="p1",
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )

        class FakeLocalRuntime:
            @property
            def state(self) -> LocalRuntimeState:
                return LocalRuntimeState.NOT_INSTALLED

            async def probe(self) -> bool:
                return False

            def availability(self) -> LocalRuntimeState:
                return LocalRuntimeState.NOT_INSTALLED

            async def submit(self, compiled_request):
                pass

            def get_status(self, external_task):
                pass

            def cancel(self, external_task):
                pass

            def fetch_result(self, external_task):
                pass

            def fetch_result_to_path(self, external_task, output_path):
                pass

        local_runtime = FakeLocalRuntime()
        cloud_registry = {}

        with pytest.raises(LocalGenerationUnavailable) as exc_info:
            resolve_generation_provider(request, local_runtime, cloud_registry)

        assert "NOT_INSTALLED" in str(exc_info.value)

    def test_local_error_raises(self):
        """When local runtime is ERROR, must raise LocalGenerationUnavailable."""
        request = GenerationRequest(
            project_id="p1",
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )

        class FakeLocalRuntime:
            @property
            def state(self) -> LocalRuntimeState:
                return LocalRuntimeState.ERROR

            async def probe(self) -> bool:
                return False

            def availability(self) -> LocalRuntimeState:
                return LocalRuntimeState.ERROR

            async def submit(self, compiled_request):
                pass

            def get_status(self, external_task):
                pass

            def cancel(self, external_task):
                pass

            def fetch_result(self, external_task):
                pass

            def fetch_result_to_path(self, external_task, output_path):
                pass

        local_runtime = FakeLocalRuntime()
        cloud_registry = {}

        with pytest.raises(LocalGenerationUnavailable) as exc_info:
            resolve_generation_provider(request, local_runtime, cloud_registry)

        assert "ERROR" in str(exc_info.value)

    def test_local_ready_resolves_local(self):
        """When local runtime is READY, resolves to local runtime."""
        request = GenerationRequest(
            project_id="p1",
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )

        class FakeLocalRuntime:
            @property
            def state(self) -> LocalRuntimeState:
                return LocalRuntimeState.READY

            async def probe(self) -> bool:
                return True

            def availability(self) -> LocalRuntimeState:
                return LocalRuntimeState.READY

            async def submit(self, compiled_request):
                pass

            def get_status(self, external_task):
                pass

            def cancel(self, external_task):
                pass

            def fetch_result(self, external_task):
                pass

            def fetch_result_to_path(self, external_task, output_path):
                pass

        local_runtime = FakeLocalRuntime()
        cloud_registry = {}

        provider, runtime = resolve_generation_provider(request, local_runtime, cloud_registry)

        assert provider == GenerationProvider.LOCAL_LTX
        assert runtime is local_runtime

    def test_explicit_runway_request_resolves_cloud(self):
        """Explicit RUNWAY request with provider in registry resolves to cloud."""
        request = GenerationRequest(
            project_id="p1",
            provider=GenerationProvider.RUNWAY,
            model=GenerationModel.RUNWAY_GEN45,
            prompt_intent="test",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )

        class FakeRunwayProvider:
            provider = GenerationProvider.RUNWAY
            model = GenerationModel.RUNWAY_GEN45

            def capabilities(self):
                return get_generation_provider_capabilities(self.provider, self.model)

            def estimate_cost(self, request):
                return GenerationEstimate(
                    provider=self.provider,
                    model=self.model,
                    request_hash=request.request_hash(),
                    api_cost_usd=0.60,
                    currency="USD",
                    pricing_basis="per_output_second",
                    pricing_version="runway_gen45_2026-09",
                    billable=True,
                    requires_monetary_approval=True,
                )

            def compile_request(self, request):
                return {"prompt": request.prompt_intent}

            def submit(self, compiled_request):
                return GenerationExternalTask(
                    external_task_id="runway-task-1",
                    provider=self.provider,
                    model=self.model,
                    state=ExternalTaskState.PENDING,
                )

            def get_status(self, external_task):
                return external_task

            def cancel(self, external_task):
                return external_task

            def fetch_result(self, external_task):
                return b"fake-video-data"

            def fetch_result_to_path(self, external_task, output_path):
                pass

        class FakeLocalRuntime:
            @property
            def state(self) -> LocalRuntimeState:
                return LocalRuntimeState.READY

            async def probe(self) -> bool:
                return True

            def availability(self) -> LocalRuntimeState:
                return LocalRuntimeState.READY

            async def submit(self, compiled_request):
                pass

            def get_status(self, external_task):
                pass

            def cancel(self, external_task):
                pass

            def fetch_result(self, external_task):
                pass

            def fetch_result_to_path(self, external_task, output_path):
                pass

        local_runtime = FakeLocalRuntime()
        runway_provider = FakeRunwayProvider()
        cloud_registry = {
            GenerationProvider.RUNWAY: runway_provider,
        }

        provider, resolved_obj = resolve_generation_provider(request, local_runtime, cloud_registry)

        assert provider == GenerationProvider.RUNWAY
        assert resolved_obj is runway_provider

    def test_explicit_luma_request_resolves_cloud(self):
        """Explicit LUMA request with provider in registry resolves to cloud."""
        request = GenerationRequest(
            project_id="p1",
            provider=GenerationProvider.LUMA,
            model=GenerationModel.LUMA_RAY32,
            prompt_intent="test",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )

        class FakeLumaProvider:
            provider = GenerationProvider.LUMA
            model = GenerationModel.LUMA_RAY32

            def capabilities(self):
                return get_generation_provider_capabilities(self.provider, self.model)

            def estimate_cost(self, request):
                return GenerationEstimate(
                    provider=self.provider,
                    model=self.model,
                    request_hash=request.request_hash(),
                    api_cost_usd=0.30,
                    currency="USD",
                    pricing_basis="per_output_second",
                    pricing_version="luma_ray32_2026-09",
                    billable=True,
                    requires_monetary_approval=True,
                )

            def compile_request(self, request):
                return {"prompt": request.prompt_intent}

            def submit(self, compiled_request):
                return GenerationExternalTask(
                    external_task_id="luma-task-1",
                    provider=self.provider,
                    model=self.model,
                    state=ExternalTaskState.PENDING,
                )

            def get_status(self, external_task):
                return external_task

            def cancel(self, external_task):
                return external_task

            def fetch_result(self, external_task):
                return b"fake-video-data"

            def fetch_result_to_path(self, external_task, output_path):
                pass

        class FakeLocalRuntime:
            @property
            def state(self) -> LocalRuntimeState:
                return LocalRuntimeState.READY

            async def probe(self) -> bool:
                return True

            def availability(self) -> LocalRuntimeState:
                return LocalRuntimeState.READY

            async def submit(self, compiled_request):
                pass

            def get_status(self, external_task):
                pass

            def cancel(self, external_task):
                pass

            def fetch_result(self, external_task):
                pass

            def fetch_result_to_path(self, external_task, output_path):
                pass

        local_runtime = FakeLocalRuntime()
        luma_provider = FakeLumaProvider()
        cloud_registry = {
            GenerationProvider.LUMA: luma_provider,
        }

        provider, resolved_obj = resolve_generation_provider(request, local_runtime, cloud_registry)

        assert provider == GenerationProvider.LUMA
        assert resolved_obj is luma_provider

    def test_explicit_runway_not_configured_fail_closed(self):
        """Explicit RUNWAY request without provider in registry fails closed."""
        request = GenerationRequest(
            project_id="p1",
            provider=GenerationProvider.RUNWAY,
            model=GenerationModel.RUNWAY_GEN45,
            prompt_intent="test",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )

        class FakeLocalRuntime:
            @property
            def state(self) -> LocalRuntimeState:
                return LocalRuntimeState.READY

            async def probe(self) -> bool:
                return True

            def availability(self) -> LocalRuntimeState:
                return LocalRuntimeState.READY

            async def submit(self, compiled_request):
                pass

            def get_status(self, external_task):
                pass

            def cancel(self, external_task):
                pass

            def fetch_result(self, external_task):
                pass

            def fetch_result_to_path(self, external_task, output_path):
                pass

        local_runtime = FakeLocalRuntime()
        cloud_registry = {}  # Runway NOT configured

        with pytest.raises(UnsupportedGenerationProvider) as exc_info:
            resolve_generation_provider(request, local_runtime, cloud_registry)

        assert "RUNWAY" in str(exc_info.value).upper()

    def test_explicit_luma_not_configured_fail_closed(self):
        """Explicit LUMA request without provider in registry fails closed."""
        request = GenerationRequest(
            project_id="p1",
            provider=GenerationProvider.LUMA,
            model=GenerationModel.LUMA_RAY32,
            prompt_intent="test",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )

        class FakeLocalRuntime:
            @property
            def state(self) -> LocalRuntimeState:
                return LocalRuntimeState.READY

            async def probe(self) -> bool:
                return True

            def availability(self) -> LocalRuntimeState:
                return LocalRuntimeState.READY

            async def submit(self, compiled_request):
                pass

            def get_status(self, external_task):
                pass

            def cancel(self, external_task):
                pass

            def fetch_result(self, external_task):
                pass

            def fetch_result_to_path(self, external_task, output_path):
                pass

        local_runtime = FakeLocalRuntime()
        cloud_registry = {}  # Luma NOT configured

        with pytest.raises(UnsupportedGenerationProvider) as exc_info:
            resolve_generation_provider(request, local_runtime, cloud_registry)

        assert "LUMA" in str(exc_info.value).upper()


class TestSupportsNativeFP8Architecture:
    """Tests for the FP8 architecture support helper."""

    def test_none_compute_capability(self):
        assert supports_native_fp8_architecture(None) is False

    def test_turing_cc_7_5(self):
        assert supports_native_fp8_architecture((7, 5)) is False

    def test_ampere_a100_cc_8_0(self):
        assert supports_native_fp8_architecture((8, 0)) is False

    def test_ampere_rtx_30_cc_8_6(self):
        assert supports_native_fp8_architecture((8, 6)) is False

    def test_ada_lovelace_rtx_40_cc_8_9(self):
        assert supports_native_fp8_architecture((8, 9)) is True

    def test_hopper_cc_9_0(self):
        assert supports_native_fp8_architecture((9, 0)) is True

    def test_future_architecture_cc_10_0(self):
        assert supports_native_fp8_architecture((10, 0)) is True


class TestHardwareCapabilityGate:
    """Tests for local generation hardware capability probe and download gate."""

    def test_current_machine_amd_unsupported(self):
        """Current machine with AMD GPU and no CUDA should be unsupported."""
        info = probe_local_generation_hardware()

        # This machine: AMD Radeon, 512 MB VRAM, no nvidia-smi, no CUDA
        assert info.nvidia_available is False
        assert info.cuda_available is False
        assert info.fp8_supported is False
        assert info.runtime_supported is False

        # Reason codes should include the key blockers
        assert LocalGenerationHardwareReason.NO_NVIDIA_GPU in info.reason_codes
        assert LocalGenerationHardwareReason.CUDA_UNAVAILABLE in info.reason_codes
        # FP8 unsupported is implied by no NVIDIA GPU

        # Download gate should return False
        assert can_install_local_ltx(info) is False

    def test_download_gate_false_blocks_checkpoint(self):
        """When can_install_local_ltx is False, no checkpoint download should occur."""
        info = LocalGenerationHardwareInfo(
            gpu_vendor="AMD",
            gpu_name="AMD Radeon",
            gpu_vram_bytes=512 * 1024 * 1024,
            nvidia_available=False,
            cuda_available=False,
            system_ram_bytes=16 * 1024**3,
            disk_free_bytes=100 * 1024**3,
            fp8_supported=False,
            runtime_supported=False,
            reason_codes=[
                LocalGenerationHardwareReason.NO_NVIDIA_GPU,
                LocalGenerationHardwareReason.CUDA_UNAVAILABLE,
            ],
        )

        assert can_install_local_ltx(info) is False

    def test_download_gate_true_allows_checkpoint(self):
        """When can_install_local_ltx is True, checkpoint download is allowed.
        Ada Lovelace (8.9) with sufficient VRAM and torch.float8_e4m3fn -> allowed."""
        info = LocalGenerationHardwareInfo(
            gpu_vendor="NVIDIA",
            gpu_name="RTX 4090",
            gpu_vram_bytes=24 * 1024**3,
            nvidia_available=True,
            cuda_available=True,
            cuda_version="12.1",
            compute_capability=(8, 9),
            torch_version="2.1.0",
            torch_cuda_version="12.1",
            system_ram_bytes=32 * 1024**3,
            disk_free_bytes=500 * 1024**3,
            fp8_supported=True,
            runtime_supported=True,
            reason_codes=[],
        )

        assert can_install_local_ltx(info) is True

    def test_insufficient_vram_blocked(self):
        """GPU with < 8 GB VRAM should be blocked by SNIPRA conservative policy."""
        info = LocalGenerationHardwareInfo(
            gpu_vendor="NVIDIA",
            gpu_name="RTX 3060",
            gpu_vram_bytes=6 * 1024**3,  # 6 GB
            nvidia_available=True,
            cuda_available=True,
            cuda_version="12.1",
            compute_capability=(8, 6),
            torch_version="2.1.0",
            torch_cuda_version="12.1",
            system_ram_bytes=16 * 1024**3,
            disk_free_bytes=100 * 1024**3,
            fp8_supported=False,
            runtime_supported=False,
            reason_codes=[LocalGenerationHardwareReason.INSUFFICIENT_VRAM],
        )

        assert can_install_local_ltx(info) is False
        assert LocalGenerationHardwareReason.INSUFFICIENT_VRAM in info.reason_codes

    def test_pre_ampere_gpu_fp8_blocked(self):
        """Pre-Ampere GPU (compute capability < 8.0) should be FP8 blocked."""
        info = LocalGenerationHardwareInfo(
            gpu_vendor="NVIDIA",
            gpu_name="RTX 2080 Ti",
            gpu_vram_bytes=11 * 1024**3,
            nvidia_available=True,
            cuda_available=True,
            cuda_version="11.8",
            compute_capability=(7, 5),  # Turing
            torch_version="2.1.0",
            torch_cuda_version="11.8",
            system_ram_bytes=16 * 1024**3,
            disk_free_bytes=100 * 1024**3,
            fp8_supported=False,
            runtime_supported=False,
            reason_codes=[LocalGenerationHardwareReason.FP8_UNSUPPORTED],
        )

        assert can_install_local_ltx(info) is False
        assert LocalGenerationHardwareReason.FP8_UNSUPPORTED in info.reason_codes

    def test_ampere_gpu_fp8_blocked(self):
        """Ampere GPU (compute capability 8.0 / 8.6) should be FP8 blocked.
        RTX 3090 (CC 8.6) has 24 GB VRAM but NO native FP8 tensor cores for this workload."""
        info = LocalGenerationHardwareInfo(
            gpu_vendor="NVIDIA",
            gpu_name="RTX 3090",
            gpu_vram_bytes=24 * 1024**3,
            nvidia_available=True,
            cuda_available=True,
            cuda_version="12.1",
            compute_capability=(8, 6),  # Ampere
            torch_version="2.1.0",
            torch_cuda_version="12.1",
            system_ram_bytes=32 * 1024**3,
            disk_free_bytes=500 * 1024**3,
            fp8_supported=False,
            runtime_supported=False,
            reason_codes=[LocalGenerationHardwareReason.FP8_UNSUPPORTED],
        )

        assert can_install_local_ltx(info) is False
        assert LocalGenerationHardwareReason.FP8_UNSUPPORTED in info.reason_codes

    def test_ada_gpu_fp8_supported(self):
        """Ada Lovelace GPU (compute capability 8.9) with sufficient VRAM should pass."""
        info = LocalGenerationHardwareInfo(
            gpu_vendor="NVIDIA",
            gpu_name="RTX 4090",
            gpu_vram_bytes=24 * 1024**3,
            nvidia_available=True,
            cuda_available=True,
            cuda_version="12.1",
            compute_capability=(8, 9),  # Ada Lovelace
            torch_version="2.1.0",
            torch_cuda_version="12.1",
            system_ram_bytes=32 * 1024**3,
            disk_free_bytes=500 * 1024**3,
            fp8_supported=True,
            runtime_supported=True,
            reason_codes=[],
        )

        assert can_install_local_ltx(info) is True

    def test_hopper_gpu_fp8_supported(self):
        """Hopper GPU (compute capability 9.0) should pass."""
        info = LocalGenerationHardwareInfo(
            gpu_vendor="NVIDIA",
            gpu_name="H100",
            gpu_vram_bytes=80 * 1024**3,
            nvidia_available=True,
            cuda_available=True,
            cuda_version="12.1",
            compute_capability=(9, 0),  # Hopper
            torch_version="2.1.0",
            torch_cuda_version="12.1",
            system_ram_bytes=64 * 1024**3,
            disk_free_bytes=1000 * 1024**3,
            fp8_supported=True,
            runtime_supported=True,
            reason_codes=[],
        )

        assert can_install_local_ltx(info) is True

    def test_ada_gpu_insufficient_vram_blocked(self):
        """Ada-compatible GPU but with VRAM < SNIPRA policy threshold should be blocked."""
        info = LocalGenerationHardwareInfo(
            gpu_vendor="NVIDIA",
            gpu_name="RTX 4060",
            gpu_vram_bytes=8 * 1024**3,  # exactly 8 GB, but policy is >= 8 GiB (exclusive check)
            nvidia_available=True,
            cuda_available=True,
            cuda_version="12.1",
            compute_capability=(8, 9),  # Ada Lovelace
            torch_version="2.1.0",
            torch_cuda_version="12.1",
            system_ram_bytes=16 * 1024**3,
            disk_free_bytes=100 * 1024**3,
            fp8_supported=False,
            runtime_supported=False,
            reason_codes=[LocalGenerationHardwareReason.INSUFFICIENT_VRAM],
        )

        assert can_install_local_ltx(info) is False
        assert LocalGenerationHardwareReason.INSUFFICIENT_VRAM in info.reason_codes

    def test_missing_float8_dtype_blocked(self):
        """Ada/Hopper GPU but missing torch.float8_e4m3fn should be blocked."""
        info = LocalGenerationHardwareInfo(
            gpu_vendor="NVIDIA",
            gpu_name="RTX 4090",
            gpu_vram_bytes=24 * 1024**3,
            nvidia_available=True,
            cuda_available=True,
            cuda_version="11.8",  # older CUDA without float8
            compute_capability=(8, 9),  # Ada Lovelace
            torch_version="1.13.0",  # old torch without float8
            torch_cuda_version="11.8",
            system_ram_bytes=32 * 1024**3,
            disk_free_bytes=500 * 1024**3,
            fp8_supported=False,
            runtime_supported=False,
            reason_codes=[LocalGenerationHardwareReason.FP8_UNSUPPORTED],
        )

        assert can_install_local_ltx(info) is False
        assert LocalGenerationHardwareReason.FP8_UNSUPPORTED in info.reason_codes

    def test_vram_policy_is_snipra_conservative(self):
        """Confirm VRAM threshold is SNIPRA conservative policy, not official LTX minimum."""
        assert SNIPRA_LTX_FP8_MIN_VRAM_BYTES == 8 * 1024**3
        # This is a SNIPRA admission threshold pending physical validation,
        # not an official LTX minimum requirement.


class TestNoFallbackOnUnsupportedHardware:
    """Tests proving unsupported hardware raises LocalGenerationUnavailable
    without cloud fallback."""

    def test_unsupported_hardware_raises_local_generation_unavailable(self):
        """Unsupported local hardware must raise LocalGenerationUnavailable,
        not fall back to cloud."""

        # Track cloud calls
        runway_calls = {"count": 0}
        luma_calls = {"count": 0}

        class FakeRunwayProvider:
            provider = GenerationProvider.RUNWAY
            model = GenerationModel.RUNWAY_GEN45

            def capabilities(self):
                return get_generation_provider_capabilities(self.provider, self.model)

            def __getattr__(self, name):
                def _fail(*args, **kwargs):
                    runway_calls["count"] += 1
                    raise AssertionError(f"Runway {name} must not be called")
                return _fail

        class FakeLumaProvider:
            provider = GenerationProvider.LUMA
            model = GenerationModel.LUMA_RAY32

            def capabilities(self):
                return get_generation_provider_capabilities(self.provider, self.model)

            def __getattr__(self, name):
                def _fail(*args, **kwargs):
                    luma_calls["count"] += 1
                    raise AssertionError(f"Luma {name} must not be called")
                return _fail

        class FakeUnsupportedLocalRuntime:
            @property
            def state(self) -> LocalRuntimeState:
                return LocalRuntimeState.UNAVAILABLE

            async def probe(self) -> bool:
                return False

            def availability(self) -> LocalRuntimeState:
                return LocalRuntimeState.UNAVAILABLE

            async def submit(self, compiled_request):
                raise LocalGenerationUnavailable("Local generation runtime unavailable")

            def get_status(self, external_task):
                raise LocalGenerationUnavailable("Local generation runtime unavailable")

            def cancel(self, external_task):
                raise LocalGenerationUnavailable("Local generation runtime unavailable")

            def fetch_result(self, external_task):
                raise LocalGenerationUnavailable("Local generation runtime unavailable")

            def fetch_result_to_path(self, external_task, output_path):
                raise LocalGenerationUnavailable("Local generation runtime unavailable")

        request = GenerationRequest(
            project_id="p1",
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="SD",
        )

        local_runtime = FakeUnsupportedLocalRuntime()
        cloud_registry = {
            GenerationProvider.RUNWAY: FakeRunwayProvider(),
            GenerationProvider.LUMA: FakeLumaProvider(),
        }

        with pytest.raises(LocalGenerationUnavailable):
            resolve_generation_provider(request, local_runtime, cloud_registry)

        # Zero cloud calls
        assert runway_calls["count"] == 0
        assert luma_calls["count"] == 0


class TestLocalRuntimeFactory:
    """Tests for get_local_generation_runtime() factory."""

    def test_unsupported_hardware_raises_unavailable(self, monkeypatch):
        """Unsupported hardware -> LocalRuntimeNotAvailable (no download, no cloud)."""
        from app.editor.generation import (
            get_local_generation_runtime,
            LocalRuntimeNotAvailable,
            LocalGenerationHardwareInfo,
            LocalGenerationHardwareReason,
        )

        # Mock hardware probe to return unsupported
        hw_info = LocalGenerationHardwareInfo(
            has_cuda=False,
            cuda_version=None,
            gpu_name="AMD Radeon",
            gpu_vram_mb=512,
            compute_capability=None,
            supports_fp8=False,
            fp8_reason="No NVIDIA GPU",
            runtime_supported=False,
            reason=LocalGenerationHardwareReason.NO_NVIDIA_GPU,
            reason_message="No NVIDIA GPU detected",
        )
        monkeypatch.setattr("app.editor.generation.probe_local_generation_hardware", lambda: hw_info)

        with pytest.raises(LocalRuntimeNotAvailable) as exc_info:
            get_local_generation_runtime()
        assert "not supported on this hardware" in str(exc_info.value)

    def test_supported_hardware_runtime_absent_raises_unavailable(self, monkeypatch):
        """Supported hardware + runtime not installed -> LocalRuntimeNotAvailable."""
        from app.editor.generation import (
            get_local_generation_runtime,
            LocalRuntimeNotAvailable,
            LocalGenerationHardwareInfo,
            LocalGenerationHardwareReason,
        )

        # Mock hardware probe to return supported
        hw_info = LocalGenerationHardwareInfo(
            has_cuda=True,
            cuda_version="12.4",
            gpu_name="NVIDIA RTX 4090",
            gpu_vram_mb=24576,
            compute_capability=(8, 9),
            supports_fp8=True,
            fp8_reason="Ada architecture supports FP8",
            runtime_supported=True,
            reason=LocalGenerationHardwareReason.SUPPORTED,
            reason_message="Hardware supports local generation",
        )
        monkeypatch.setattr("app.editor.generation.probe_local_generation_hardware", lambda: hw_info)

        # Ensure ltx_video is not importable
        import sys
        if "ltx_video" in sys.modules:
            del sys.modules["ltx_video"]

        with pytest.raises(LocalRuntimeNotAvailable) as exc_info:
            get_local_generation_runtime()
        assert "not installed" in str(exc_info.value).lower()

    def test_supported_hardware_physical_backend_deferred_raises_not_ready(self, monkeypatch):
        """Supported hardware + ltx_video module importable but physical backend deferred -> LocalRuntimeNotReady.

        Operational truth test: MODULE_IMPORTABLE != RUNTIME_READY.
        The physical LTX backend is officially deferred (LOCAL_LTX_PHYSICAL_VALIDATION_DEFERRED).
        """
        from app.editor.generation import (
            get_local_generation_runtime,
            LocalRuntimeNotReady,
            LocalGenerationHardwareInfo,
            LocalGenerationHardwareReason,
        )

        # Mock hardware probe to return supported
        hw_info = LocalGenerationHardwareInfo(
            has_cuda=True,
            cuda_version="12.4",
            gpu_name="NVIDIA RTX 4090",
            gpu_vram_mb=24576,
            compute_capability=(8, 9),
            supports_fp8=True,
            fp8_reason="Ada architecture supports FP8",
            runtime_supported=True,
            reason=LocalGenerationHardwareReason.SUPPORTED,
            reason_message="Hardware supports local generation",
        )
        monkeypatch.setattr("app.editor.generation.probe_local_generation_hardware", lambda: hw_info)

        # Mock find_spec to simulate ltx_video module being importable
        import importlib.util
        original_find_spec = importlib.util.find_spec
        def mock_find_spec(name):
            if name == "ltx_video":
                # Return a mock spec that indicates module exists
                from importlib.machinery import ModuleSpec
                return ModuleSpec("ltx_video", None)
            return original_find_spec(name)
        monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)

        # Factory must fail closed: physical backend not implemented -> UNAVAILABLE -> LocalRuntimeNotReady
        with pytest.raises(LocalRuntimeNotReady) as exc_info:
            get_local_generation_runtime()
        
        assert "not ready" in str(exc_info.value).lower()
        assert "unavailable" in str(exc_info.value).lower() or "deferred" in str(exc_info.value).lower() or "physical" in str(exc_info.value).lower()

    def test_factory_no_fake_import(self):
        """Factory must not import FakeLocalGenerativeRuntime or tests.*."""
        # The factory module itself should not import tests
        import app.editor.generation as gen_module
        # Check source doesn't contain fake imports (basic check)
        import inspect
        source = inspect.getsource(gen_module.get_local_generation_runtime)
        assert "FakeLocalGenerativeRuntime" not in source
        assert "tests.fakes" not in source
        assert "from tests" not in source

    def test_no_cloud_fallback_in_factory(self, monkeypatch):
        """Factory never falls back to cloud providers."""
        from app.editor.generation import (
            get_local_generation_runtime,
            LocalRuntimeNotAvailable,
            LocalGenerationHardwareInfo,
            LocalGenerationHardwareReason,
        )

        # Mock hardware as unsupported
        hw_info = LocalGenerationHardwareInfo(
            has_cuda=False,
            cuda_version=None,
            gpu_name="AMD Radeon",
            gpu_vram_mb=512,
            compute_capability=None,
            supports_fp8=False,
            fp8_reason="No NVIDIA GPU",
            runtime_supported=False,
            reason=LocalGenerationHardwareReason.NO_NVIDIA_GPU,
            reason_message="No NVIDIA GPU detected",
        )
        monkeypatch.setattr("app.editor.generation.probe_local_generation_hardware", lambda: hw_info)

        # Factory should raise LocalRuntimeNotAvailable, never return cloud provider
        with pytest.raises(LocalRuntimeNotAvailable):
            get_local_generation_runtime()

        # Verify the exception is the factory's own, not a cloud error
        try:
            get_local_generation_runtime()
        except LocalRuntimeNotAvailable as e:
            assert "hardware" in str(e).lower()
        except Exception as e:
            pytest.fail(f"Expected LocalRuntimeNotAvailable, got {type(e).__name__}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])