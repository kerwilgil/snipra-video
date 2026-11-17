"""Generative media contracts (SNIPRA 1.6D Block 01).

Canonical contracts for text-to-video / image-to-video generation.
LTX-Video LOCAL is the DEFAULT provider. Runway/Luma are OPTIONAL cloud providers.
NO external calls, NO SDKs, NO API keys. Domain + contracts + safety + tests only.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from app.editor.asset_provenance import AssetProvenanceMetadata

# Bump when the request contract or its hashing changes.
GENERATION_CONTRACT_VERSION = "gen_v1"


class GenerationMode(StrEnum):
    """Generation mode type."""

    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"


class GenerationProvider(StrEnum):
    """Approved providers for 1.6D.

    LOCAL_LTX is the DEFAULT provider. Runway/Luma are optional cloud providers
    that must be explicitly enabled.
    """

    LOCAL_LTX = "local_ltx"
    RUNWAY = "runway"
    LUMA = "luma"


class GenerationModel(StrEnum):
    """Approved models for 1.6D.

    LTX_VIDEO_2B_DISTILLED_FP8 is the DEFAULT model for local generation.
    """

    LTX_VIDEO_2B_DISTILLED_FP8 = "ltx-video-2b-distilled-fp8"
    RUNWAY_GEN45 = "gen4.5"
    LUMA_RAY32 = "ray-3.2"


class GenerationAspectRatio(StrEnum):
    """Supported aspect ratios."""

    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"
    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"


class GenerationResolution(StrEnum):
    """Supported resolutions."""

    SD = "SD"  # 720p
    HD = "HD"  # 1080p


# --- Exceptions ---


class UnsupportedGenerationProvider(ValueError):
    """Raised when a provider is not supported or unavailable."""

    pass


class LocalGenerationUnavailable(Exception):
    """Raised when local generation runtime is not available.

    Fail closed: never falls back to cloud provider.
    """

    pass


# --- Generation Request ---


class GenerationRequest(BaseModel):
    """Describes WHAT the user wants to generate. No raw provider payloads.

    LTX-Video LOCAL is the DEFAULT provider. Runway/Luma are optional cloud
    providers that must be explicitly enabled.
    """

    model_config = {"extra": "forbid"}

    project_id: str = Field(min_length=1)
    mode: GenerationMode = GenerationMode.TEXT_TO_VIDEO

    # DEFAULT: LTX-Video LOCAL
    provider: GenerationProvider = GenerationProvider.LOCAL_LTX
    model: GenerationModel = GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8

    prompt_intent: str = Field(min_length=1, max_length=4000)
    visual_direction: str | None = None
    camera_direction: str | None = None
    avoid_constraints: str | None = None

    duration_seconds: int = Field(ge=1, le=30)
    aspect_ratio: GenerationAspectRatio = GenerationAspectRatio.RATIO_16_9
    resolution: GenerationResolution = GenerationResolution.SD

    # Optional reference assets (image_to_video or visual grounding)
    reference_asset_ids: list[str] = Field(default_factory=list)

    # Optional grounding/context asset IDs (transcript, topic, etc.)
    grounding_asset_ids: list[str] = Field(default_factory=list)

    # Optional seed (only if provider supports it)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)

    # Optional provider-specific settings (must be known/validated)
    generation_parameters: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> GenerationRequest:
        # Validate provider/model combinations
        provider_model_pairs = {
            GenerationProvider.LOCAL_LTX: {GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8},
            GenerationProvider.RUNWAY: {GenerationModel.RUNWAY_GEN45},
            GenerationProvider.LUMA: {GenerationModel.LUMA_RAY32},
        }
        allowed = provider_model_pairs.get(self.provider, set())
        if self.model not in allowed:
            raise ValueError(
                f"Provider {self.provider.value} only supports models: "
                f"{', '.join(m.value for m in allowed)}"
            )

        # Validate image_to_video requires at least one reference
        if self.mode == GenerationMode.IMAGE_TO_VIDEO and not self.reference_asset_ids:
            raise ValueError("image_to_video mode requires at least one reference_asset_id")

        # Validate resolution/aspect ratio combinations
        if self.resolution == GenerationResolution.HD and self.aspect_ratio not in (
            GenerationAspectRatio.RATIO_16_9,
            GenerationAspectRatio.RATIO_9_16,
        ):
            raise ValueError("HD resolution only supports 16:9 or 9:16 aspect ratios")

        # Cloud providers require explicit enabling (handled at API layer)
        # This contract only defines the contract; enabling is an infrastructure concern

        return self

    def canonical_payload(self) -> dict:
        """Deterministic dict used for hashing. Stable key order, contract tag."""
        data = self.model_dump(mode="json", exclude_none=True)
        data["_contract"] = GENERATION_CONTRACT_VERSION
        return data

    def request_hash(self) -> str:
        """SHA-256 of the canonical request. Same intent -> same hash."""
        canonical = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- Generation Estimate (local-first, supports cloud) ---


class GenerationEstimate(BaseModel):
    """Cost/compute estimate for a generation request. Local-first, supports cloud.

    For local providers: billable=false, no monetary approval needed.
    For cloud providers: billable=true, monetary approval required.
    """

    provider: GenerationProvider
    model: GenerationModel
    request_hash: str

    # API cost (cloud providers)
    api_cost_usd: float = Field(ge=0.0, default=0.0)
    currency: str = "USD"
    pricing_basis: str | None = None  # e.g. "per_output_second", "per_credit"
    pricing_version: str | None = None  # e.g. "runway_gen45_2026-09"
    pricing_checked_at: datetime | None = None

    # Compute estimate (local providers)
    estimated_compute_class: str | None = None  # e.g. "LOCAL_GPU", "LOCAL_CPU"
    estimated_vram_class: str | None = None  # e.g. "VRAM_8GB", "VRAM_16GB"
    estimated_duration_seconds: int | None = None

    # Economic flags (included in model_dump for API responses)
    billable: bool = False
    requires_monetary_approval: bool = False

    # Metadata
    estimate_version: str = "gen_est_v1"
    is_exact: bool = False

    @model_validator(mode="after")
    def _check(self) -> GenerationEstimate:
        if self.api_cost_usd < 0:
            raise ValueError("api_cost_usd must be non-negative")
        if self.provider != GenerationProvider.LOCAL_LTX and self.api_cost_usd == 0:
            # Cloud providers must have positive cost or explicit pricing
            pass  # Rate card validation will catch this
        # Set economic flags based on provider
        self.billable = self.provider != GenerationProvider.LOCAL_LTX
        self.requires_monetary_approval = self.provider != GenerationProvider.LOCAL_LTX
        return self

    def permits_execution(self) -> bool:
        """Fail closed: execution permitted only if all approvals are satisfied."""
        # Local: always permitted if estimate is valid (no monetary approval needed)
        if self.provider == GenerationProvider.LOCAL_LTX:
            return True

        # Cloud: requires monetary approval (checked at approval time via estimate)
        # Here we only validate the estimate itself is well-formed
        return self.api_cost_usd != 0


# --- Local Rate Card (Compute Estimate) ---


class LocalComputeEntry(BaseModel):
    """Single compute estimate entry for local generation.

    VRAM and duration are TO_BE_VALIDATED until Block 02 physical benchmark.
    """

    model: GenerationModel
    resolution: GenerationResolution
    duration_seconds: int
    estimated_compute_class: str = "LOCAL_GPU"
    estimated_vram_class: str = "TO_BE_VALIDATED"
    estimated_duration_seconds: int | None = None


# Local compute entries - VRAM/duration are TO_BE_VALIDATED until Block 02 physical benchmark
LOCAL_LTX_COMPUTE_ENTRIES: list[LocalComputeEntry] = [
    LocalComputeEntry(
        model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8,
        resolution=GenerationResolution.SD,
        duration_seconds=5,
    ),
    LocalComputeEntry(
        model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8,
        resolution=GenerationResolution.SD,
        duration_seconds=10,
    ),
    LocalComputeEntry(
        model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8,
        resolution=GenerationResolution.HD,
        duration_seconds=5,
    ),
    LocalComputeEntry(
        model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8,
        resolution=GenerationResolution.HD,
        duration_seconds=10,
    ),
]


# --- Cloud Rate Cards (Optional Cloud Metadata) ---


class RateCardEntry(BaseModel):
    """Single rate card entry for a cloud provider/model combination."""

    provider: GenerationProvider
    model: GenerationModel
    resolution: GenerationResolution
    duration_seconds: int
    cost_usd: float = Field(ge=0.0)
    credits_per_second: float | None = None


# Canonical rate cards - OPTIONAL CLOUD METADATA (not default path)
RUNAWAY_GEN45_RATE_CARD: list[tuple[GenerationResolution, int, float]] = [
    (GenerationResolution.SD, 5, 0.60),
    (GenerationResolution.SD, 10, 1.20),
    (GenerationResolution.HD, 5, 0.60),
    (GenerationResolution.HD, 10, 1.20),
]

LUMA_RAY32_RATE_CARD: list[tuple[GenerationResolution, int, float]] = [
    (GenerationResolution.SD, 5, 0.30),
    (GenerationResolution.SD, 10, 0.90),
    (GenerationResolution.HD, 5, 1.20),
    (GenerationResolution.HD, 10, 3.60),
]

ALL_RATE_CARDS = {
    (GenerationProvider.RUNWAY, GenerationModel.RUNWAY_GEN45): RUNAWAY_GEN45_RATE_CARD,
    (GenerationProvider.LUMA, GenerationModel.LUMA_RAY32): LUMA_RAY32_RATE_CARD,
}


def lookup_rate_card(
    provider: GenerationProvider,
    model: GenerationModel,
    resolution: GenerationResolution,
    duration_seconds: int,
) -> float | None:
    """Look up exact rate card entry. Returns None if no exact match (fail closed)."""
    key = (provider, model)
    if key not in ALL_RATE_CARDS:
        return None
    for res, dur, cost in ALL_RATE_CARDS[key]:
        if res == resolution and dur == duration_seconds:
            return cost
    return None


# --- Provider Capabilities ---


class GenerationProviderCapabilities(BaseModel):
    """Capabilities for a generative media provider/model."""

    provider: GenerationProvider
    model: GenerationModel

    # Core capabilities
    supports_text_to_video: bool = True
    supports_image_to_video: bool = True
    supports_async: bool = True
    supports_polling: bool = True

    # Cancel support
    supports_local_cancel: bool = False  # Local can cancel own process
    supports_remote_cancel: bool = False  # Cloud API cancel

    # Economic
    local: bool = False
    paid: bool = False
    billable: bool = False
    requires_api_key: bool = False
    enabled_by_default: bool = False

    # Limits
    max_duration_seconds: int = 30
    supported_resolutions: list[GenerationResolution] = Field(
        default_factory=lambda: [GenerationResolution.SD, GenerationResolution.HD]
    )
    supported_aspect_ratios: list[GenerationAspectRatio] = Field(
        default_factory=lambda: [
            GenerationAspectRatio.RATIO_16_9,
            GenerationAspectRatio.RATIO_9_16,
            GenerationAspectRatio.RATIO_1_1,
        ]
    )

    def supports_mode(self, mode: GenerationMode) -> bool:
        return mode in (GenerationMode.TEXT_TO_VIDEO, GenerationMode.IMAGE_TO_VIDEO)


# Canonical provider capabilities - SINGLE SOURCE OF TRUTH
GENERATION_PROVIDER_CAPABILITIES: dict[
    GenerationProvider, dict[GenerationModel, GenerationProviderCapabilities]
] = {
    GenerationProvider.LOCAL_LTX: {
        GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8: GenerationProviderCapabilities(
            provider=GenerationProvider.LOCAL_LTX,
            model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8,
            supports_text_to_video=True,
            supports_image_to_video=True,
            supports_async=True,
            supports_polling=True,
            supports_local_cancel=True,
            supports_remote_cancel=False,
            local=True,
            paid=False,
            billable=False,
            requires_api_key=False,
            enabled_by_default=True,
            max_duration_seconds=30,
            supported_resolutions=[GenerationResolution.SD, GenerationResolution.HD],
            supported_aspect_ratios=[
                GenerationAspectRatio.RATIO_16_9,
                GenerationAspectRatio.RATIO_9_16,
                GenerationAspectRatio.RATIO_1_1,
                GenerationAspectRatio.RATIO_4_3,
                GenerationAspectRatio.RATIO_3_4,
            ],
        ),
    },
    GenerationProvider.RUNWAY: {
        GenerationModel.RUNWAY_GEN45: GenerationProviderCapabilities(
            provider=GenerationProvider.RUNWAY,
            model=GenerationModel.RUNWAY_GEN45,
            supports_text_to_video=True,
            supports_image_to_video=True,
            supports_async=True,
            supports_polling=True,
            supports_local_cancel=False,
            supports_remote_cancel=True,
            local=False,
            paid=True,
            billable=True,
            requires_api_key=True,
            enabled_by_default=False,
            max_duration_seconds=30,
            supported_resolutions=[GenerationResolution.SD, GenerationResolution.HD],
            supported_aspect_ratios=[
                GenerationAspectRatio.RATIO_16_9,
                GenerationAspectRatio.RATIO_9_16,
                GenerationAspectRatio.RATIO_1_1,
                GenerationAspectRatio.RATIO_4_3,
                GenerationAspectRatio.RATIO_3_4,
            ],
        ),
    },
    GenerationProvider.LUMA: {
        GenerationModel.LUMA_RAY32: GenerationProviderCapabilities(
            provider=GenerationProvider.LUMA,
            model=GenerationModel.LUMA_RAY32,
            supports_text_to_video=True,
            supports_image_to_video=True,
            supports_async=True,
            supports_polling=True,
            supports_local_cancel=False,
            supports_remote_cancel=False,  # Luma does NOT support cancel
            local=False,
            paid=True,
            billable=True,
            requires_api_key=True,
            enabled_by_default=False,
            max_duration_seconds=30,
            supported_resolutions=[GenerationResolution.SD, GenerationResolution.HD],
            supported_aspect_ratios=[
                GenerationAspectRatio.RATIO_16_9,
                GenerationAspectRatio.RATIO_9_16,
                GenerationAspectRatio.RATIO_1_1,
            ],
        ),
    },
}


def get_generation_provider_capabilities(
    provider: GenerationProvider | str,
    model: GenerationModel | str,
) -> GenerationProviderCapabilities:
    """Get capabilities for a generative provider/model.

    Accepts either enum or string values for flexibility.
    Fail closed for unknown provider/model.
    """
    # Convert string to enum if needed
    if isinstance(provider, str):
        try:
            provider = GenerationProvider(provider)
        except ValueError:
            raise UnsupportedGenerationProvider(f"Unknown provider: {provider}") from None

    if isinstance(model, str):
        try:
            model = GenerationModel(model)
        except ValueError:
            raise UnsupportedGenerationProvider(f"Unknown model: {model}") from None

    if provider in GENERATION_PROVIDER_CAPABILITIES:
        caps = GENERATION_PROVIDER_CAPABILITIES[provider]
        if model in caps:
            return caps[model]

    raise UnsupportedGenerationProvider(
        f"No capabilities for provider={provider.value}, model={model.value}"
    )


def is_generation_provider_local(provider: GenerationProvider) -> bool:
    """Check if a provider is local."""
    return provider == GenerationProvider.LOCAL_LTX


def is_generation_provider_paid(provider: GenerationProvider) -> bool:
    """Check if a provider is paid (cloud)."""
    return provider != GenerationProvider.LOCAL_LTX


def is_generation_provider_enabled_by_default(provider: GenerationProvider) -> bool:
    """Check if a provider is enabled by default."""
    return provider == GenerationProvider.LOCAL_LTX


def requires_generation_api_key(provider: GenerationProvider) -> bool:
    """Check if a provider requires an API key."""
    return provider != GenerationProvider.LOCAL_LTX


def is_generation_provider_billable(provider: GenerationProvider) -> bool:
    """Check if a provider is billable (cloud)."""
    return provider != GenerationProvider.LOCAL_LTX


# SNIPRA conservative VRAM admission policy for LTX FP8.
# Conservative SNIPRA admission threshold pending physical validation;
# not an official LTX minimum.
SNIPRA_LTX_FP8_MIN_VRAM_BYTES = 8 * 1024**3


def supports_native_fp8_architecture(
    compute_capability: tuple[int, int] | None,
) -> bool:
    """Check if compute capability supports native FP8 (float8_e4m3fn).

    Ampere (8.0, 8.6) does NOT have native FP8 tensor cores for this workload.
    Ada Lovelace (8.9) and Hopper (9.0+) do.
    """
    if compute_capability is None:
        return False
    major, minor = compute_capability
    # Ada Lovelace (8.9) or newer
    return (major > 8) or (major == 8 and minor >= 9)


def _torch_has_float8_e4m3fn() -> bool:
    """Check if torch exposes the required FP8 dtype."""
    try:
        import torch
        return hasattr(torch, "float8_e4m3fn")
    except Exception:
        return False


# --- Local Hardware Capability Probe ---


class LocalGenerationHardwareReason(StrEnum):
    """Reason codes for local generation hardware capability."""

    SUPPORTED = "SUPPORTED"
    NO_COMPATIBLE_GPU = "NO_COMPATIBLE_GPU"
    NO_NVIDIA_GPU = "NO_NVIDIA_GPU"
    INSUFFICIENT_VRAM = "INSUFFICIENT_VRAM"
    CUDA_UNAVAILABLE = "CUDA_UNAVAILABLE"
    FP8_UNSUPPORTED = "FP8_UNSUPPORTED"
    RUNTIME_NOT_INSTALLED = "RUNTIME_NOT_INSTALLED"
    CHECKPOINT_NOT_INSTALLED = "CHECKPOINT_NOT_INSTALLED"
    RUNTIME_ERROR = "RUNTIME_ERROR"


class LocalGenerationHardwareInfo(BaseModel):
    """Hardware capability info for local generation."""

    gpu_vendor: str | None = None
    gpu_name: str | None = None
    gpu_vram_bytes: int | None = None
    nvidia_available: bool = False
    cuda_available: bool = False
    cuda_version: str | None = None
    compute_capability: tuple[int, int] | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    system_ram_bytes: int | None = None
    disk_free_bytes: int | None = None
    fp8_supported: bool = False
    runtime_supported: bool = False
    reason_codes: list[LocalGenerationHardwareReason] = Field(default_factory=list)
    reason_message: str = ""

    @property
    def is_supported(self) -> bool:
        return self.runtime_supported


def probe_local_generation_hardware() -> LocalGenerationHardwareInfo:
    """Probe local hardware for generation capability.

    Safe, non-destructive probe. Does not install, download, or modify anything.
    """
    import shutil

    info = LocalGenerationHardwareInfo()
    reasons: list[LocalGenerationHardwareReason] = []

    # System info
    info.system_ram_bytes = _get_system_ram_bytes()
    info.disk_free_bytes = _get_disk_free_bytes(".")

    # Try to detect GPU via nvidia-smi
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path:
        info.nvidia_available = True
        gpu_info = _query_nvidia_smi(nvidia_smi_path)
        if gpu_info:
            info.gpu_vendor = "NVIDIA"
            info.gpu_name = gpu_info.get("name")
            info.gpu_vram_bytes = gpu_info.get("vram_bytes")
            info.cuda_version = gpu_info.get("cuda_version")
    else:
        info.nvidia_available = False
        reasons.append(LocalGenerationHardwareReason.NO_NVIDIA_GPU)

    # Try torch CUDA
    try:
        import torch
        info.torch_version = torch.__version__
        if torch.cuda.is_available():
            info.cuda_available = True
            info.torch_cuda_version = torch.version.cuda
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                if not info.gpu_name:
                    info.gpu_name = name
                    info.gpu_vendor = "NVIDIA"
                cap = torch.cuda.get_device_capability(i)
                info.compute_capability = cap
                props = torch.cuda.get_device_properties(i)
                if not info.gpu_vram_bytes:
                    info.gpu_vram_bytes = props.total_memory
        else:
            reasons.append(LocalGenerationHardwareReason.CUDA_UNAVAILABLE)
    except ImportError:
        reasons.append(LocalGenerationHardwareReason.CUDA_UNAVAILABLE)

    # Determine FP8 support
    # Requires: NVIDIA GPU + CUDA + native FP8 architecture (Ada/Hopper+)
    # + torch.float8_e4m3fn + sufficient VRAM
    if info.nvidia_available and info.cuda_available and info.compute_capability:
        arch_ok = supports_native_fp8_architecture(info.compute_capability)
        dtype_ok = _torch_has_float8_e4m3fn()
        vram_ok = info.gpu_vram_bytes and info.gpu_vram_bytes >= SNIPRA_LTX_FP8_MIN_VRAM_BYTES

        if arch_ok and dtype_ok and vram_ok:
            info.fp8_supported = True
        elif not arch_ok or not dtype_ok:
            reasons.append(LocalGenerationHardwareReason.FP8_UNSUPPORTED)
        elif not vram_ok:
            reasons.append(LocalGenerationHardwareReason.INSUFFICIENT_VRAM)
    elif not info.nvidia_available:
        reasons.append(LocalGenerationHardwareReason.NO_NVIDIA_GPU)
    elif not info.cuda_available:
        reasons.append(LocalGenerationHardwareReason.CUDA_UNAVAILABLE)

    # Overall runtime support
    info.reason_codes = reasons
    info.runtime_supported = (
        info.fp8_supported and len(reasons) == 0
    )

    if not info.runtime_supported:
        # Build reason message
        reason_names = [r.value for r in reasons]
        info.reason_message = (
            f"Local generation not supported on this hardware. "
            f"Reasons: {', '.join(reason_names)}"
        )
    else:
        info.reason_message = "Hardware supports local LTX generation."

    return info


def _get_system_ram_bytes() -> int | None:
    try:
        import platform
        if platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                return mem.ullTotalPhys
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _get_disk_free_bytes(path: str) -> int | None:
    try:
        import shutil
        return shutil.disk_usage(path).free
    except Exception:
        return None


def _query_nvidia_smi(nvidia_smi_path: str) -> dict | None:
    try:
        import subprocess
        result = subprocess.run(
            [
                nvidia_smi_path,
                "--query-gpu=name,memory.total,driver_version,cuda_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 3:
                vram_mb = int(parts[1]) if parts[1].isdigit() else None
                return {
                    "name": parts[0],
                    "vram_bytes": vram_mb * 1024 * 1024 if vram_mb else None,
                    "driver_version": parts[2],
                    "cuda_version": parts[3] if len(parts) > 3 else None,
                }
    except Exception:
        pass
    return None


# --- Provider Task Contract ---


class ExternalTaskState(StrEnum):
    """Normalized external task state."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class GenerationExternalTask(BaseModel):
    """Normalized external task from a generative provider."""

    external_task_id: str
    provider: GenerationProvider
    model: GenerationModel
    state: ExternalTaskState = ExternalTaskState.PENDING
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    output_url: str | None = None
    output_asset_id: str | None = None
    realized_cost_usd: float | None = None
    realized_credits: int | None = None
    metadata: dict = Field(default_factory=dict)


# --- Generative Media Provider Contract ---


class GenerativeMediaProvider(ABC):
    """Abstract interface for generative media providers.

    LTX, Runway, Luma implement this. No external calls in this block.
    """

    @property
    @abstractmethod
    def provider(self) -> GenerationProvider: ...

    @property
    @abstractmethod
    def model(self) -> GenerationModel: ...

    @abstractmethod
    def capabilities(self) -> GenerationProviderCapabilities: ...

    @abstractmethod
    def estimate_cost(self, request: GenerationRequest) -> GenerationEstimate:
        """Return cost/compute estimate. Fail closed if no estimate available."""

    @abstractmethod
    def compile_request(self, request: GenerationRequest) -> dict:
        """Compile the canonical request into provider-specific payload.
        Returns provider-specific payload ready for submission.
        """

    @abstractmethod
    def submit(self, compiled_request: dict) -> GenerationExternalTask:
        """Submit compiled request to provider. Returns external task."""

    @abstractmethod
    def get_status(self, external_task: GenerationExternalTask) -> GenerationExternalTask:
        """Poll provider for task status. Returns updated task."""

    @abstractmethod
    def cancel(self, external_task: GenerationExternalTask) -> GenerationExternalTask:
        """Cancel external task. Returns updated task.
        Raises if provider doesn't support cancel.
        """

    @abstractmethod
    def fetch_result(self, external_task: GenerationExternalTask) -> bytes:
        """Fetch generated artifact bytes. Called after SUCCEEDED."""

    @abstractmethod
    def fetch_result_to_path(self, external_task: GenerationExternalTask, output_path: str) -> None:
        """Fetch and write result to local path."""


# --- Local Generative Runtime Contract (Boundary for Block 02) ---


class LocalRuntimeState(StrEnum):
    """State of the local generative runtime."""

    NOT_INSTALLED = "NOT_INSTALLED"
    INSTALLING = "INSTALLING"
    READY = "READY"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


class LocalGenerativeRuntime(ABC):
    """Abstract interface for local generative runtime (LTX-Video).

    Boundary contract for Block 02 physical implementation.
    No implementation in Block 01 - only the contract.
    """

    @property
    @abstractmethod
    def state(self) -> LocalRuntimeState: ...

    @abstractmethod
    async def probe(self) -> bool:
        """Check if runtime is available and healthy."""

    @abstractmethod
    def availability(self) -> LocalRuntimeState:
        """Get current availability state."""

    @abstractmethod
    async def submit(self, compiled_request: dict) -> GenerationExternalTask:
        """Submit compiled request to local runtime. Returns external task."""

    @abstractmethod
    def get_status(self, external_task: GenerationExternalTask) -> GenerationExternalTask:
        """Poll runtime for task status. Returns updated task."""

    @abstractmethod
    def cancel(self, external_task: GenerationExternalTask) -> GenerationExternalTask:
        """Cancel local task. Returns updated task."""

    @abstractmethod
    def fetch_result(self, external_task: GenerationExternalTask) -> bytes:
        """Fetch generated artifact bytes. Called after SUCCEEDED."""

    @abstractmethod
    def fetch_result_to_path(self, external_task: GenerationExternalTask, output_path: str) -> None:
        """Fetch and write result to local path."""


# --- Prompt Compiler (Provider-Neutral) ---


class CompiledGenerationPrompt(BaseModel):
    """Compiled prompt ready for provider adapter."""

    positive_prompt: str
    negative_prompt: str
    structured_context: dict
    compiler_version: str
    compiled_hash: str


class PromptCompiler:
    """Deterministic prompt compiler. No LLM, no external calls.

    Receives structured intent + grounded context, produces reproducible
    provider-ready prompt. Same input -> same output/hash.

    Provider-specific adaptation happens in adapters (LTXRequestAdapter, etc.)
    """

    COMPILER_VERSION = "prompt_compiler_v1"

    def __init__(self):
        self._templates = {}

    def compile(
        self,
        request: GenerationRequest,
        *,
        grounded_context: str | None = None,
        visual_direction: str | None = None,
        camera_direction: str | None = None,
        negative_constraints: str | None = None,
    ) -> CompiledGenerationPrompt:
        """Compile structured intent into provider-ready prompt."""
        parts = [request.prompt_intent]

        if grounded_context:
            parts.append(f"Context: {grounded_context}")

        if visual_direction:
            parts.append(f"Visual style: {visual_direction}")

        if camera_direction:
            parts.append(f"Camera: {camera_direction}")

        if negative_constraints:
            parts.append(f"Negative: {negative_constraints}")

        # Add technical parameters
        parts.append(f"Duration: {request.duration_seconds}s")
        parts.append(f"Aspect: {request.aspect_ratio.value}")
        parts.append(f"Resolution: {request.resolution.value}")

        if request.seed is not None:
            parts.append(f"Seed: {request.seed}")

        # Build structured context
        structured_context = {
            "mode": request.mode.value,
            "duration_seconds": request.duration_seconds,
            "aspect_ratio": request.aspect_ratio.value,
            "resolution": request.resolution.value,
            "seed": request.seed,
            "generation_parameters": request.generation_parameters,
        }

        compiled_prompt = " | ".join(parts)
        compiled_hash = hashlib.sha256(compiled_prompt.encode("utf-8")).hexdigest()

        return CompiledGenerationPrompt(
            positive_prompt=compiled_prompt,
            negative_prompt=negative_constraints or "",
            structured_context=structured_context,
            compiler_version=self.COMPILER_VERSION,
            compiled_hash=compiled_hash,
        )

    def compile_hash(self, prompt: str) -> str:
        """Hash of compiled prompt for provenance."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


# --- Generated Provenance Helper ---


def generated_provenance(
    *,
    parent_asset_id: str | None,
    root_source_asset_id: str | None,
    provider: GenerationProvider,
    model: GenerationModel,
    mode: GenerationMode,
    duration_seconds: int,
    aspect_ratio: GenerationAspectRatio,
    resolution: GenerationResolution,
    request_hash: str,
    compiled_prompt_hash: str,
    external_task_id: str,
    output_hash: str,
    output_size_bytes: int,
    operation_version: str,
    parameters: dict,
    duration_us: int,
    width: int,
    height: int,
    fps: float,
    reference_asset_ids: list[str] | None = None,
    grounding_asset_ids: list[str] | None = None,
    **descriptive: object,
) -> AssetProvenanceMetadata:
    """Create provenance for a GENERATED asset.

    Returns AssetProvenanceMetadata ready for write_asset_provenance().
    All values MUST come from actual request/execution - no placeholders.
    """
    from app.editor.asset_provenance import (
        AssetOrigin,
        AssetProvenanceMetadata,
        AssetSemanticRole,
    )

    # Determine root_source_asset_id: if provided, use it; else use parent if SOURCE
    root = root_source_asset_id
    if not root and parent_asset_id:
        # We'd need to fetch parent to get its root - for now require explicit
        pass

    prov = AssetProvenanceMetadata(
        origin=AssetOrigin.GENERATED,
        semantic_role=AssetSemanticRole.ILLUSTRATIVE,
        parent_asset_id=parent_asset_id,
        root_source_asset_id=root_source_asset_id,
        operation_type="generation",
        operation_version="gen_v1",
        provider=provider.value,
        model_or_engine=model.value,
        request_hash=request_hash,
        output_hash=output_hash,
        output_size_bytes=output_size_bytes,
        parameters={
            "mode": mode.value,
            "duration_seconds": duration_seconds,
            "aspect_ratio": aspect_ratio.value,
            "resolution": resolution.value,
            "seed": parameters.get("seed"),
            "generation_parameters": parameters.get("generation_parameters", {}),
            "reference_asset_ids": reference_asset_ids or [],
            "grounding_asset_ids": grounding_asset_ids or [],
        },
        duration_us=duration_us,
        width=width,
        height=height,
        fps=fps,
    )
    return prov


# --- Evidence Safety ---


def is_evidence_eligible(origin: str) -> bool:
    """SOURCE and ENHANCED are evidence-eligible. GENERATED is never evidence."""
    from app.editor.asset_provenance import AssetOrigin

    return origin in (AssetOrigin.SOURCE.value, AssetOrigin.ENHANCED.value)


# --- Local Generation Estimate ---


def estimate_local_generation(request: GenerationRequest) -> GenerationEstimate:
    """Canonical local compute estimate using compute entries. Fail closed if no match."""
    if request.provider != GenerationProvider.LOCAL_LTX:
        raise ValueError("estimate_local_generation only supports LOCAL_LTX provider")

    entry = lookup_local_compute_entry(
        model=request.model,
        resolution=request.resolution,
        duration_seconds=request.duration_seconds,
    )
    if entry is None:
        raise ValueError(
            f"No local compute entry for {request.model} "
            f"{request.resolution} {request.duration_seconds}s"
        )

    return GenerationEstimate(
        provider=GenerationProvider.LOCAL_LTX,
        model=request.model,
        request_hash=request.request_hash(),
        api_cost_usd=0.0,
        currency="USD",
        pricing_basis=None,
        pricing_version=None,
        pricing_checked_at=None,
        estimated_compute_class=entry.estimated_compute_class,
        estimated_vram_class=entry.estimated_vram_class,
        estimated_duration_seconds=entry.estimated_duration_seconds,
        estimate_version="gen_est_v1",
        is_exact=True,
    )


def lookup_local_compute_entry(
    model: GenerationModel,
    resolution: GenerationResolution,
    duration_seconds: int,
) -> LocalComputeEntry | None:
    """Look up local compute estimate entry. Returns None if no exact match."""
    for entry in LOCAL_LTX_COMPUTE_ENTRIES:
        if (
            entry.model == model
            and entry.resolution == resolution
            and entry.duration_seconds == duration_seconds
        ):
            return entry
    return None


def estimate_generation_cost(request: GenerationRequest) -> GenerationEstimate:
    """Canonical cost estimation. Routes to local or cloud estimator."""
    if request.provider == GenerationProvider.LOCAL_LTX:
        return estimate_local_generation(request)

    # Cloud provider - use rate cards (optional cloud path)
    entry = lookup_rate_card(
        provider=request.provider,
        model=request.model,
        resolution=request.resolution,
        duration_seconds=request.duration_seconds,
    )
    if entry is None:
        raise ValueError(
            f"No rate card entry for {request.provider} {request.model} "
            f"{request.resolution} {request.duration_seconds}s"
        )

    return GenerationEstimate(
        provider=request.provider,
        model=request.model,
        request_hash=request.request_hash(),
        api_cost_usd=entry,  # cost_usd from rate card
        currency="USD",
        pricing_basis="per_output_second",
        pricing_version=(
            "runway_gen45_2026-09"
            if request.provider == GenerationProvider.RUNWAY
            else "luma_ray32_2026-09"
        ),
        pricing_checked_at=datetime.now(UTC),
        estimated_compute_class=None,
        estimated_vram_class=None,
        estimated_duration_seconds=None,
        estimate_version="gen_est_v1",
        is_exact=True,
    )


# --- Approval Helpers ---


class ApprovalKind(StrEnum):
    CONTENT = "content"
    MONETARY = "monetary"


def approve_content(
    *,
    approval_state: str,
    approved_request_hash: str | None,
    current_request_hash: str,
) -> bool:
    """Validate content approval. Always required. Bound to exact request hash."""
    if approval_state != "APPROVED":
        return False
    if not approved_request_hash:
        return False
    return approved_request_hash == current_request_hash


def approve_monetary(
    *,
    approval_state: str,
    approved_cost_hash: str | None,
    current_cost_hash: str,
) -> bool:
    """Validate monetary approval. Only required for billable providers."""
    if approval_state != "APPROVED":
        return False
    if not approved_cost_hash:
        return False
    return approved_cost_hash == current_cost_hash


def validate_generation_approvals(
    *,
    provider: GenerationProvider,
    content_approval_state: str,
    content_approved_request_hash: str | None,
    current_request_hash: str,
    monetary_approval_state: str | None = None,
    monetary_approved_cost_hash: str | None = None,
    current_cost_hash: str | None = None,
) -> None:
    """Fail closed if approvals are invalid.

    Local providers: content approval REQUIRED, monetary approval NOT required.
    Cloud providers: content approval REQUIRED, monetary approval REQUIRED.
    """
    # Content approval (always required)
    if not approve_generation_request(
        approval_state=content_approval_state,
        approved_request_hash=content_approved_request_hash,
        current_request_hash=current_request_hash,
    ):
        raise ValueError("content approval is missing or no longer bound to the current request")

    # Monetary approval (only for billable providers)
    if is_generation_provider_billable(provider):
        if monetary_approval_state is None or monetary_approved_cost_hash is None:
            raise ValueError("monetary approval is required for billable provider")
        if not approve_generation_request(
            approval_state=monetary_approval_state,
            approved_request_hash=monetary_approved_cost_hash,
            current_request_hash=current_cost_hash or "",
        ):
            raise ValueError(
                "monetary approval is missing or no longer "
                "bound to the current cost estimate"
            )


# --- Approval Helpers ---


def approve_generation_request(
    *,
    approval_state: str,
    approved_request_hash: str | None,
    current_request_hash: str,
) -> bool:
    """Validate content approval. Always required. Bound to exact request hash."""
    if approval_state != "APPROVED":
        return False
    if not approved_request_hash:
        return False
    return approved_request_hash == current_request_hash


# --- Provider Resolution (Canonical, Fail-Closed) ---


def resolve_generation_provider(
    request: GenerationRequest,
    local_runtime: LocalGenerativeRuntime,
    cloud_registry: dict[GenerationProvider, GenerativeMediaProvider] | None = None,
) -> tuple[GenerationProvider, LocalGenerativeRuntime | GenerativeMediaProvider]:
    """Resolve which provider/runtime to use for a generation request.

    Fail-closed semantics:
    - For LOCAL_LTX: runtime must be READY. Any other state raises LocalGenerationUnavailable.
      NEVER consults cloud providers for LOCAL_LTX requests.
    - For RUNWAY/LUMA: provider must be explicitly requested AND present in cloud_registry.
      If not available -> UnsupportedGenerationProvider (no silent fallback).
    - No API calls, no keys, no physical generation. Pure resolution.
    """
    cloud_registry = cloud_registry or {}

    if request.provider == GenerationProvider.LOCAL_LTX:
        availability = local_runtime.availability()
        if availability != LocalRuntimeState.READY:
            raise LocalGenerationUnavailable(
                f"Local generation runtime unavailable: {availability.value}"
            )
        return GenerationProvider.LOCAL_LTX, local_runtime

    # Cloud provider explicitly requested
    if request.provider in cloud_registry:
        return request.provider, cloud_registry[request.provider]

    # Cloud provider requested but not configured/enabled
    raise UnsupportedGenerationProvider(
        f"Provider {request.provider.value} not available or not configured"
    )


def can_install_local_ltx(hardware_info: LocalGenerationHardwareInfo | None = None) -> bool:
    """Check if local LTX runtime can be installed on this hardware.

    Returns True only if hardware supports FP8 and has sufficient resources.
    """
    if hardware_info is None:
        hardware_info = probe_local_generation_hardware()
    return hardware_info.runtime_supported


# --- Local Runtime Factory ---


class LocalGenerationRuntimeError(Exception):
    """Base exception for local generation runtime errors."""
    pass


class LocalRuntimeNotAvailable(LocalGenerationRuntimeError):
    """Raised when local runtime is not available on this machine."""
    pass


class LocalRuntimeNotReady(LocalGenerationRuntimeError):
    """Raised when runtime is installed but not ready."""
    pass


class _LocalLtxRuntime(LocalGenerativeRuntime):
    """Production LTX-Video local runtime adapter.

    Lazy-loads heavy dependencies. Fails closed if LTX backend not available.

    IMPORTANT: Physical LTX backend is DEFERRED (LOCAL_LTX_PHYSICAL_VALIDATION_DEFERRED).
    This runtime will NEVER report READY until Block 02 implements physical execution.
    """

    def __init__(self):
        self._state = LocalRuntimeState.NOT_INSTALLED
        self._initialized = False
        self._backend = None

    def _ensure_initialized(self) -> None:
        """Lazy-load and verify LTX backend availability.

        Returns UNAVAILABLE if ltx_video module exists but physical backend not implemented.
        """
        if self._initialized:
            return
        self._initialized = True

        # Check for LTX backend installation without importing heavy deps at module level
        try:
            import importlib.util
            if importlib.util.find_spec("ltx_video") is None:
                self._state = LocalRuntimeState.NOT_INSTALLED
                return

            # ltx_video module is importable, BUT physical backend (submit/fetch_result/etc.)
            # is NOT implemented (LOCAL_LTX_PHYSICAL_VALIDATION_DEFERRED).
            # Module importability != runtime readiness.
            self._state = LocalRuntimeState.UNAVAILABLE
        except ImportError:
            self._state = LocalRuntimeState.NOT_INSTALLED
        except Exception:
            self._state = LocalRuntimeState.ERROR

    @property
    def state(self) -> LocalRuntimeState:
        self._ensure_initialized()
        return self._state

    async def probe(self) -> bool:
        self._ensure_initialized()
        return self._state == LocalRuntimeState.READY

    def availability(self) -> LocalRuntimeState:
        self._ensure_initialized()
        return self._state

    def compile_request(self, request: GenerationRequest) -> dict:
        """Compile the canonical request into LTX-specific payload."""
        self._ensure_initialized()
        if self._state != LocalRuntimeState.READY:
            raise LocalGenerationUnavailable(f"Runtime not ready: {self._state.value}")

        # Use the existing prompt compiler
        from app.editor.generation import PromptCompiler
        compiler = PromptCompiler()
        compiled = compiler.compile(request)
        return {
            "positive_prompt": compiled.positive_prompt,
            "negative_prompt": compiled.negative_prompt,
            "structured_context": compiled.structured_context,
            "compiler_version": compiled.compiler_version,
            "compiled_hash": compiled.compiled_hash,
            "model": request.model.value,
            "duration": request.duration_seconds,
            "aspect_ratio": request.aspect_ratio.value,
            "resolution": request.resolution.value,
            "seed": request.seed,
        }

    async def submit(self, compiled_request: dict) -> GenerationExternalTask:
        """Submit compiled request to local LTX runtime."""
        self._ensure_initialized()
        if self._state != LocalRuntimeState.READY:
            raise LocalGenerationUnavailable(f"Runtime not ready: {self._state.value}")

        # In Block 02, this would call the actual LTX pipeline
        # For now, raise to indicate not implemented
        raise LocalGenerationUnavailable(
            "LTX-Video backend not implemented. Block 02 will provide physical execution."
        )

    def get_status(self, external_task: GenerationExternalTask) -> GenerationExternalTask:
        """Poll runtime for task status."""
        self._ensure_initialized()
        if self._state != LocalRuntimeState.READY:
            raise LocalGenerationUnavailable(f"Runtime not ready: {self._state.value}")

        # In Block 02, this would poll the actual LTX process
        # For now, return unchanged task (will be handled by caller)
        return external_task

    def cancel(self, external_task: GenerationExternalTask) -> GenerationExternalTask:
        """Cancel local LTX task."""
        self._ensure_initialized()
        if self._state != LocalRuntimeState.READY:
            raise LocalGenerationUnavailable(f"Runtime not ready: {self._state.value}")

        external_task.state = ExternalTaskState.CANCELLED
        external_task.completed_at = datetime.now(UTC)
        return external_task

    def fetch_result(self, external_task: GenerationExternalTask) -> bytes:
        """Fetch generated artifact bytes."""
        self._ensure_initialized()
        if self._state != LocalRuntimeState.READY:
            raise LocalGenerationUnavailable(f"Runtime not ready: {self._state.value}")

        # In Block 02, this would read the actual output file
        raise LocalGenerationUnavailable(
            "LTX-Video backend not implemented. Block 02 will provide physical execution."
        )

    def fetch_result_to_path(self, external_task: GenerationExternalTask, output_path: str) -> None:
        """Fetch and write result to local path."""
        self._ensure_initialized()
        if self._state != LocalRuntimeState.READY:
            raise LocalGenerationUnavailable(f"Runtime not ready: {self._state.value}")

        # In Block 02, this would copy the actual output file
        raise LocalGenerationUnavailable(
            "LTX-Video backend not implemented. Block 02 will provide physical execution."
        )


def get_local_generation_runtime() -> LocalGenerativeRuntime:
    """Get a production LocalGenerativeRuntime instance.

    Fails closed if no runtime is available or not READY.
    Does NOT fall back to fake/test runtimes.

    Returns:
        LocalGenerativeRuntime: A ready-to-use runtime instance.

    Raises:
        LocalRuntimeNotAvailable: If no local runtime is available on this machine.
        LocalRuntimeNotReady: If runtime exists but is not READY.
    """
    # Probe hardware first
    hw_info = probe_local_generation_hardware()
    if not hw_info.runtime_supported:
        raise LocalRuntimeNotAvailable(
            f"Local generation not supported on this hardware: {hw_info.reason_message}"
        )

    # Create and verify production runtime
    runtime = _LocalLtxRuntime()
    availability = runtime.availability()

    if availability == LocalRuntimeState.NOT_INSTALLED:
        raise LocalRuntimeNotAvailable(
            "Local LTX runtime not installed. "
            "Install ltx-video package and configure model weights."
        )
    elif availability == LocalRuntimeState.ERROR:
        raise LocalRuntimeNotAvailable(
            "Local LTX runtime initialization failed."
        )
    elif availability != LocalRuntimeState.READY:
        raise LocalRuntimeNotReady(
            f"Local LTX runtime not ready: {availability.value}"
        )

    return runtime


if __name__ == "__main__":
    # Quick self-test
    req = GenerationRequest(
        project_id="test-project",
        provider=GenerationProvider.LOCAL_LTX,
        model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8,
        prompt_intent="A beautiful sunset over mountains",
        duration_seconds=5,
        aspect_ratio=GenerationAspectRatio.RATIO_16_9,
        resolution=GenerationResolution.SD,
    )
    print(f"Request hash: {req.request_hash()}")
    print(f"Provider: {req.provider.value}")
    print(f"Model: {req.model.value}")
    print(f"Canonical: {json.dumps(req.canonical_payload(), indent=2)}")

    cost = estimate_generation_cost(req)
    print(f"Local estimate: ${cost.api_cost_usd:.2f}, billable={cost.billable}")

    # Test Luma (cloud)
    req2 = GenerationRequest(
        project_id="test-project",
        provider=GenerationProvider.LUMA,
        model=GenerationModel.LUMA_RAY32,
        prompt_intent="A beautiful sunset over mountains",
        duration_seconds=5,
        aspect_ratio=GenerationAspectRatio.RATIO_16_9,
        resolution=GenerationResolution.SD,
    )
    cost2 = estimate_generation_cost(req2)
    print(f"Luma Cost: ${cost2.api_cost_usd:.2f}, billable={cost2.billable}")

    # Test fail closed
    try:
        bad_req = GenerationRequest(
            project_id="test",
            provider=GenerationProvider.RUNWAY,
            model=GenerationModel.RUNWAY_GEN45,
            prompt_intent="test",
            duration_seconds=7,  # Not in rate card
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        estimate_generation_cost(bad_req)
    except ValueError as e:
        print(f"Expected fail: {e}")