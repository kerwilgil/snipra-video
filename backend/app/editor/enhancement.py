"""Video enhancement contracts (SNIPRA 1.6C Block 01).

Intent-level request + provider interface only. NO FFmpeg pipeline, NO
Real-ESRGAN, NO external binary is executed in this block - Block 02 (FFmpeg)
and the optional Block 03 (Real-ESRGAN) implement engines against these
contracts without changing them.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from app.providers.capabilities import CostClass

# Bump when the request contract or its hashing changes.
ENHANCEMENT_CONTRACT_VERSION = "enh_v1"


class EnhancementProfile(StrEnum):
    """Typed enhancement intent. The concrete FFmpeg graph is Block 02's job."""

    DEFAULT = "DEFAULT"  # future: lanczos scale + light unsharp + light eq
    UPSCALE = "UPSCALE"  # resolution increase focus
    CLEANUP = "CLEANUP"  # denoise / deblock focus
    CUSTOM = "CUSTOM"  # explicit typed knobs below drive the result


class EnhancementRequest(BaseModel):
    """Describes WHAT the user wants, never HOW. There is deliberately no
    `raw_shell_command` / `raw_ffmpeg_args` field: the API can never inject a
    command line."""

    model_config = {"extra": "forbid"}  # reject unknown fields (e.g. raw args)

    source_asset_id: str = Field(min_length=1)
    profile: EnhancementProfile = EnhancementProfile.DEFAULT

    target_width: int | None = Field(default=None, ge=16, le=15360)
    target_height: int | None = Field(default=None, ge=16, le=15360)
    upscale_factor: float | None = Field(default=None, ge=1.0, le=8.0)

    sharpen_strength: float | None = Field(default=None, ge=0.0, le=3.0)
    contrast: float | None = Field(default=None, ge=0.0, le=3.0)
    brightness: float | None = Field(default=None, ge=-1.0, le=1.0)
    saturation: float | None = Field(default=None, ge=0.0, le=3.0)

    denoise_strength: float | None = Field(default=None, ge=0.0, le=3.0)

    preserve_audio: bool = True

    @model_validator(mode="after")
    def _check(self) -> EnhancementRequest:
        if self.upscale_factor is not None and (self.target_width or self.target_height):
            raise ValueError("give either upscale_factor or target dimensions, not both")
        if self.profile == EnhancementProfile.UPSCALE and not (
            self.upscale_factor or self.target_width or self.target_height
        ):
            raise ValueError("UPSCALE profile needs upscale_factor or a target dimension")
        return self

    def canonical_payload(self) -> dict:
        """Deterministic dict used for hashing. Stable key order, contract tag."""
        data = self.model_dump(mode="json", exclude_none=True)
        data["_contract"] = ENHANCEMENT_CONTRACT_VERSION
        return data

    def request_hash(self) -> str:
        """SHA-256 of the canonical request. Same intent -> same hash,
        regardless of field insertion order. Used to bind approvals."""
        canonical = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EnhancementCapabilities(BaseModel):
    """What an enhancement engine can do. Block 02's FFmpeg provider fills this
    in; no external providers (NVIDIA VSR etc.) are registered here."""

    engine_name: str
    engine_version: str = ""
    supports_upscale: bool = False
    supports_sharpen: bool = False
    supports_color_adjustment: bool = False
    supports_denoise: bool = False
    supports_gpu: bool = False
    requires_external_binary: bool = True

    def supports_profile(self, profile: EnhancementProfile) -> bool:
        if profile == EnhancementProfile.UPSCALE:
            return self.supports_upscale
        if profile == EnhancementProfile.CLEANUP:
            return self.supports_denoise or self.supports_sharpen
        return True  # DEFAULT / CUSTOM are always representable


class EnhancementEstimate(BaseModel):
    """Cost + effort estimate. Monetary cost and CPU/time cost are separate
    concepts: local FFmpeg is `LOCAL` with a known 0.0 monetary cost even
    though it burns CPU time."""

    cost_class: CostClass = CostClass.UNKNOWN
    monetary_cost_known: bool = False
    monetary_cost_usd: float | None = Field(default=None, ge=0.0)
    estimated_seconds: int | None = Field(default=None, ge=0)
    notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> EnhancementEstimate:
        if self.monetary_cost_known and self.monetary_cost_usd is None:
            raise ValueError("monetary_cost_known=True requires monetary_cost_usd")
        return self

    def permits_execution(self) -> bool:
        """Fail closed: a paid operation with an unknown monetary cost must
        never run. LOCAL / a known monetary cost may proceed."""
        if self.cost_class in (CostClass.LOCAL, CostClass.FREE_TIER, CostClass.SUBSCRIPTION):
            return True
        return self.monetary_cost_known


class VideoEnhancementProvider(ABC):
    """Interface Block 02 (`FFmpegEnhancementProvider`) and an optional Block 03
    (`RealESRGANEnhancementProvider`) implement without changing this contract.
    """

    @abstractmethod
    def capabilities(self) -> EnhancementCapabilities: ...

    @abstractmethod
    def validate_request(self, request: EnhancementRequest) -> None:
        """Raise on a request this engine cannot satisfy. No side effects."""
        ...

    @abstractmethod
    def estimate(self, request: EnhancementRequest) -> EnhancementEstimate: ...

    @abstractmethod
    def prepare(self, request: EnhancementRequest) -> dict:
        """Resolve the request into an engine-internal plan (still no execution)."""
        ...

    @abstractmethod
    def enhance(self, request: EnhancementRequest, *, output_path: str) -> dict:
        """Produce the enhanced artifact at output_path and return a result
        descriptor. Implemented by Block 02+, not here."""
        ...
