"""Canonical asset provenance contract (SNIPRA 1.6C Block 01).

`EditorAsset` stays the single canonical asset entity for SOURCE / ENHANCED /
GENERATED material - there is NO parallel `GeneratedAsset` / `EnhancedAsset`
table. Specialised provenance lives as a typed contract inside
`EditorAsset.asset_metadata` under the `asset_provenance` namespace key.

This block defines the contract only. It does not run FFmpeg, Real-ESRGAN, or
any generative engine, and makes no external/API calls.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

# Namespace key inside EditorAsset.asset_metadata. Mirrors scene_cache's
# "video_intelligence" namespace so provenance never collides with other
# metadata consumers (e.g. auto_import's flat "origin" key).
PROVENANCE_METADATA_KEY = "asset_provenance"

_SHA256_CHUNK_BYTES = 1024 * 1024  # 1 MiB streaming reads


class AssetOrigin(StrEnum):
    """How an EditorAsset came to exist."""

    SOURCE = "SOURCE"  # original / imported material - never mutated by enhancement
    ENHANCED = "ENHANCED"  # derived from another EditorAsset; keeps lineage
    GENERATED = "GENERATED"  # reserved for 1.6D; no generation implemented here


class AssetSemanticRole(StrEnum):
    """Whether an asset may stand in as visual evidence."""

    EVIDENTIARY = "EVIDENTIARY"  # can be cited as showing what really happened
    ILLUSTRATIVE = "ILLUSTRATIVE"  # decorative / synthetic; never real evidence


class AssetProvenanceMetadata(BaseModel):
    """Typed view of `EditorAsset.asset_metadata["asset_provenance"]`.

    Serialised with `model_dump(mode="json")` and stored under
    `PROVENANCE_METADATA_KEY`. Must never carry secrets or API keys.
    """

    origin: AssetOrigin
    semantic_role: AssetSemanticRole

    # Lineage - enough to answer "which asset did this come from?" and
    # "what is the original source?". Not a full DAG.
    parent_asset_id: str | None = None
    root_source_asset_id: str | None = None

    # Operation identity
    operation_type: str | None = None  # e.g. "enhancement"
    operation_version: str | None = None  # contract/pipeline version tag

    # Engine identity (never credentials)
    provider: str | None = None  # e.g. "ffmpeg"
    model_or_engine: str | None = None  # e.g. "ffmpeg 7.1 / profile DEFAULT"

    # Request / output integrity
    request_hash: str | None = None  # hash of the exact request that produced this
    output_hash: str | None = None  # FULL SHA-256 of the final artifact
    output_size_bytes: int | None = Field(default=None, ge=0)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Optional descriptive fields (safe subset)
    parameters: dict | None = None
    source_fingerprint: str | None = None  # partial/source fingerprint, NOT output_hash
    duration_us: int | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    fps: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _check_invariants(self) -> AssetProvenanceMetadata:
        # GENERATED assets can never masquerade as real evidence.
        if (
            self.origin == AssetOrigin.GENERATED
            and self.semantic_role != AssetSemanticRole.ILLUSTRATIVE
        ):
            raise ValueError("GENERATED assets must have semantic_role=ILLUSTRATIVE")
        # SOURCE has no parent and is its own lineage root.
        if self.origin == AssetOrigin.SOURCE and self.parent_asset_id is not None:
            raise ValueError("SOURCE assets must not have a parent_asset_id")
        # ENHANCED/GENERATED are derived: they must record where they came from.
        # For GENERATED text_to_video, there is no parent asset (generated from scratch).
        if self.origin != AssetOrigin.SOURCE:
            if self.origin == AssetOrigin.GENERATED:
                # text_to_video generation has no parent asset
                params = self.parameters or {}
                mode = params.get("mode")
                if mode != "text_to_video":
                    if not self.parent_asset_id:
                        raise ValueError(f"{self.origin} assets require a parent_asset_id")
                    # root_source_asset_id is optional for GENERATED: only set when
                    # there is a factual SOURCE root. I2V from a GENERATED reference
                    # with no factual root will have parent but no root.
                else:
                    # text_to_video: no parent, no root source needed
                    pass
            else:
                # ENHANCED always requires parent and root
                if not self.parent_asset_id:
                    raise ValueError(f"{self.origin} assets require a parent_asset_id")
                if not self.root_source_asset_id:
                    raise ValueError(f"{self.origin} assets require a root_source_asset_id")
        # An explicit secret-looking key must never be persisted here.
        _bad = ("api_key", "secret", "token", "password")
        for key in self.parameters or {}:
            lowered = str(key).lower()
            if any(marker in lowered for marker in _bad):
                raise ValueError(f"parameters must not contain credential-like key {key!r}")
        return self

    def to_metadata_value(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True)


def source_provenance(
    asset_id: str,
    *,
    semantic_role: AssetSemanticRole = AssetSemanticRole.EVIDENTIARY,
    source_fingerprint: str | None = None,
    **descriptive: object,
) -> AssetProvenanceMetadata:
    """Provenance for an original SOURCE asset: no parent, root == itself."""
    return AssetProvenanceMetadata(
        origin=AssetOrigin.SOURCE,
        semantic_role=semantic_role,
        parent_asset_id=None,
        root_source_asset_id=asset_id,
        source_fingerprint=source_fingerprint,
        **descriptive,  # type: ignore[arg-type]
    )


def enhanced_provenance(
    *,
    parent: AssetProvenanceMetadata,
    parent_asset_id: str,
    provider: str,
    model_or_engine: str,
    request_hash: str,
    output_hash: str,
    output_size_bytes: int,
    operation_version: str,
    semantic_role: AssetSemanticRole = AssetSemanticRole.ILLUSTRATIVE,
    parameters: dict | None = None,
    **descriptive: object,
) -> AssetProvenanceMetadata:
    """Provenance for an ENHANCED asset derived from `parent`.

    `root_source_asset_id` walks back to the SOURCE ancestor via `parent`.
    An enhanced asset defaults to ILLUSTRATIVE - it must never claim to be the
    physical original source, even when derived from EVIDENTIARY material.
    """
    root = parent.root_source_asset_id or (
        parent_asset_id if parent.origin == AssetOrigin.SOURCE else None
    )
    if not root:
        raise ValueError("cannot resolve root_source_asset_id from parent provenance")
    return AssetProvenanceMetadata(
        origin=AssetOrigin.ENHANCED,
        semantic_role=semantic_role,
        parent_asset_id=parent_asset_id,
        root_source_asset_id=root,
        operation_type="enhancement",
        operation_version=operation_version,
        provider=provider,
        model_or_engine=model_or_engine,
        request_hash=request_hash,
        output_hash=output_hash,
        output_size_bytes=output_size_bytes,
        parameters=parameters,
        **descriptive,  # type: ignore[arg-type]
    )


def read_asset_provenance(asset_metadata: dict | None) -> AssetProvenanceMetadata | None:
    """Return typed provenance from an asset_metadata dict, or None when absent."""
    if not asset_metadata:
        return None
    raw = asset_metadata.get(PROVENANCE_METADATA_KEY)
    if not isinstance(raw, dict):
        return None
    return AssetProvenanceMetadata.model_validate(raw)


def write_asset_provenance(
    asset_metadata: dict | None, provenance: AssetProvenanceMetadata
) -> dict:
    """Return a NEW asset_metadata dict with provenance written under the
    namespace key. Never mutates the input dict."""
    out = dict(asset_metadata or {})
    out[PROVENANCE_METADATA_KEY] = provenance.to_metadata_value()
    return out


def infer_legacy_origin(asset_metadata: dict | None) -> AssetOrigin | None:
    """Conservative interpretation for pre-1.6C assets that have no explicit
    provenance block.

    Returns SOURCE only when nothing suggests the asset is derived; returns
    None (unknown - caller decides) for ambiguous cases. NEVER guesses
    ENHANCED or GENERATED. Legacy assets keep loading either way.
    """
    if read_asset_provenance(asset_metadata) is not None:
        return read_asset_provenance(asset_metadata).origin  # type: ignore[union-attr]
    md = asset_metadata or {}
    # auto_import writes a flat "origin": "AUTO_IMPORT" - that material is a
    # source import (SOURCE_RANGES) or a flattened render, both SOURCE-shaped.
    flat_origin = str(md.get("origin", "")).upper()
    if flat_origin in ("", "AUTO_IMPORT", "ORIGINAL", "SOURCE", "UPLOAD", "IMPORT"):
        return AssetOrigin.SOURCE
    return None


def compute_full_file_sha256(path: str | Path, *, chunk_bytes: int = _SHA256_CHUNK_BYTES) -> str:
    """Streaming full-file SHA-256 hex digest.

    This is the artifact OUTPUT integrity hash - distinct from the partial
    first-MiB `EditorAsset.fingerprint` used for source/cache identity. Reads
    the file in bounded chunks; never loads it whole into RAM.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"not a file: {p}")
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            block = fh.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
