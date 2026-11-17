"""SNIPRA 1.6C Block 01 - asset provenance + enhancement foundation contracts.

Contract-only coverage: no FFmpeg / Real-ESRGAN / API execution.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlmodel import Session

from app.core.db import engine, init_db
from app.editor import service as editor_service
from app.editor.asset_provenance import (
    PROVENANCE_METADATA_KEY,
    AssetOrigin,
    AssetProvenanceMetadata,
    AssetSemanticRole,
    compute_full_file_sha256,
    enhanced_provenance,
    infer_legacy_origin,
    read_asset_provenance,
    source_provenance,
    write_asset_provenance,
)
from app.editor.asset_tasks import (
    VALID_TRANSITIONS,
    AssetTaskApprovalState,
    AssetTaskKind,
    AssetTaskStatus,
    InvalidTaskTransition,
    approval_is_valid,
    assert_transition,
    can_transition,
    compute_request_hash,
    estimate_permits_execution,
    guard_ready_to_queue,
    is_terminal,
)
from app.editor.enhancement import (
    EnhancementCapabilities,
    EnhancementEstimate,
    EnhancementProfile,
    EnhancementRequest,
    VideoEnhancementProvider,
)
from app.editor.models import AssetTask, EditorAsset, EditorProject
from app.providers.capabilities import CostClass

# --------------------------------------------------------------------------- #
# Enums                                                                        #
# --------------------------------------------------------------------------- #

def test_asset_origin_values_exact():
    assert [o.value for o in AssetOrigin] == ["SOURCE", "ENHANCED", "GENERATED"]


def test_asset_semantic_role_values_exact():
    assert {r.value for r in AssetSemanticRole} == {"EVIDENTIARY", "ILLUSTRATIVE"}


def test_asset_task_kind_values():
    assert {k.value for k in AssetTaskKind} == {"ENHANCEMENT", "GENERATION"}


def test_asset_task_status_values_exact():
    assert [s.value for s in AssetTaskStatus] == [
        "ESTIMATING",
        "AWAITING_APPROVAL",
        "QUEUED",
        "RUNNING",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
    ]


# --------------------------------------------------------------------------- #
# GENERATED -> ILLUSTRATIVE invariant                                          #
# --------------------------------------------------------------------------- #

def test_generated_must_be_illustrative():
    with pytest.raises(ValidationError, match="ILLUSTRATIVE"):
        AssetProvenanceMetadata(
            origin=AssetOrigin.GENERATED,
            semantic_role=AssetSemanticRole.EVIDENTIARY,
            parent_asset_id="p1",
            root_source_asset_id="r1",
        )


def test_generated_illustrative_ok():
    prov = AssetProvenanceMetadata(
        origin=AssetOrigin.GENERATED,
        semantic_role=AssetSemanticRole.ILLUSTRATIVE,
        parent_asset_id="p1",
        root_source_asset_id="r1",
    )
    assert prov.origin == AssetOrigin.GENERATED
    assert prov.semantic_role == AssetSemanticRole.ILLUSTRATIVE


# --------------------------------------------------------------------------- #
# Lineage                                                                      #
# --------------------------------------------------------------------------- #

def test_source_lineage_no_parent_root_is_self():
    prov = source_provenance("asset-A")
    assert prov.origin == AssetOrigin.SOURCE
    assert prov.parent_asset_id is None
    assert prov.root_source_asset_id == "asset-A"


def test_source_with_parent_rejected():
    with pytest.raises(ValidationError, match="must not have a parent"):
        AssetProvenanceMetadata(
            origin=AssetOrigin.SOURCE,
            semantic_role=AssetSemanticRole.EVIDENTIARY,
            parent_asset_id="oops",
            root_source_asset_id="asset-A",
        )


def test_enhanced_requires_parent_and_root():
    with pytest.raises(ValidationError, match="require a parent_asset_id"):
        AssetProvenanceMetadata(
            origin=AssetOrigin.ENHANCED,
            semantic_role=AssetSemanticRole.ILLUSTRATIVE,
        )


def test_enhanced_lineage_points_at_source():
    src = source_provenance("src-1")
    enh = enhanced_provenance(
        parent=src,
        parent_asset_id="src-1",
        provider="ffmpeg",
        model_or_engine="ffmpeg 7.1 / DEFAULT",
        request_hash="rh",
        output_hash="oh" * 32,
        output_size_bytes=123,
        operation_version="enh_v1",
    )
    assert enh.origin == AssetOrigin.ENHANCED
    assert enh.parent_asset_id == "src-1"
    assert enh.root_source_asset_id == "src-1"
    assert enh.semantic_role == AssetSemanticRole.ILLUSTRATIVE  # never claims to be the source


def test_enhanced_multi_hop_root_walks_back_to_source():
    src = source_provenance("src-1")
    e1 = enhanced_provenance(
        parent=src, parent_asset_id="src-1", provider="ffmpeg",
        model_or_engine="x", request_hash="r1", output_hash="a" * 64,
        output_size_bytes=1, operation_version="enh_v1",
    )
    e2 = enhanced_provenance(
        parent=e1, parent_asset_id="enh-1", provider="ffmpeg",
        model_or_engine="x", request_hash="r2", output_hash="b" * 64,
        output_size_bytes=1, operation_version="enh_v1",
    )
    assert e2.parent_asset_id == "enh-1"
    assert e2.root_source_asset_id == "src-1"


# --------------------------------------------------------------------------- #
# Full-file SHA-256 (distinct from the partial fingerprint)                    #
# --------------------------------------------------------------------------- #

def test_full_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "clip.bin"
    payload = os.urandom(4096)
    p.write_bytes(payload)
    assert compute_full_file_sha256(p) == hashlib.sha256(payload).hexdigest()


def test_full_sha256_streaming_largeish_fixture(tmp_path):
    # > 3 chunks at a deliberately tiny chunk size; hashes the WHOLE file,
    # not just the first MiB like EditorAsset.fingerprint does.
    p = tmp_path / "big.bin"
    payload = os.urandom(1024 * 1024 + 777) + b"tail-marker"
    p.write_bytes(payload)
    got = compute_full_file_sha256(p, chunk_bytes=64 * 1024)
    assert got == hashlib.sha256(payload).hexdigest()
    # And it differs from a first-MiB-only digest.
    partial = hashlib.sha256(payload[: 1024 * 1024]).hexdigest()
    assert got != partial


def test_full_sha256_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute_full_file_sha256(tmp_path / "nope.bin")


# --------------------------------------------------------------------------- #
# AssetTask state machine                                                      #
# --------------------------------------------------------------------------- #

def test_all_statuses_have_a_transition_entry():
    assert set(VALID_TRANSITIONS) == set(AssetTaskStatus)


def test_terminal_states_have_no_exits():
    for s in (AssetTaskStatus.COMPLETED, AssetTaskStatus.FAILED, AssetTaskStatus.CANCELLED):
        assert is_terminal(s)
        assert VALID_TRANSITIONS[s] == frozenset()


def test_linear_forward_progress():
    assert can_transition("ESTIMATING", "AWAITING_APPROVAL")
    assert can_transition("AWAITING_APPROVAL", "QUEUED")
    assert can_transition("QUEUED", "RUNNING")
    assert can_transition("RUNNING", "COMPLETED")


def test_invalid_transitions_rejected():
    for bad in [
        ("ESTIMATING", "RUNNING"),
        ("ESTIMATING", "QUEUED"),
        ("AWAITING_APPROVAL", "RUNNING"),
        ("QUEUED", "COMPLETED"),
        ("COMPLETED", "RUNNING"),
        ("FAILED", "QUEUED"),
        ("CANCELLED", "RUNNING"),
    ]:
        assert not can_transition(*bad)
        with pytest.raises(InvalidTaskTransition):
            assert_transition(*bad)


def test_every_nonterminal_can_fail_and_cancel():
    for s in (
        AssetTaskStatus.ESTIMATING,
        AssetTaskStatus.AWAITING_APPROVAL,
        AssetTaskStatus.QUEUED,
        AssetTaskStatus.RUNNING,
    ):
        assert can_transition(s, AssetTaskStatus.FAILED)
        assert can_transition(s, AssetTaskStatus.CANCELLED)


# --------------------------------------------------------------------------- #
# Request hashing + approval binding                                           #
# --------------------------------------------------------------------------- #

def test_request_hash_is_field_order_independent():
    a = EnhancementRequest(source_asset_id="a1", profile="UPSCALE", upscale_factor=2.0)
    b = EnhancementRequest(upscale_factor=2.0, profile="UPSCALE", source_asset_id="a1")
    assert a.request_hash() == b.request_hash()


def test_request_hash_changes_with_intent():
    a = EnhancementRequest(source_asset_id="a1", sharpen_strength=0.5)
    b = EnhancementRequest(source_asset_id="a1", sharpen_strength=0.9)
    assert a.request_hash() != b.request_hash()


def test_compute_request_hash_order_independent():
    assert compute_request_hash({"x": 1, "y": 2}) == compute_request_hash({"y": 2, "x": 1})


def test_approval_valid_only_when_bound_to_current_hash():
    rh = "abc123"
    assert approval_is_valid(
        approval_state=AssetTaskApprovalState.APPROVED,
        approved_request_hash=rh,
        current_request_hash=rh,
    )


def test_changed_request_invalidates_approval():
    assert not approval_is_valid(
        approval_state=AssetTaskApprovalState.APPROVED,
        approved_request_hash="old-hash",
        current_request_hash="new-hash",
    )


def test_unapproved_or_missing_hash_is_invalid():
    assert not approval_is_valid(
        approval_state=AssetTaskApprovalState.PENDING,
        approved_request_hash="h",
        current_request_hash="h",
    )
    assert not approval_is_valid(
        approval_state=AssetTaskApprovalState.APPROVED,
        approved_request_hash=None,
        current_request_hash="h",
    )


# --------------------------------------------------------------------------- #
# Cost policy - fail closed                                                     #
# --------------------------------------------------------------------------- #

def test_local_estimate_permits_execution():
    est = EnhancementEstimate(
        cost_class=CostClass.LOCAL, monetary_cost_known=True, monetary_cost_usd=0.0
    )
    assert est.permits_execution()
    assert estimate_permits_execution(est)
    assert estimate_permits_execution(est.model_dump(mode="json"))


def test_unknown_paid_estimate_fails_closed():
    est = EnhancementEstimate(cost_class=CostClass.PAID_API, monetary_cost_known=False)
    assert not est.permits_execution()
    assert not estimate_permits_execution(est)


def test_known_paid_estimate_permits():
    est = EnhancementEstimate(
        cost_class=CostClass.PAID_API, monetary_cost_known=True, monetary_cost_usd=0.42
    )
    assert est.permits_execution()


def test_no_estimate_fails_closed():
    assert not estimate_permits_execution(None)


def test_estimate_known_without_amount_rejected():
    with pytest.raises(ValidationError):
        EnhancementEstimate(monetary_cost_known=True)


def test_guard_ready_to_queue_blocks_bad_approval():
    with pytest.raises(InvalidTaskTransition, match="approval"):
        guard_ready_to_queue(
            approval_state=AssetTaskApprovalState.APPROVED,
            approved_request_hash="old",
            current_request_hash="new",
            cost_estimate=EnhancementEstimate(
                cost_class=CostClass.LOCAL, monetary_cost_known=True, monetary_cost_usd=0.0
            ),
        )


def test_guard_ready_to_queue_blocks_unknown_paid_cost():
    with pytest.raises(InvalidTaskTransition, match="cost estimate"):
        guard_ready_to_queue(
            approval_state=AssetTaskApprovalState.APPROVED,
            approved_request_hash="h",
            current_request_hash="h",
            cost_estimate=EnhancementEstimate(
                cost_class=CostClass.PAID_API, monetary_cost_known=False
            ),
        )


def test_guard_ready_to_queue_passes_when_all_ok():
    guard_ready_to_queue(
        approval_state=AssetTaskApprovalState.APPROVED,
        approved_request_hash="h",
        current_request_hash="h",
        cost_estimate=EnhancementEstimate(
            cost_class=CostClass.LOCAL, monetary_cost_known=True, monetary_cost_usd=0.0
        ),
    )


# --------------------------------------------------------------------------- #
# EnhancementRequest validation / no command injection                         #
# --------------------------------------------------------------------------- #

def test_request_rejects_raw_command_fields():
    with pytest.raises(ValidationError):
        EnhancementRequest(source_asset_id="a1", raw_ffmpeg_args="-vf scale=2:2")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        EnhancementRequest(source_asset_id="a1", raw_shell_command="rm -rf /")  # type: ignore[call-arg]


def test_request_rejects_upscale_and_target_together():
    with pytest.raises(ValidationError, match="not both"):
        EnhancementRequest(
            source_asset_id="a1", profile="UPSCALE", upscale_factor=2.0, target_width=1920
        )


def test_request_rejects_out_of_range_knobs():
    with pytest.raises(ValidationError):
        EnhancementRequest(source_asset_id="a1", saturation=99.0)
    with pytest.raises(ValidationError):
        EnhancementRequest(source_asset_id="a1", target_width=4)


def test_upscale_profile_needs_a_target():
    with pytest.raises(ValidationError, match="UPSCALE profile"):
        EnhancementRequest(source_asset_id="a1", profile="UPSCALE")


def test_request_requires_source_asset_id():
    with pytest.raises(ValidationError):
        EnhancementRequest(source_asset_id="")


# --------------------------------------------------------------------------- #
# Provider contract + capabilities                                             #
# --------------------------------------------------------------------------- #

def test_provider_interface_is_abstract():
    with pytest.raises(TypeError):
        VideoEnhancementProvider()  # type: ignore[abstract]


def test_capabilities_profile_support():
    caps = EnhancementCapabilities(
        engine_name="ffmpeg", supports_upscale=True, supports_sharpen=True,
        supports_color_adjustment=True, supports_denoise=False,
    )
    assert caps.supports_profile(EnhancementProfile.DEFAULT)
    assert caps.supports_profile(EnhancementProfile.UPSCALE)
    assert caps.supports_profile(EnhancementProfile.CLEANUP)  # sharpen counts
    caps_no_clean = EnhancementCapabilities(engine_name="x")
    assert not caps_no_clean.supports_profile(EnhancementProfile.UPSCALE)
    assert not caps_no_clean.supports_profile(EnhancementProfile.CLEANUP)


# --------------------------------------------------------------------------- #
# Secrets never in the contract                                                #
# --------------------------------------------------------------------------- #

def test_provenance_rejects_credential_like_parameter_keys():
    for bad in ("api_key", "openai_api_key", "SECRET", "auth_token", "password"):
        with pytest.raises(ValidationError, match="credential-like"):
            AssetProvenanceMetadata(
                origin=AssetOrigin.ENHANCED,
                semantic_role=AssetSemanticRole.ILLUSTRATIVE,
                parent_asset_id="p",
                root_source_asset_id="r",
                parameters={bad: "x"},
            )


def test_asset_task_has_no_secret_columns():
    cols = set(AssetTask.model_fields)
    for c in cols:
        low = c.lower()
        assert "key" not in low or c == "output_asset_id"
        assert "secret" not in low and "token" not in low and "password" not in low


# --------------------------------------------------------------------------- #
# Metadata read/write + legacy compatibility                                   #
# --------------------------------------------------------------------------- #

def test_write_provenance_does_not_mutate_input_and_roundtrips():
    original = {"unrelated": 1}
    prov = source_provenance("a1")
    out = write_asset_provenance(original, prov)
    assert original == {"unrelated": 1}  # untouched
    assert PROVENANCE_METADATA_KEY in out
    assert out["unrelated"] == 1
    back = read_asset_provenance(out)
    assert back is not None
    assert back.origin == AssetOrigin.SOURCE
    assert back.root_source_asset_id == "a1"


def test_legacy_asset_without_provenance_reads_none():
    assert read_asset_provenance(None) is None
    assert read_asset_provenance({}) is None
    assert read_asset_provenance({"scene_cache": {}}) is None


def test_infer_legacy_origin_conservative():
    # empty / import-shaped legacy metadata -> SOURCE
    assert infer_legacy_origin(None) == AssetOrigin.SOURCE
    assert infer_legacy_origin({}) == AssetOrigin.SOURCE
    assert infer_legacy_origin({"origin": "AUTO_IMPORT"}) == AssetOrigin.SOURCE
    # ambiguous non-import marker -> None (never guesses ENHANCED/GENERATED)
    assert infer_legacy_origin({"origin": "SOMETHING_ELSE"}) is None
    # explicit provenance block wins
    md = write_asset_provenance({}, source_provenance("x"))
    assert infer_legacy_origin(md) == AssetOrigin.SOURCE


# --------------------------------------------------------------------------- #
# SOURCE immutability + "enhancement creates a NEW asset"                       #
# --------------------------------------------------------------------------- #

@pytest.fixture
def _project_and_source(tmp_path):
    init_db()
    with Session(engine) as session:
        project = editor_service.create_project(session, "prov-block01")
        src_file = tmp_path / "source.mp4"
        src_file.write_bytes(b"original source bytes - immutable " + os.urandom(64))
        asset = editor_service.add_asset(
            session, project, src_file, duration_us=1_000_000, width=640, height=480
        )
        return project.id, asset.id, str(src_file), asset.fingerprint


def test_enhancement_creates_new_asset_leaving_source_intact(_project_and_source, tmp_path):
    project_id, src_id, src_path, src_fp_before = _project_and_source
    src_bytes_before = Path(src_path).read_bytes()

    # Simulate what Block 02 will do: produce a NEW file, register a NEW asset.
    out_file = tmp_path / "enhanced.mp4"
    out_file.write_bytes(b"ENHANCED bytes " + os.urandom(128))
    out_hash = compute_full_file_sha256(out_file)

    with Session(engine) as session:
        project = session.get(EditorProject, project_id)
        src = session.get(EditorAsset, src_id)
        src_prov = read_asset_provenance(src.asset_metadata) or source_provenance(src_id)
        enh_prov = enhanced_provenance(
            parent=src_prov,
            parent_asset_id=src_id,
            provider="ffmpeg",
            model_or_engine="ffmpeg / DEFAULT",
            request_hash="rh-1",
            output_hash=out_hash,
            output_size_bytes=out_file.stat().st_size,
            operation_version="enh_v1",
        )
        enhanced_asset = editor_service.add_asset(
            session, project, out_file, duration_us=1_000_000, width=1280, height=960,
            asset_metadata=write_asset_provenance({}, enh_prov),
        )
        enhanced_id = enhanced_asset.id

    # Source file + row are untouched.
    assert Path(src_path).read_bytes() == src_bytes_before
    with Session(engine) as session:
        src_after = session.get(EditorAsset, src_id)
        assert src_after.original_path == src_path
        assert src_after.fingerprint == src_fp_before
        assert read_asset_provenance(src_after.asset_metadata) is None  # source never rewritten
        enh_after = session.get(EditorAsset, enhanced_id)
        assert enh_after.id != src_id
        prov = read_asset_provenance(enh_after.asset_metadata)
        assert prov.origin == AssetOrigin.ENHANCED
        assert prov.parent_asset_id == src_id
        assert prov.root_source_asset_id == src_id
        assert prov.output_hash == out_hash
        assert prov.provider == "ffmpeg"


def test_no_auto_timeline_replacement_helpers_exist():
    """Contract: an ENHANCED asset never auto-replaces the source in a timeline.
    None of the 1.6C Block-01 contract modules may import the timeline module
    or expose insert/replace helpers."""
    import app.editor.asset_provenance as ap
    import app.editor.asset_tasks as at
    import app.editor.enhancement as en

    for mod in (ap, at, en):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "import timeline" not in src and "from app.editor import timeline" not in src
        names = {n.lower() for n in dir(mod)}
        assert not any(
            ("timeline" in n and ("insert" in n or "replace" in n)) for n in names
        )
