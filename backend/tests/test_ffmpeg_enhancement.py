"""SNIPRA 1.6C Block 02 - local FFmpeg enhancement provider + task lifecycle.

Pure-contract coverage plus real (physical) FFmpeg runs on a tiny locally
generated fixture. No Real-ESRGAN, no GPU, no external/API calls.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlmodel import Session

import app.editor.asset_task_service as ats
from app.core.db import engine, init_db
from app.editor import service as editor_service
from app.editor.asset_provenance import (
    AssetOrigin,
    AssetSemanticRole,
    compute_full_file_sha256,
    read_asset_provenance,
)
from app.editor.asset_task_service import (
    AmbiguousSourceProvenance,
    PostCommitReadError,
    approve_task,
    attach_estimate,
    create_enhancement_task,
    execute_enhancement_task,
    queue_task,
    update_request,
)
from app.editor.asset_tasks import AssetTaskApprovalState, AssetTaskStatus, InvalidTaskTransition
from app.editor.enhancement import EnhancementProfile, EnhancementRequest
from app.editor.ffmpeg_enhancement import (
    OUTPUT_CONTAINER,
    VIDEO_CODEC,
    FFmpegEnhancementError,
    FFmpegEnhancementProvider,
    FFmpegEnhancementUnavailable,
    default_enhanced_output_path,
)
from app.editor.models import AssetTask, EditorAsset, EditorProject

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="FFmpeg no disponible")

_BOGUS = "/nonexistent/definitely-not-ffmpeg"


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def physical_source(tmp_path_factory) -> Path:
    """~4s 320x240 clip with an AAC sine tone."""
    if not _HAS_FFMPEG:
        pytest.skip("FFmpeg no disponible")
    out = tmp_path_factory.mktemp("enh_src") / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=4",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(out),
        ],
        capture_output=True, text=True, check=True,
    )
    return out


@pytest.fixture(scope="session")
def physical_source_noaudio(tmp_path_factory) -> Path:
    if not _HAS_FFMPEG:
        pytest.skip("FFmpeg no disponible")
    out = tmp_path_factory.mktemp("enh_src_na") / "source_na.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=3",
            "-an", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(out),
        ],
        capture_output=True, text=True, check=True,
    )
    return out


@pytest.fixture
def provider(physical_source) -> FFmpegEnhancementProvider:
    return FFmpegEnhancementProvider(resolve_source_path=lambda _aid: physical_source)


@pytest.fixture
def db_session():
    init_db()
    with Session(engine) as session:
        yield session


def _new_project_with_source(session: Session, src_path: Path, *, metadata: dict | None = None):
    project = editor_service.create_project(session, "block02")
    asset = editor_service.add_asset(
        session, project, src_path,
        duration_us=4_000_000, width=320, height=240, has_audio=True,
        asset_metadata=metadata or {},
    )
    return project, asset


# --------------------------------------------------------------------------- #
# Capabilities / engine identity                                              #
# --------------------------------------------------------------------------- #

def test_capabilities_shape():
    caps = FFmpegEnhancementProvider().capabilities()
    assert caps.engine_name == "ffmpeg"
    assert caps.supports_upscale and caps.supports_sharpen
    assert caps.supports_color_adjustment and caps.supports_denoise
    assert caps.supports_gpu is False
    assert caps.requires_external_binary is True


def test_capabilities_support_every_profile():
    caps = FFmpegEnhancementProvider().capabilities()
    for profile in EnhancementProfile:
        assert caps.supports_profile(profile)


def test_missing_binary_reports_unavailable():
    p = FFmpegEnhancementProvider(ffmpeg_path=_BOGUS, ffprobe_path=_BOGUS)
    assert p.is_available() is False
    assert p.engine_version() == ""


def test_enhance_fails_closed_when_binary_missing(tmp_path):
    p = FFmpegEnhancementProvider(
        ffmpeg_path=_BOGUS, ffprobe_path=_BOGUS,
        resolve_source_path=lambda _a: tmp_path / "x.mp4",
    )
    with pytest.raises(FFmpegEnhancementUnavailable):
        p.enhance(EnhancementRequest(source_asset_id="a"), output_path=str(tmp_path / "o.mp4"))


@requires_ffmpeg
def test_engine_version_detected_when_present():
    v = FFmpegEnhancementProvider().engine_version()
    assert v and "ffmpeg" in v.lower()


# --------------------------------------------------------------------------- #
# Request validation                                                          #
# --------------------------------------------------------------------------- #

def test_validate_request_rejects_custom_without_any_knob():
    p = FFmpegEnhancementProvider()
    with pytest.raises(ValueError, match="CUSTOM"):
        p.validate_request(EnhancementRequest(source_asset_id="a", profile="CUSTOM"))


def test_validate_request_accepts_default():
    FFmpegEnhancementProvider().validate_request(EnhancementRequest(source_asset_id="a"))


def test_request_still_rejects_raw_command_fields():
    with pytest.raises(ValidationError):
        EnhancementRequest(source_asset_id="a", raw_ffmpeg_args="-vf scale=2:2")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        EnhancementRequest(source_asset_id="a", raw_shell_command="rm -rf /")  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Estimate - always LOCAL, always $0, never raises                            #
# --------------------------------------------------------------------------- #

def test_estimate_is_local_zero_cost(provider):
    est = provider.estimate(EnhancementRequest(source_asset_id="a"))
    assert est.cost_class.value == "local"
    assert est.monetary_cost_known is True
    assert est.monetary_cost_usd == 0.0
    assert est.permits_execution() is True


def test_estimate_never_raises_on_unresolvable_source():
    p = FFmpegEnhancementProvider(
        resolve_source_path=lambda _a: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    est = p.estimate(EnhancementRequest(source_asset_id="a"))
    assert est.cost_class.value == "local"
    assert est.permits_execution() is True


@requires_ffmpeg
def test_estimate_seconds_scales_with_profile(provider):
    default_est = provider.estimate(EnhancementRequest(source_asset_id="a"))
    upscale_est = provider.estimate(
        EnhancementRequest(source_asset_id="a", profile="UPSCALE", upscale_factor=2.0)
    )
    assert default_est.estimated_seconds and upscale_est.estimated_seconds
    assert upscale_est.estimated_seconds > default_est.estimated_seconds


# --------------------------------------------------------------------------- #
# Dimension resolution - aspect ratio + even + clamp                          #
# --------------------------------------------------------------------------- #

def test_dimensions_from_upscale_factor():
    p = FFmpegEnhancementProvider()
    req = EnhancementRequest(source_asset_id="a", profile="UPSCALE", upscale_factor=2.0)
    assert p._resolve_dimensions(req, 320, 240) == (640, 480)


def test_dimensions_target_width_only_preserves_aspect():
    p = FFmpegEnhancementProvider()
    req = EnhancementRequest(source_asset_id="a", profile="UPSCALE", target_width=960)
    w, h = p._resolve_dimensions(req, 320, 240)
    assert w == 960
    assert abs((w / h) - (320 / 240)) < 0.02


def test_dimensions_target_height_only_preserves_aspect():
    p = FFmpegEnhancementProvider()
    req = EnhancementRequest(source_asset_id="a", profile="UPSCALE", target_height=480)
    w, h = p._resolve_dimensions(req, 320, 240)
    assert h == 480
    assert abs((w / h) - (320 / 240)) < 0.02


def test_dimensions_both_targets_taken_explicitly():
    p = FFmpegEnhancementProvider()
    req = EnhancementRequest(source_asset_id="a", target_width=1280, target_height=720)
    assert p._resolve_dimensions(req, 320, 240) == (1280, 720)


def test_dimensions_always_even():
    p = FFmpegEnhancementProvider()
    req = EnhancementRequest(source_asset_id="a", profile="UPSCALE", upscale_factor=1.5)
    w, h = p._resolve_dimensions(req, 333, 247)
    assert w % 2 == 0 and h % 2 == 0


def test_dimensions_clamped_to_contract_maximum():
    p = FFmpegEnhancementProvider()
    req = EnhancementRequest(source_asset_id="a", profile="UPSCALE", upscale_factor=8.0)
    w, h = p._resolve_dimensions(req, 4000, 4000)
    assert w <= 15360 and h <= 15360


def test_no_resize_normalises_odd_source_to_even():
    p = FFmpegEnhancementProvider()
    req = EnhancementRequest(source_asset_id="a")
    assert p._resolve_dimensions(req, 321, 241) == (320, 240)


# --------------------------------------------------------------------------- #
# Filter graph - typed knobs only, shell-safe                                 #
# --------------------------------------------------------------------------- #

def _chain(req: EnhancementRequest, sw: int = 320, sh: int = 240) -> str:
    p = FFmpegEnhancementProvider()
    tw, th = p._resolve_dimensions(req, sw, sh)
    return p._build_filter_chain(req, sw, sh, tw, th)


def test_chain_default_profile_is_gentle():
    chain = _chain(EnhancementRequest(source_asset_id="a"))
    assert "unsharp=" in chain and "eq=" in chain
    assert "scale=" not in chain  # DEFAULT does not resize by itself


def test_chain_upscale_uses_lanczos_scale():
    chain = _chain(EnhancementRequest(source_asset_id="a", profile="UPSCALE", upscale_factor=2.0))
    assert "scale=640:480:flags=lanczos" in chain


def test_chain_cleanup_denoise_scales_with_strength():
    def cln(strength):
        return _chain(
            EnhancementRequest(source_asset_id="a", profile="CLEANUP", denoise_strength=strength)
        )

    weak, strong = cln(0.5), cln(3.0)
    assert "hqdn3d=" in weak and "hqdn3d=" in strong
    assert weak != strong


def test_chain_custom_maps_only_typed_knobs():
    chain = _chain(
        EnhancementRequest(
            source_asset_id="a", profile="CUSTOM",
            sharpen_strength=1.0, contrast=1.2, saturation=0.9, denoise_strength=0.5,
        )
    )
    assert "hqdn3d=" in chain and "unsharp=" in chain
    assert "contrast=1.2" in chain and "saturation=0.9" in chain


def test_chain_custom_omits_unset_eq_terms():
    chain = _chain(EnhancementRequest(source_asset_id="a", profile="CUSTOM", contrast=1.1))
    assert "contrast=1.1" in chain
    assert "brightness=" not in chain and "saturation=" not in chain


@pytest.mark.parametrize(
    "req",
    [
        EnhancementRequest(source_asset_id="a"),
        EnhancementRequest(source_asset_id="a", profile="UPSCALE", upscale_factor=4.0),
        EnhancementRequest(source_asset_id="a", profile="CLEANUP", denoise_strength=3.0),
        EnhancementRequest(
            source_asset_id="a", profile="CUSTOM",
            sharpen_strength=3.0, contrast=3.0, brightness=-1.0, saturation=0.0,
            denoise_strength=3.0, target_width=1920,
        ),
    ],
)
def test_chain_is_always_shell_safe(req):
    chain = _chain(req)
    for forbidden in (";", "|", "&", "$", "`", "\n", "'", '"', " ", "\\", "\t"):
        assert forbidden not in chain


def test_chain_matches_safe_alphabet():
    from app.editor.ffmpeg_enhancement import _SAFE_CHAIN_RE

    req = EnhancementRequest(
        source_asset_id="a", profile="CUSTOM", brightness=-0.5, sharpen_strength=2.5
    )
    assert _SAFE_CHAIN_RE.match(_chain(req))


# --------------------------------------------------------------------------- #
# Plan determinism + audio strategy                                          #
# --------------------------------------------------------------------------- #

@requires_ffmpeg
def test_prepare_is_deterministic(provider):
    req = EnhancementRequest(source_asset_id="a", profile="UPSCALE", upscale_factor=2.0)
    assert provider.prepare(req) == provider.prepare(req)


@requires_ffmpeg
def test_prepare_plan_carries_no_volatile_fields(provider):
    plan = provider.prepare(EnhancementRequest(source_asset_id="a"))
    assert "created_at" not in plan and "timestamp" not in plan
    assert plan["request_hash"] == EnhancementRequest(source_asset_id="a").request_hash()
    assert plan["contract_version"] == "enh_v1"


def test_audio_args_copy_for_mp4_safe_codec():
    assert FFmpegEnhancementProvider._audio_args(True, "aac") == ["-c:a", "copy"]


def test_audio_args_transcode_for_unsafe_codec():
    args = FFmpegEnhancementProvider._audio_args(True, "pcm_s16le")
    assert args == ["-c:a", "aac", "-b:a", "160k"]


def test_audio_args_disabled_is_an():
    assert FFmpegEnhancementProvider._audio_args(False, "aac") == ["-an"]


# --------------------------------------------------------------------------- #
# Output path safety                                                          #
# --------------------------------------------------------------------------- #

def test_default_output_path_under_project_dir():
    out = default_enhanced_output_path("proj123", "task456", "DEFAULT")
    assert out.name == f"task456-DEFAULT.{OUTPUT_CONTAINER}"
    assert "proj123" in out.parts and "enhanced" in out.parts


def test_default_output_path_strips_dangerous_profile_chars():
    out = default_enhanced_output_path("p", "t", "../../etc/passwd")
    assert ".." not in out.name and "/" not in out.name


# --------------------------------------------------------------------------- #
# Physical enhancement                                                        #
# --------------------------------------------------------------------------- #

def _probe(path: Path) -> dict:
    raw = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
         str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    import json
    data = json.loads(raw)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])
    return {
        "width": int(v["width"]), "height": int(v["height"]),
        "duration": float(data["format"].get("duration", 0)), "has_audio": has_audio,
    }


@requires_ffmpeg
def test_physical_default_keeps_dimensions_and_audio(provider, physical_source, tmp_path):
    src_hash = hashlib.sha256(physical_source.read_bytes()).hexdigest()
    out = tmp_path / "def.mp4"
    t0 = time.monotonic()
    res = provider.enhance(EnhancementRequest(source_asset_id="a"), output_path=str(out))
    elapsed = time.monotonic() - t0
    assert out.is_file() and out.stat().st_size > 0
    info = _probe(out)
    assert (info["width"], info["height"]) == (320, 240)
    assert info["has_audio"] is True
    assert res["output_hash"] == compute_full_file_sha256(out)
    assert res["output_size_bytes"] == out.stat().st_size
    assert res["duration_us"] > 0 and res["fps"] > 0
    # Source untouched.
    assert hashlib.sha256(physical_source.read_bytes()).hexdigest() == src_hash
    assert elapsed < 60


@requires_ffmpeg
def test_physical_upscale_factor_doubles_even(provider, tmp_path):
    out = tmp_path / "up.mp4"
    provider.enhance(
        EnhancementRequest(source_asset_id="a", profile="UPSCALE", upscale_factor=2.0),
        output_path=str(out),
    )
    info = _probe(out)
    assert (info["width"], info["height"]) == (640, 480)


@requires_ffmpeg
def test_physical_upscale_target_width_only_preserves_aspect(provider, tmp_path):
    out = tmp_path / "upw.mp4"
    provider.enhance(
        EnhancementRequest(source_asset_id="a", profile="UPSCALE", target_width=640),
        output_path=str(out),
    )
    info = _probe(out)
    assert info["width"] == 640 and info["height"] == 480


@requires_ffmpeg
def test_physical_cleanup_profile_runs_cpu_only(provider, tmp_path):
    out = tmp_path / "cln.mp4"
    res = provider.enhance(
        EnhancementRequest(source_asset_id="a", profile="CLEANUP"), output_path=str(out)
    )
    assert _probe(out)["width"] == 320
    assert res["parameters"]["video_codec"] == VIDEO_CODEC


@requires_ffmpeg
def test_physical_custom_profile(provider, tmp_path):
    out = tmp_path / "cus.mp4"
    res = provider.enhance(
        EnhancementRequest(
            source_asset_id="a", profile="CUSTOM",
            sharpen_strength=1.0, contrast=1.1, denoise_strength=0.5,
        ),
        output_path=str(out),
    )
    assert _probe(out)["width"] == 320
    assert "sharpen_strength" in res["parameters"]


@requires_ffmpeg
def test_physical_preserve_audio_true(provider, tmp_path):
    out = tmp_path / "a1.mp4"
    provider.enhance(
        EnhancementRequest(source_asset_id="a", preserve_audio=True), output_path=str(out)
    )
    assert _probe(out)["has_audio"] is True


@requires_ffmpeg
def test_physical_preserve_audio_false_strips_audio(provider, tmp_path):
    out = tmp_path / "a0.mp4"
    res = provider.enhance(
        EnhancementRequest(source_asset_id="a", preserve_audio=False), output_path=str(out)
    )
    assert _probe(out)["has_audio"] is False
    assert res["has_audio"] is False


@requires_ffmpeg
def test_physical_source_without_audio(physical_source_noaudio, tmp_path):
    p = FFmpegEnhancementProvider(resolve_source_path=lambda _a: physical_source_noaudio)
    out = tmp_path / "na.mp4"
    res = p.enhance(EnhancementRequest(source_asset_id="a"), output_path=str(out))
    assert res["has_audio"] is False
    assert _probe(out)["has_audio"] is False


@requires_ffmpeg
def test_physical_output_hash_is_full_file_not_first_mib(provider, tmp_path):
    out = tmp_path / "h.mp4"
    res = provider.enhance(EnhancementRequest(source_asset_id="a"), output_path=str(out))
    whole = hashlib.sha256(out.read_bytes()).hexdigest()
    first_mib = hashlib.sha256(out.read_bytes()[: 1024 * 1024]).hexdigest()
    assert res["output_hash"] == whole
    if out.stat().st_size > 1024 * 1024:
        assert res["output_hash"] != first_mib


@requires_ffmpeg
def test_physical_refuses_to_write_over_source(provider, physical_source):
    with pytest.raises(FFmpegEnhancementError, match="source"):
        provider.enhance(
            EnhancementRequest(source_asset_id="a"), output_path=str(physical_source)
        )


# --------------------------------------------------------------------------- #
# AssetTask lifecycle                                                         #
# --------------------------------------------------------------------------- #

def test_task_starts_in_estimating(db_session, physical_source):
    project, asset = _new_project_with_source(db_session, physical_source)
    task = create_enhancement_task(
        db_session, project_id=project.id,
        request=EnhancementRequest(source_asset_id=asset.id),
    )
    assert task.status == AssetTaskStatus.ESTIMATING.value
    assert task.kind == "ENHANCEMENT"
    assert task.request_hash == EnhancementRequest(source_asset_id=asset.id).request_hash()


@requires_ffmpeg
def test_estimate_moves_to_awaiting_approval_with_local_zero(db_session, physical_source):
    project, asset = _new_project_with_source(db_session, physical_source)
    task = create_enhancement_task(
        db_session, project_id=project.id,
        request=EnhancementRequest(source_asset_id=asset.id),
    )
    task = attach_estimate(db_session, task.id)
    assert task.status == AssetTaskStatus.AWAITING_APPROVAL.value
    assert task.cost_estimate["cost_class"] == "local"
    assert task.cost_estimate["monetary_cost_usd"] == 0.0


def test_approval_requires_matching_request_hash(db_session, physical_source):
    project, asset = _new_project_with_source(db_session, physical_source)
    task = create_enhancement_task(
        db_session, project_id=project.id,
        request=EnhancementRequest(source_asset_id=asset.id),
    )
    with pytest.raises(ValueError, match="approval hash"):
        approve_task(db_session, task.id, request_hash="deadbeef")
    task = approve_task(db_session, task.id, request_hash=task.request_hash)
    assert task.approval_state == AssetTaskApprovalState.APPROVED.value


def test_changed_request_voids_existing_approval(db_session, physical_source):
    project, asset = _new_project_with_source(db_session, physical_source)
    task = create_enhancement_task(
        db_session, project_id=project.id,
        request=EnhancementRequest(source_asset_id=asset.id),
    )
    approve_task(db_session, task.id, request_hash=task.request_hash)
    task = update_request(
        db_session, task.id,
        request=EnhancementRequest(source_asset_id=asset.id, sharpen_strength=1.5),
    )
    assert task.approval_state == AssetTaskApprovalState.PENDING.value
    assert task.approved_request_hash is None


@requires_ffmpeg
def test_queue_blocked_without_valid_approval(db_session, physical_source):
    project, asset = _new_project_with_source(db_session, physical_source)
    task = create_enhancement_task(
        db_session, project_id=project.id,
        request=EnhancementRequest(source_asset_id=asset.id),
    )
    attach_estimate(db_session, task.id)
    with pytest.raises(InvalidTaskTransition):
        queue_task(db_session, task.id)


def test_queue_blocked_when_estimate_missing(db_session, physical_source):
    project, asset = _new_project_with_source(db_session, physical_source)
    task = create_enhancement_task(
        db_session, project_id=project.id,
        request=EnhancementRequest(source_asset_id=asset.id),
    )
    approve_task(db_session, task.id, request_hash=task.request_hash)
    # Force the AWAITING_APPROVAL state without an estimate.
    task.status = AssetTaskStatus.AWAITING_APPROVAL.value
    db_session.add(task)
    db_session.commit()
    with pytest.raises(InvalidTaskTransition, match="cost estimate"):
        queue_task(db_session, task.id)


# --- full physical lifecycle, executed once and shared -------------------- #

@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    if not _HAS_FFMPEG:
        pytest.skip("FFmpeg no disponible")
    src = tmp_path_factory.mktemp("lc") / "src.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=4",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(src),
        ],
        capture_output=True, text=True, check=True,
    )
    src_hash = hashlib.sha256(src.read_bytes()).hexdigest()
    init_db()
    started = time.monotonic()
    with Session(engine) as session:
        project, asset = _new_project_with_source(session, src)
        base_version = project.current_version
        request = EnhancementRequest(source_asset_id=asset.id, profile="DEFAULT")
        task = create_enhancement_task(session, project_id=project.id, request=request)
        task = attach_estimate(session, task.id)
        task = approve_task(session, task.id, request_hash=task.request_hash)
        task = queue_task(session, task.id)
        task = execute_enhancement_task(session, task.id)
        result = {
            "task_id": task.id,
            "project_id": project.id,
            "source_asset_id": asset.id,
            "src_path": src,
            "src_hash": src_hash,
            "base_version": base_version,
            "elapsed": time.monotonic() - started,
        }
    return result


@requires_ffmpeg
def test_lifecycle_reaches_completed(completed):
    with Session(engine) as session:
        task = session.get(AssetTask, completed["task_id"])
    assert task.status == AssetTaskStatus.COMPLETED.value
    assert task.progress == 1.0
    assert task.output_asset_id is not None
    assert task.error is None


@requires_ffmpeg
def test_lifecycle_creates_new_enhanced_asset(completed):
    with Session(engine) as session:
        task = session.get(AssetTask, completed["task_id"])
        enhanced = session.get(EditorAsset, task.output_asset_id)
    assert enhanced.id != completed["source_asset_id"]
    prov = read_asset_provenance(enhanced.asset_metadata)
    assert prov.origin == AssetOrigin.ENHANCED
    assert prov.semantic_role == AssetSemanticRole.ILLUSTRATIVE


@requires_ffmpeg
def test_lifecycle_lineage_points_back_to_source(completed):
    with Session(engine) as session:
        task = session.get(AssetTask, completed["task_id"])
        enhanced = session.get(EditorAsset, task.output_asset_id)
    prov = read_asset_provenance(enhanced.asset_metadata)
    assert prov.parent_asset_id == completed["source_asset_id"]
    assert prov.root_source_asset_id == completed["source_asset_id"]


@requires_ffmpeg
def test_lifecycle_records_full_hash_size_and_tech_metadata(completed):
    with Session(engine) as session:
        task = session.get(AssetTask, completed["task_id"])
        enhanced = session.get(EditorAsset, task.output_asset_id)
    prov = read_asset_provenance(enhanced.asset_metadata)
    out_file = Path(enhanced.original_path)
    assert prov.output_hash == compute_full_file_sha256(out_file)
    assert prov.output_size_bytes == out_file.stat().st_size
    assert prov.provider == "ffmpeg"
    assert prov.operation_type == "enhancement"
    assert prov.operation_version == "enh_v1"
    assert prov.request_hash == task.request_hash
    assert prov.width == 320 and prov.height == 240
    assert prov.duration_us > 0 and prov.fps > 0
    # output_hash is the whole-file digest, not EditorAsset's first-MiB fingerprint
    assert prov.output_hash == hashlib.sha256(out_file.read_bytes()).hexdigest()
    assert enhanced.fingerprint == hashlib.sha256(out_file.read_bytes()[: 1024 * 1024]).hexdigest()


@requires_ffmpeg
def test_lifecycle_leaves_source_asset_untouched(completed):
    with Session(engine) as session:
        src = session.get(EditorAsset, completed["source_asset_id"])
    assert read_asset_provenance(src.asset_metadata) is None
    assert src.original_path == str(completed["src_path"])
    assert hashlib.sha256(completed["src_path"].read_bytes()).hexdigest() == completed["src_hash"]


@requires_ffmpeg
def test_lifecycle_does_not_mutate_timeline(completed):
    with Session(engine) as session:
        from app.editor.models import EditorProject

        project = session.get(EditorProject, completed["project_id"])
        assert project.current_version == completed["base_version"]
        version = editor_service.current_version(session, project)
        referenced = {
            clip.get("asset_id")
            for track in version.timeline.get("tracks", [])
            for clip in track.get("clips", [])
        }
        task = session.get(AssetTask, completed["task_id"])
    assert task.output_asset_id not in referenced


@requires_ffmpeg
def test_lifecycle_no_via_or_evidence_side_effects(completed):
    with Session(engine) as session:
        task = session.get(AssetTask, completed["task_id"])
        enhanced = session.get(EditorAsset, task.output_asset_id)
    assert set(enhanced.asset_metadata) == {"asset_provenance"}


@requires_ffmpeg
def test_completed_task_cannot_be_rerun(completed):
    with Session(engine) as session, pytest.raises(InvalidTaskTransition):
        execute_enhancement_task(session, completed["task_id"])


# --- failure / atomicity ------------------------------------------------- #

class _ExplodingProvider(FFmpegEnhancementProvider):
    def enhance(self, request, *, output_path):  # noqa: ARG002
        raise FFmpegEnhancementError("simulated ffmpeg failure")


@requires_ffmpeg
def test_failure_marks_task_failed_without_phantom_asset(db_session, physical_source):
    project, asset = _new_project_with_source(db_session, physical_source)
    request = EnhancementRequest(source_asset_id=asset.id)
    task = create_enhancement_task(db_session, project_id=project.id, request=request)
    task = attach_estimate(db_session, task.id)
    task = approve_task(db_session, task.id, request_hash=task.request_hash)
    task = queue_task(db_session, task.id)
    assets_before = len(editor_service.project_assets(db_session, project.id))

    task = execute_enhancement_task(db_session, task.id, provider=_ExplodingProvider())

    assert task.status == AssetTaskStatus.FAILED.value
    assert task.output_asset_id is None
    assert "simulated ffmpeg failure" in task.error
    assert len(editor_service.project_assets(db_session, project.id)) == assets_before
    out_path = default_enhanced_output_path(project.id, task.id, "DEFAULT")
    assert not out_path.exists()
    assert not out_path.with_suffix(".partial" + out_path.suffix).exists()


@requires_ffmpeg
def test_execute_rejects_when_hash_no_longer_matches_approval(db_session, physical_source):
    project, asset = _new_project_with_source(db_session, physical_source)
    request = EnhancementRequest(source_asset_id=asset.id)
    task = create_enhancement_task(db_session, project_id=project.id, request=request)
    task = attach_estimate(db_session, task.id)
    task = approve_task(db_session, task.id, request_hash=task.request_hash)
    task = queue_task(db_session, task.id)
    # Tamper: the request hash drifts away from what was approved.
    task.request_hash = "0" * 64
    db_session.add(task)
    db_session.commit()
    with pytest.raises(InvalidTaskTransition, match="approval"):
        execute_enhancement_task(db_session, task.id)


# --- legacy source provenance ------------------------------------------- #

@requires_ffmpeg
def test_legacy_source_without_provenance_is_treated_as_source(db_session, physical_source):
    project, asset = _new_project_with_source(db_session, physical_source, metadata={})
    request = EnhancementRequest(source_asset_id=asset.id)
    task = create_enhancement_task(db_session, project_id=project.id, request=request)
    task = attach_estimate(db_session, task.id)
    task = approve_task(db_session, task.id, request_hash=task.request_hash)
    task = queue_task(db_session, task.id)
    task = execute_enhancement_task(db_session, task.id)
    assert task.status == AssetTaskStatus.COMPLETED.value
    with Session(engine) as s:
        enhanced = s.get(EditorAsset, task.output_asset_id)
    prov = read_asset_provenance(enhanced.asset_metadata)
    assert prov.root_source_asset_id == asset.id


@requires_ffmpeg
def test_ambiguous_source_provenance_fails_closed(db_session, physical_source):
    project, asset = _new_project_with_source(
        db_session, physical_source, metadata={"origin": "SOMETHING_WEIRD"}
    )
    request = EnhancementRequest(source_asset_id=asset.id)
    task = create_enhancement_task(db_session, project_id=project.id, request=request)
    task = attach_estimate(db_session, task.id)
    task = approve_task(db_session, task.id, request_hash=task.request_hash)
    task = queue_task(db_session, task.id)
    task = execute_enhancement_task(db_session, task.id)
    assert task.status == AssetTaskStatus.FAILED.value
    assert "provenance" in task.error.lower()
    assert task.output_asset_id is None


def test_ambiguous_source_helper_raises_directly(db_session, physical_source):
    from app.editor.asset_task_service import _source_provenance_for

    _project, asset = _new_project_with_source(
        db_session, physical_source, metadata={"origin": "SOMETHING_WEIRD"}
    )
    with pytest.raises(AmbiguousSourceProvenance):
        _source_provenance_for(db_session, asset.id)


# --------------------------------------------------------------------------- #
# Block 02R - atomic asset/task finalisation                                  #
# --------------------------------------------------------------------------- #

def _queued_task(session: Session, project, asset, **req_kw) -> AssetTask:
    request = EnhancementRequest(source_asset_id=asset.id, **req_kw)
    task = create_enhancement_task(session, project_id=project.id, request=request)
    task = attach_estimate(session, task.id)
    task = approve_task(session, task.id, request_hash=task.request_hash)
    return queue_task(session, task.id)


def _enhanced_asset_count(session: Session, project_id: str) -> int:
    n = 0
    for a in editor_service.project_assets(session, project_id):
        prov = read_asset_provenance(a.asset_metadata)
        if prov is not None and prov.origin == AssetOrigin.ENHANCED:
            n += 1
    return n


class _OneShotRaiser:
    """Raises RuntimeError on its first call, then delegates to the real fn."""

    def __init__(self, real):
        self._real = real
        self.fired = False

    def __call__(self, *a, **kw):
        if not self.fired:
            self.fired = True
            raise RuntimeError("injected finalisation-commit failure")
        return self._real(*a, **kw)


@requires_ffmpeg
def test_finalization_commit_failure_rolls_back_no_phantom_asset(
    db_session, physical_source, monkeypatch
):
    project, asset = _new_project_with_source(db_session, physical_source)
    src_hash = hashlib.sha256(physical_source.read_bytes()).hexdigest()
    task = _queued_task(db_session, project, asset)

    monkeypatch.setattr(ats, "_finalize_commit", _OneShotRaiser(ats._finalize_commit))
    task = execute_enhancement_task(db_session, task.id)

    # Failure recorded via a real RUNNING -> FAILED transition.
    assert task.status == AssetTaskStatus.FAILED.value
    assert task.output_asset_id is None
    assert task.progress == 0.0
    assert task.error and "injected finalisation-commit failure" in task.error

    # Nothing durable leaked: re-read in a fresh session.
    with Session(engine) as fresh:
        durable = fresh.get(AssetTask, task.id)
        assert durable.status == AssetTaskStatus.FAILED.value
        assert durable.output_asset_id is None
        assert _enhanced_asset_count(fresh, project.id) == 0

    out_path = default_enhanced_output_path(project.id, task.id, "DEFAULT")
    assert not out_path.exists()
    assert not out_path.with_suffix(".partial" + out_path.suffix).exists()
    assert hashlib.sha256(physical_source.read_bytes()).hexdigest() == src_hash


@requires_ffmpeg
def test_failure_after_flush_before_commit_rolls_back(db_session, physical_source, monkeypatch):
    project, asset = _new_project_with_source(db_session, physical_source)
    task = _queued_task(db_session, project, asset)

    def _boom() -> None:
        raise RuntimeError("crash between asset flush and finalisation commit")

    monkeypatch.setattr(ats, "_pre_commit_barrier", _boom)
    task = execute_enhancement_task(db_session, task.id)

    assert task.status == AssetTaskStatus.FAILED.value
    assert task.output_asset_id is None
    assert "crash between asset flush" in task.error
    with Session(engine) as fresh:
        assert fresh.get(AssetTask, task.id).status == AssetTaskStatus.FAILED.value
        assert _enhanced_asset_count(fresh, project.id) == 0
    out_path = default_enhanced_output_path(project.id, task.id, "DEFAULT")
    assert not out_path.exists()


@requires_ffmpeg
def test_success_path_is_a_single_finalisation_commit(db_session, physical_source, monkeypatch):
    project, asset = _new_project_with_source(db_session, physical_source)
    task = _queued_task(db_session, project, asset)

    real_commit = db_session.commit
    commits: list[int] = []

    def _counting_commit():
        commits.append(1)
        return real_commit()

    monkeypatch.setattr(db_session, "commit", _counting_commit)
    task = execute_enhancement_task(db_session, task.id)

    assert task.status == AssetTaskStatus.COMPLETED.value
    assert task.output_asset_id is not None
    # Exactly two durable commits during execution: the RUNNING checkpoint and
    # the single atomic finalisation (asset row + task completion together).
    assert len(commits) == 2

    with Session(engine) as fresh:
        durable_task = fresh.get(AssetTask, task.id)
        assert durable_task.status == AssetTaskStatus.COMPLETED.value
        enhanced = fresh.get(EditorAsset, durable_task.output_asset_id)
        assert enhanced is not None
        assert read_asset_provenance(enhanced.asset_metadata).origin == AssetOrigin.ENHANCED


@requires_ffmpeg
def test_no_second_commit_regression(db_session, physical_source, monkeypatch):
    """Asset registration must not commit on its own - it is folded into the
    finalisation commit. A reintroduced add_asset commit would make this 3."""
    project, asset = _new_project_with_source(db_session, physical_source)
    task = _queued_task(db_session, project, asset)

    recorded: list[bool] = []
    real_add_asset = editor_service.add_asset

    def _spy_add_asset(*a, **kw):
        recorded.append(kw.get("commit", True))
        return real_add_asset(*a, **kw)

    monkeypatch.setattr(ats.editor_service, "add_asset", _spy_add_asset)

    real_commit = db_session.commit
    commits: list[int] = []

    def _counting_commit():
        commits.append(1)
        return real_commit()

    monkeypatch.setattr(db_session, "commit", _counting_commit)

    task = execute_enhancement_task(db_session, task.id)

    assert task.status == AssetTaskStatus.COMPLETED.value
    assert recorded == [False]  # add_asset called once, without its own commit
    assert len(commits) == 2  # checkpoint + single finalisation, never 3


@requires_ffmpeg
def test_atomic_success_source_untouched_and_no_timeline_change(db_session, physical_source):
    project, asset = _new_project_with_source(db_session, physical_source)
    src_hash = hashlib.sha256(physical_source.read_bytes()).hexdigest()
    base_version = project.current_version
    task = _queued_task(db_session, project, asset)
    task = execute_enhancement_task(db_session, task.id)

    assert task.status == AssetTaskStatus.COMPLETED.value
    with Session(engine) as fresh:
        src = fresh.get(EditorAsset, asset.id)
        assert read_asset_provenance(src.asset_metadata) is None
        proj = fresh.get(EditorProject, project.id)
        assert proj.current_version == base_version
    assert hashlib.sha256(physical_source.read_bytes()).hexdigest() == src_hash


def test_add_asset_default_commit_persists_immediately(db_session, physical_source):
    project = editor_service.create_project(db_session, "bc-default")
    created = editor_service.add_asset(db_session, project, physical_source, duration_us=1)
    with Session(engine) as other:
        assert other.get(EditorAsset, created.id) is not None


def test_add_asset_commit_false_defers_persistence(db_session, physical_source):
    project = editor_service.create_project(db_session, "bc-deferred")
    created = editor_service.add_asset(
        db_session, project, physical_source, duration_us=1, commit=False
    )
    # Flushed: visible in this session, primary key populated.
    assert created.id
    assert db_session.get(EditorAsset, created.id) is not None
    # Not committed: a rollback erases it entirely.
    db_session.rollback()
    assert db_session.get(EditorAsset, created.id) is None


# --------------------------------------------------------------------------- #
# Block 02R2 - post-commit boundary + bounded fingerprint read                #
# --------------------------------------------------------------------------- #

first_mib_fingerprint = editor_service.first_mib_fingerprint


@requires_ffmpeg
def test_post_commit_refresh_failure_keeps_task_completed(
    db_session, physical_source, monkeypatch
):
    """A read-back failure AFTER the finalisation commit succeeded must not
    roll back and must never become COMPLETED -> FAILED."""
    project, asset = _new_project_with_source(db_session, physical_source)
    src_hash = hashlib.sha256(physical_source.read_bytes()).hexdigest()
    task = _queued_task(db_session, project, asset)

    real_refresh = db_session.refresh
    fired = {"v": False}

    def flaky_refresh(obj, *a, **k):
        if (
            not fired["v"]
            and isinstance(obj, AssetTask)
            and getattr(obj, "status", None) == AssetTaskStatus.COMPLETED.value
        ):
            fired["v"] = True
            raise RuntimeError("injected post-commit refresh failure")
        return real_refresh(obj, *a, **k)

    monkeypatch.setattr(db_session, "refresh", flaky_refresh)

    task = execute_enhancement_task(db_session, task.id)

    assert fired["v"] is True  # the post-commit refresh really was exercised
    assert task.status == AssetTaskStatus.COMPLETED.value  # never FAILED
    assert task.output_asset_id is not None

    with Session(engine) as fresh:
        durable = fresh.get(AssetTask, task.id)
        assert durable.status == AssetTaskStatus.COMPLETED.value
        assert durable.output_asset_id is not None
        assert durable.error is None
        enhanced = fresh.get(EditorAsset, durable.output_asset_id)
        assert enhanced is not None
        assert read_asset_provenance(enhanced.asset_metadata).origin == AssetOrigin.ENHANCED
        assert Path(enhanced.original_path).is_file()
    # source untouched
    assert hashlib.sha256(physical_source.read_bytes()).hexdigest() == src_hash


@requires_ffmpeg
def test_post_commit_total_read_failure_raises_distinct_error_not_failed(
    db_session, physical_source, monkeypatch
):
    """If even the fallback re-read fails, a distinguishable PostCommitReadError
    is raised - the task stays durably COMPLETED, never FAILED."""
    project, asset = _new_project_with_source(db_session, physical_source)
    task = _queued_task(db_session, project, asset)
    task_id = task.id

    def _no_refresh(obj, *a, **k):
        raise RuntimeError("refresh down")

    def _no_get(model, ident, *a, **k):
        raise RuntimeError("get down")

    real_refresh = db_session.refresh

    def flaky_refresh(obj, *a, **k):
        if isinstance(obj, AssetTask) and getattr(obj, "status", None) == "COMPLETED":
            monkeypatch.setattr(db_session, "get", _no_get)
            raise RuntimeError("refresh down")
        return real_refresh(obj, *a, **k)

    monkeypatch.setattr(db_session, "refresh", flaky_refresh)

    with pytest.raises(PostCommitReadError):
        execute_enhancement_task(db_session, task_id)

    monkeypatch.undo()
    with Session(engine) as fresh:
        durable = fresh.get(AssetTask, task_id)
        assert durable.status == AssetTaskStatus.COMPLETED.value
        assert durable.output_asset_id is not None
        assert fresh.get(EditorAsset, durable.output_asset_id) is not None


def test_fingerprint_is_sha256_of_first_mib_small_file(db_session, tmp_path):
    small = tmp_path / "small.bin"
    small.write_bytes(b"snipra-small-fixture" * 7)  # < 1 MiB
    expected = hashlib.sha256(small.read_bytes()).hexdigest()
    project = editor_service.create_project(db_session, "fp-small")
    asset = editor_service.add_asset(db_session, project, small, duration_us=1)
    assert asset.fingerprint == expected
    assert first_mib_fingerprint(small) == expected


def test_fingerprint_ignores_bytes_past_first_mib(db_session, tmp_path):
    from app.editor.asset_provenance import compute_full_file_sha256

    big = tmp_path / "big.bin"
    head = b"H" * (1024 * 1024)
    big.write_bytes(head + b"ORIGINAL-TAIL")
    fp_before = first_mib_fingerprint(big)
    full_before = compute_full_file_sha256(big)
    assert fp_before == hashlib.sha256(head).hexdigest()

    project = editor_service.create_project(db_session, "fp-big")
    a1 = editor_service.add_asset(db_session, project, big, duration_us=1)
    assert a1.fingerprint == fp_before

    # Rewrite ONLY the tail (bytes past the first MiB).
    with big.open("r+b") as fh:
        fh.seek(1024 * 1024)
        fh.write(b"CHANGED-TAIL!")

    a2 = editor_service.add_asset(db_session, project, big, duration_us=1)
    assert a2.fingerprint == fp_before  # first-MiB fingerprint unchanged
    assert compute_full_file_sha256(big) != full_before  # full-file hash DID change


def test_add_asset_does_not_read_whole_file(db_session, tmp_path, monkeypatch):
    big = tmp_path / "noload.bin"
    big.write_bytes(b"x" * (1024 * 1024 + 4096))
    expected = hashlib.sha256(b"x" * (1024 * 1024)).hexdigest()

    def _forbidden(self, *a, **k):
        raise AssertionError("add_asset must not call Path.read_bytes()")

    monkeypatch.setattr(Path, "read_bytes", _forbidden)
    project = editor_service.create_project(db_session, "no-read-bytes")
    asset = editor_service.add_asset(db_session, project, big, duration_us=1)
    assert asset.fingerprint == expected


def test_add_asset_source_has_no_read_bytes_call():
    src = Path(editor_service.__file__).read_text(encoding="utf-8")
    start = src.index("def add_asset(")
    end = src.index("\ndef ", start)
    assert "read_bytes()" not in src[start:end]


# --------------------------------------------------------------------------- #
# Module hygiene                                                              #
# --------------------------------------------------------------------------- #

def test_provider_module_has_no_shell_true_and_no_network():
    src = Path(
        __import__("app.editor.ffmpeg_enhancement", fromlist=["__file__"]).__file__
    ).read_text(encoding="utf-8")
    assert "shell=True" not in src
    for banned in ("import requests", "import httpx", "openai", "anthropic", "realesrgan"):
        assert banned not in src.lower()


def test_task_service_makes_no_timeline_or_via_calls():
    src = Path(
        __import__("app.editor.asset_task_service", fromlist=["__file__"]).__file__
    ).read_text(encoding="utf-8")
    assert "apply_operation" not in src
    assert "via_evidence" not in src and "run_via" not in src
    assert "shell=True" not in src
