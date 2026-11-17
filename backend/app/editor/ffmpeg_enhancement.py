"""Local FFmpeg video enhancement provider (SNIPRA 1.6C Block 02).

Implements the `VideoEnhancementProvider` contract frozen in Block 01 with a
CPU-only, deterministic FFmpeg pipeline. No Real-ESRGAN, no generative B-roll,
no GPU requirement, no external/API calls. The concrete filter graph is built
purely from typed enums / validated numbers / module constants - never from
free text, and never through a shell (`app.core.media.run` uses argv only).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from app.core import media
from app.core.config import settings
from app.core.sanitize import sanitize_error_text
from app.editor.asset_provenance import compute_full_file_sha256
from app.editor.asset_tasks import EnhancementValidationError
from app.editor.enhancement import (
    ENHANCEMENT_CONTRACT_VERSION,
    EnhancementCapabilities,
    EnhancementEstimate,
    EnhancementProfile,
    EnhancementRequest,
    VideoEnhancementProvider,
)
from app.providers.capabilities import CostClass

# --------------------------------------------------------------------------- #
# Constants - the entire "how" lives here, centralised and test-pinned.        #
# --------------------------------------------------------------------------- #

#: Portable CPU baseline. No NVENC, no GPU filter is ever required.
VIDEO_CODEC = "libx264"
VIDEO_PRESET = "medium"
VIDEO_CRF = "18"
PIXEL_FORMAT = "yuv420p"
OUTPUT_CONTAINER = "mp4"

#: Audio: copy when the source codec is already MP4-safe, else deterministic
#: transcode to AAC at this bitrate.
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "160k"
_MP4_SAFE_AUDIO = frozenset({"aac", "mp3", "ac3", "eac3", "alac"})

MAX_DIMENSION = 15360  # mirrors EnhancementRequest's target_* upper bound
MIN_DIMENSION = 2

# DEFAULT profile - "local deterministic enhancement", deliberately gentle.
_DEFAULT_UNSHARP = "unsharp=5:5:0.6:5:5:0.0"
_DEFAULT_EQ = "eq=contrast=1.03:saturation=1.04:brightness=0.0"

# CLEANUP profile - conservative CPU denoise + a touch of sharpening.
_CLEANUP_DENOISE_BASE = 1.6  # hqdn3d luma spatial at denoise_strength == 1.0
_CLEANUP_UNSHARP = "unsharp=5:5:0.3:5:5:0.0"

# UPSCALE profile - Lanczos resample + light detail recovery.
_UPSCALE_UNSHARP = "unsharp=5:5:0.5:5:5:0.0"

_SCALE_FLAGS = "flags=lanczos"

#: A filter chain may only contain this alphabet once built. No `;` (graph
#: separator), no `|`, `&`, `$`, backtick, quotes, whitespace or brackets.
_SAFE_CHAIN_RE = re.compile(r"^[A-Za-z0-9_=:,.*/()+-]*$")


class FFmpegEnhancementUnavailable(RuntimeError):
    """FFmpeg / ffprobe binary is missing or not runnable. Fail closed."""


class FFmpegEnhancementError(RuntimeError):
    """The FFmpeg run failed, or its output did not pass media validation."""


def _fmt(value: float) -> str:
    """Format a validated float for a filter argument (never user text)."""
    return f"{float(value):.4f}".rstrip("0").rstrip(".") or "0"


def _even(n: float, *, lo: int = MIN_DIMENSION, hi: int = MAX_DIMENSION) -> int:
    """Clamp to the contract range and round down to an even integer (yuv420p)."""
    i = int(round(n))
    i = max(lo, min(hi, i))
    return i - (i % 2)


def _default_resolve_source_path(asset_id: str) -> Path:
    """Look up EditorAsset.original_path. Kept out of the pure pipeline so tests
    can inject a plain callable."""
    from sqlmodel import Session

    from app.core.db import engine
    from app.editor.models import EditorAsset

    with Session(engine) as session:
        asset = session.get(EditorAsset, asset_id)
        if asset is None:
            raise FFmpegEnhancementError(f"source asset {asset_id!r} not found")
        return Path(asset.original_path)


class FFmpegEnhancementProvider(VideoEnhancementProvider):
    """CPU-only FFmpeg enhancement engine bound to the Block 01 contracts."""

    def __init__(
        self,
        *,
        ffmpeg_path: str | None = None,
        ffprobe_path: str | None = None,
        resolve_source_path: Callable[[str], Path] | None = None,
        run_timeout: int = 3600,
    ) -> None:
        self._ffmpeg = ffmpeg_path or settings.ffmpeg_path
        self._ffprobe = ffprobe_path or settings.ffprobe_path
        self._resolve_source_path = resolve_source_path or _default_resolve_source_path
        self._run_timeout = run_timeout

    # ----------------------------------------------------------------- #
    # Engine identity                                                    #
    # ----------------------------------------------------------------- #

    def _binary_status(self, path: str) -> tuple[bool, str]:
        """Mirror of app.core.deps._check but bound to this instance's configured
        path. Runs `<bin> -version` locally; never touches the network."""
        resolved = shutil.which(path) or (path if Path(path).exists() else None)
        if not resolved:
            return False, ""
        try:
            out = subprocess.run(
                [resolved, "-version"], capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return False, ""
        if out.returncode != 0:
            return False, ""
        version = out.stdout.splitlines()[0] if out.stdout else ""
        return True, version

    def is_available(self) -> bool:
        return self._binary_status(self._ffmpeg)[0] and self._binary_status(self._ffprobe)[0]

    def engine_version(self) -> str:
        """First line of `ffmpeg -version`. No network. Empty when unavailable."""
        return self._binary_status(self._ffmpeg)[1]

    def capabilities(self) -> EnhancementCapabilities:
        return EnhancementCapabilities(
            engine_name="ffmpeg",
            engine_version=self.engine_version(),
            supports_upscale=True,
            supports_sharpen=True,
            supports_color_adjustment=True,
            supports_denoise=True,
            supports_gpu=False,
            requires_external_binary=True,
        )

    def _require_binary(self) -> None:
        if not self.is_available():
            raise FFmpegEnhancementUnavailable(
                "FFmpeg/ffprobe is not available; local enhancement cannot run"
            )

    # ----------------------------------------------------------------- #
    # Request validation (no side effects, no execution)                 #
    # ----------------------------------------------------------------- #

    def validate_request(self, request: EnhancementRequest) -> None:
        caps = self.capabilities()
        if not caps.supports_profile(request.profile):
            raise EnhancementValidationError(f"engine cannot satisfy profile {request.profile}")
        if request.profile == EnhancementProfile.CUSTOM and not self._has_any_adjustment(request):
            raise EnhancementValidationError("CUSTOM profile needs at least one typed adjustment")

    @staticmethod
    def _has_any_adjustment(request: EnhancementRequest) -> bool:
        return any(
            v is not None
            for v in (
                request.target_width,
                request.target_height,
                request.upscale_factor,
                request.sharpen_strength,
                request.contrast,
                request.brightness,
                request.saturation,
                request.denoise_strength,
            )
        )

    # ----------------------------------------------------------------- #
    # Estimate - always LOCAL, always $0                                 #
    # ----------------------------------------------------------------- #

    def estimate(self, request: EnhancementRequest) -> EnhancementEstimate:
        seconds: int | None = None
        note = "local FFmpeg (CPU); no monetary cost"
        try:
            src = self._resolve_source_path(request.source_asset_id)
            info = media.probe(src)
            factor = 4.0 if request.profile == EnhancementProfile.UPSCALE else 2.0
            seconds = max(1, int(info.duration * factor))
        except Exception as exc:  # noqa: BLE001 - estimate must not raise
            note = f"local FFmpeg (CPU); {sanitize_error_text(str(exc))[:120]}"
        return EnhancementEstimate(
            cost_class=CostClass.LOCAL,
            monetary_cost_known=True,
            monetary_cost_usd=0.0,
            estimated_seconds=seconds,
            notes=note,
        )

    # ----------------------------------------------------------------- #
    # Deterministic plan                                                 #
    # ----------------------------------------------------------------- #

    def prepare(self, request: EnhancementRequest) -> dict:
        src = self._resolve_source_path(request.source_asset_id)
        info = media.probe(src)
        src_w, src_h = int(info.width), int(info.height)
        if src_w <= 0 or src_h <= 0:
            raise FFmpegEnhancementError("could not read source video dimensions")

        target_w, target_h = self._resolve_dimensions(request, src_w, src_h)
        chain = self._build_filter_chain(request, src_w, src_h, target_w, target_h)
        if not _SAFE_CHAIN_RE.match(chain):
            raise FFmpegEnhancementError(f"refusing unsafe filter chain: {chain!r}")

        src_codec = self._audio_codec(src) if info.has_audio else None
        keep_audio = bool(request.preserve_audio and info.has_audio)
        audio_args = self._audio_args(keep_audio, src_codec)

        return {
            "contract_version": ENHANCEMENT_CONTRACT_VERSION,
            "request_hash": request.request_hash(),
            "source_asset_id": request.source_asset_id,
            "source_path": str(src),
            "profile": request.profile.value,
            "src_width": src_w,
            "src_height": src_h,
            "src_has_audio": info.has_audio,
            "src_audio_codec": src_codec,
            "target_width": target_w,
            "target_height": target_h,
            "filter_chain": chain,
            "video_args": [
                "-c:v", VIDEO_CODEC, "-preset", VIDEO_PRESET,
                "-crf", VIDEO_CRF, "-pix_fmt", PIXEL_FORMAT,
            ],
            "audio_args": audio_args,
            "keep_audio": keep_audio,
            "container": OUTPUT_CONTAINER,
        }

    def _resolve_dimensions(
        self, request: EnhancementRequest, src_w: int, src_h: int
    ) -> tuple[int, int]:
        """Final even (width, height). Aspect ratio preserved unless the caller
        pins both dimensions explicitly."""
        if request.upscale_factor is not None:
            f = float(request.upscale_factor)
            return _even(src_w * f), _even(src_h * f)
        tw, th = request.target_width, request.target_height
        if tw is not None and th is not None:
            return _even(tw), _even(th)
        if tw is not None:
            return _even(tw), _even(src_h * (tw / src_w))
        if th is not None:
            return _even(src_w * (th / src_h)), _even(th)
        # No resize requested - normalise to even so libx264/yuv420p is safe.
        return _even(src_w), _even(src_h)

    def _build_filter_chain(
        self,
        request: EnhancementRequest,
        src_w: int,
        src_h: int,
        target_w: int,
        target_h: int,
    ) -> str:
        resized = (target_w, target_h) != (src_w, src_h)
        parts: list[str] = []
        profile = request.profile

        if resized:
            parts.append(f"scale={target_w}:{target_h}:{_SCALE_FLAGS}")

        if profile == EnhancementProfile.DEFAULT:
            parts += [_DEFAULT_UNSHARP, _DEFAULT_EQ]
        elif profile == EnhancementProfile.UPSCALE:
            parts.append(_UPSCALE_UNSHARP)
        elif profile == EnhancementProfile.CLEANUP:
            strength = 1.0 if request.denoise_strength is None else float(request.denoise_strength)
            luma = _CLEANUP_DENOISE_BASE * max(0.0, strength)
            parts.append(
                f"hqdn3d={_fmt(luma)}:{_fmt(luma * 0.75)}:{_fmt(luma * 4)}:{_fmt(luma * 4)}"
            )
            parts.append(_CLEANUP_UNSHARP)
        else:  # CUSTOM - typed knobs only
            if request.denoise_strength is not None and request.denoise_strength > 0:
                luma = _CLEANUP_DENOISE_BASE * float(request.denoise_strength)
                parts.append(
                    f"hqdn3d={_fmt(luma)}:{_fmt(luma * 0.75)}:{_fmt(luma * 4)}:{_fmt(luma * 4)}"
                )
            if request.sharpen_strength is not None and request.sharpen_strength > 0:
                amount = min(1.5, float(request.sharpen_strength) * 0.5)
                parts.append(f"unsharp=5:5:{_fmt(amount)}:5:5:0.0")
            eq = self._custom_eq(request)
            if eq:
                parts.append(eq)

        # Guarantee an even canvas even when no explicit scale ran.
        if not resized and (src_w % 2 or src_h % 2):
            parts.insert(0, "crop=trunc(iw/2)*2:trunc(ih/2)*2")

        return ",".join(parts)

    @staticmethod
    def _custom_eq(request: EnhancementRequest) -> str:
        terms: list[str] = []
        if request.contrast is not None:
            terms.append(f"contrast={_fmt(request.contrast)}")
        if request.brightness is not None:
            terms.append(f"brightness={_fmt(request.brightness)}")
        if request.saturation is not None:
            terms.append(f"saturation={_fmt(request.saturation)}")
        return f"eq={':'.join(terms)}" if terms else ""

    @staticmethod
    def _audio_args(keep_audio: bool, src_codec: str | None) -> list[str]:
        if not keep_audio:
            return ["-an"]
        if src_codec in _MP4_SAFE_AUDIO:
            return ["-c:a", "copy"]
        return ["-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE]

    def _audio_codec(self, path: str | Path) -> str | None:
        try:
            out = media.run(
                [
                    self._ffprobe, "-v", "error", "-select_streams", "a:0",
                    "-show_entries", "stream=codec_name",
                    "-of", "default=nokey=1:noprint_wrappers=1", str(path),
                ],
                timeout=60,
            )
            value = (out.stdout or "").strip().lower()
            return value or None
        except media.FFmpegError:
            return None

    def _probe_fps(self, path: str | Path) -> float:
        try:
            out = media.run(
                [
                    self._ffprobe, "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=r_frame_rate",
                    "-of", "default=nokey=1:noprint_wrappers=1", str(path),
                ],
                timeout=60,
            )
            raw = (out.stdout or "").strip()
            if "/" in raw:
                num, den = raw.split("/", 1)
                den_f = float(den)
                return round(float(num) / den_f, 3) if den_f else 0.0
            return round(float(raw), 3) if raw else 0.0
        except (media.FFmpegError, ValueError):
            return 0.0

    # ----------------------------------------------------------------- #
    # Execution                                                          #
    # ----------------------------------------------------------------- #

    def enhance(self, request: EnhancementRequest, *, output_path: str) -> dict:
        self._require_binary()
        plan = self.prepare(request)

        src = Path(plan["source_path"]).resolve()
        out = Path(output_path)
        if not src.is_file():
            raise FFmpegEnhancementError(f"source file missing: {src}")
        if out.resolve() == src:
            raise FFmpegEnhancementError("output path must not be the source file")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".partial" + (out.suffix or f".{OUTPUT_CONTAINER}"))

        cmd = [self._ffmpeg, "-y", "-i", str(src)]
        if plan["filter_chain"]:
            cmd += ["-vf", plan["filter_chain"]]
        cmd += ["-map", "0:v:0"]
        if plan["keep_audio"]:
            cmd += ["-map", "0:a:0"]
        cmd += list(plan["video_args"])
        cmd += list(plan["audio_args"])
        cmd += ["-movflags", "+faststart", "-f", OUTPUT_CONTAINER, str(tmp)]

        try:
            media.run(cmd, timeout=self._run_timeout)
            self._verify_output(tmp, plan)
        except media.FFmpegError as exc:
            tmp.unlink(missing_ok=True)
            raise FFmpegEnhancementError(sanitize_error_text(str(exc))) from exc
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        os.replace(tmp, out)

        info = media.probe(out)
        result = {
            "output_path": str(out),
            "output_hash": compute_full_file_sha256(out),
            "output_size_bytes": out.stat().st_size,
            "width": int(info.width),
            "height": int(info.height),
            "duration_us": max(0, round(info.duration * 1_000_000)),
            "fps": self._probe_fps(out),
            "has_audio": info.has_audio,
            "provider": "ffmpeg",
            "model_or_engine": f"{self.engine_version()} / {plan['profile']}",
            "request_hash": plan["request_hash"],
            "operation_version": ENHANCEMENT_CONTRACT_VERSION,
            "parameters": self._safe_parameters(request, plan),
            "filter_chain": plan["filter_chain"],
        }
        return result

    def _verify_output(self, path: Path, plan: dict) -> None:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FFmpegEnhancementError("FFmpeg produced no output file")
        info = media.probe(path)
        if info.width <= 0 or info.height <= 0:
            raise FFmpegEnhancementError("output has no decodable video stream")
        if info.duration <= 0:
            raise FFmpegEnhancementError("output has no duration")
        if (info.width, info.height) != (plan["target_width"], plan["target_height"]):
            raise FFmpegEnhancementError(
                f"output dimensions {info.width}x{info.height} != "
                f"expected {plan['target_width']}x{plan['target_height']}"
            )
        if plan["keep_audio"] and not info.has_audio:
            raise FFmpegEnhancementError("preserve_audio requested but output has no audio")
        if not plan["keep_audio"] and info.has_audio:
            raise FFmpegEnhancementError("audio was disabled but output still carries audio")

    @staticmethod
    def _safe_parameters(request: EnhancementRequest, plan: dict) -> dict:
        """Only normalised, non-secret values. Sourced entirely from the typed
        request + the deterministic plan."""
        params: dict = {
            "profile": plan["profile"],
            "filter_chain": plan["filter_chain"],
            "target_width": plan["target_width"],
            "target_height": plan["target_height"],
            "preserve_audio": request.preserve_audio,
            "video_codec": VIDEO_CODEC,
            "video_preset": VIDEO_PRESET,
            "video_crf": VIDEO_CRF,
            "container": OUTPUT_CONTAINER,
        }
        for knob in (
            "upscale_factor", "sharpen_strength", "contrast",
            "brightness", "saturation", "denoise_strength",
        ):
            value = getattr(request, knob)
            if value is not None:
                params[knob] = value
        return params


def default_enhanced_output_path(project_id: str, task_id: str, profile: str) -> Path:
    """SNIPRA-controlled output location. Never the source, never traversable:
    every path component is a server-side id / enum value."""
    safe_profile = re.sub(r"[^A-Za-z0-9_-]", "", profile) or "enh"
    root = (settings.editor_dir / project_id / "enhanced").resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = (root / f"{task_id}-{safe_profile}.{OUTPUT_CONTAINER}").resolve()
    if root not in target.parents:
        raise FFmpegEnhancementError("resolved output path escaped the project directory")
    return target


def ffmpeg_binary_present() -> bool:
    """Cheap check used by callers before creating a task."""
    return bool(shutil.which(settings.ffmpeg_path)) or Path(settings.ffmpeg_path).exists()
