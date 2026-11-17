"""Video Intelligence provider interface and implementations."""

from __future__ import annotations

import asyncio
import json
import math
import re
from abc import ABC, abstractmethod

from pydantic import BaseModel

from app.core import media
from app.video_intelligence.chunk_executor import run_chunks_bounded
from app.video_intelligence.chunking import (
    AnalysisChunk,
    AnalysisStrategy,
    VideoAnalysisPlan,
    plan_analysis,
)
from app.video_intelligence.models import (
    ConfidenceSource,
    EvidenceNature,
    EvidenceProvenance,
    ExtractedFrame,
    NatureSource,
    TemporalSource,
    VideoAnswer,
    VideoEventCandidate,
    VideoIntelligenceError,
    VideoIntelligenceErrorCode,
    VideoIntelligenceRequest,
    VideoIntelligenceResult,
    VideoObservation,
    VideoProcessingMetadata,
)


class ProviderCapabilities(BaseModel):
    ask_video: bool = False
    summarize_video: bool = False
    dense_captioning: bool = False
    scene_detection: bool = False


class VideoIntelligenceProvider(ABC):
    """Abstract base class for Video Intelligence providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        pass

    @abstractmethod
    async def ask_video(self, request: VideoIntelligenceRequest) -> VideoIntelligenceResult:
        pass

    @abstractmethod
    async def summarize_video(self, request: VideoIntelligenceRequest) -> VideoIntelligenceResult:
        pass




class ExistingSnipraVLMProvider:
    """Provider using Snipra's existing VLM infrastructure."""

    name = "snipra_vlm"
    capabilities = ProviderCapabilities(
        ask_video=True,
        summarize_video=True,
        dense_captioning=False,
        scene_detection=True,
    )

    def __init__(self):
        from app.core.config import settings

        self.settings = settings
        self._detector = None
        self._detector_initialized = False
        self._detector_configuration: tuple | None = None
        self._availability: dict | None = None
        self._vision_provider: str = ""
        self._vision_model: str = ""

    def _ensure_detector(self):
        """Resuelve el proveedor VISUAL (independiente del textual) y valida
        capabilities + credenciales reales. Nunca marca disponible un provider
        que no puede recibir frames; nunca cae a una API pagada sin que el
        usuario la haya elegido.
        """
        configuration = tuple(getattr(self.settings, key, "") for key in (
            "llm_provider", "llm_model", "vision_provider", "vision_model", "nvidia_vision_model",
        )) + tuple(bool(getattr(self.settings, key, "")) for key in (
            "anthropic_api_key", "openai_api_key", "nvidia_api_key",
        ))
        if (self._detector_initialized and configuration == self._detector_configuration
                and self.settings.vision_provider != "nvidia"):
            return
        self._detector_configuration = configuration
        self._vision_provider, self._vision_model = "", ""

        from app.providers.nvidia_catalog import NvidiaCatalogError
        from app.providers.vision import resolve_vision
        from app.video_intelligence.models import VideoIntelligenceErrorCode

        # El proveedor visual se resuelve aparte: vision_provider explícito o,
        # con "auto", el proveedor de texto SOLO si realmente soporta visión.
        text_provider = (self.settings.llm_provider or "").lower()
        try:
            self._vision_provider, self._vision_model = resolve_vision(
                text_provider, self.settings.llm_model, via=True
            )
        except NvidiaCatalogError as exc:
            self._availability = {
                "available": False,
                "code": VideoIntelligenceErrorCode.PROVIDER_UNAVAILABLE,
                "reason": str(exc),
            }
            self._detector_initialized = True
            return
        self._detector_initialized = True

        provider = self._vision_provider
        if not provider:
            self._availability = {
                "available": False,
                "code": VideoIntelligenceErrorCode.VISUAL_PROVIDER_NOT_CONFIGURED,
                "reason": (
                    "No hay proveedor visual configurado para VIA "
                    f"(texto: {text_provider!r}). Configura vision_provider en "
                    "Ajustes (claude_code/codex/nvidia) o un proveedor de texto "
                    "con vision."
                ),
            }
            return

        # POLITICA DE COSTO (ver app/providers/costs.py y vision.resolve_vision):
        # - "auto" nunca escala a una API pagada: solo sigue al proveedor de
        #   texto si ya soporta vision. Si el texto es una API pagada elegida
        #   deliberadamente por el usuario, auto acompaña esa eleccion (ya paga
        #   por el texto; no es fallback oculto).
        # - configuracion explicita (vision_provider=...) es la unica forma de
        #   usar un provider visual diferente al textual; nvidia solo explicita.
        # Por eso aqui NO hay guard extra: la resolucion ya impone la politica,
        # y abajo se separa MISSING_CREDENTIALS (fallo cerrado) de disponible.

        if provider == "claude" and not self.settings.anthropic_api_key:
            self._availability = {
                "available": False,
                "code": VideoIntelligenceErrorCode.MISSING_PROVIDER_CREDENTIALS,
                "reason": "Vision 'claude' selected but ANTHROPIC_API_KEY is not configured",
            }
        elif provider == "openai" and not self.settings.openai_api_key:
            self._availability = {
                "available": False,
                "code": VideoIntelligenceErrorCode.MISSING_PROVIDER_CREDENTIALS,
                "reason": "Vision 'openai' selected but OPENAI_API_KEY is not configured",
            }
        elif provider == "nvidia" and not self.settings.nvidia_api_key:
            self._availability = {
                "available": False,
                "code": VideoIntelligenceErrorCode.MISSING_PROVIDER_CREDENTIALS,
                "reason": "Vision 'nvidia' selected but NVIDIA_API_KEY is not configured",
            }
        elif provider in ("claude_code", "codex"):
            from app.providers.cli_subscription import resolve_provider_cli

            if resolve_provider_cli(provider) is None:
                self._availability = {
                    "available": False,
                    "code": VideoIntelligenceErrorCode.VISUAL_PROVIDER_UNSUPPORTED,
                    "reason": (
                        f"El CLI de {provider} no esta instalado o no esta en el "
                        "PATH de la maquina que corre Snipra."
                    ),
                }
            else:
                self._availability = {"available": True, "code": None, "reason": None}
        else:
            self._availability = {"available": True, "code": None, "reason": None}

    @property
    def detector(self):
        """Get the detector, initializing lazily."""
        self._ensure_detector()
        return self._detector

    def get_availability(self) -> dict:
        """Get VIA availability status."""
        self._ensure_detector()
        return self._availability.copy()

    def resolve_visual_identity(self) -> tuple[str, str]:
        """Resolve visual identity (NVIDIA discovery may refresh its local cache).

        Returns:
            tuple: (visual_provider, visual_model)
        """
        self._ensure_detector()
        return (self._vision_provider or "snipra_vlm", self._vision_model or "unknown")

    def _check_availability(self):
        """Check availability and raise if not available."""
        avail = self.get_availability()
        if not avail["available"]:
            from app.video_intelligence.models import VideoIntelligenceError
            raise VideoIntelligenceError(
                avail["code"],
                avail["reason"],
            )

    async def ask_video(
        self,
        request: VideoIntelligenceRequest,
        analysis_plan: VideoAnalysisPlan | None = None,
    ) -> VideoIntelligenceResult:
        """Execute a VideoAnalysisPlan for ask_video.

        FULL_SCAN (plan.strategy, always exactly 1 AnalysisChunk - see
        chunking.plan_analysis) takes the ORIGINAL single-call path verbatim:
        same prompt, same all-or-nothing failure behavior (raises, does not
        return a partial result), byte-for-byte what ask_video did before
        Phase 4.1 Slice 3. This is deliberate - the well-tested short-clip
        smoke path must not regress.

        TARGETED_SCAN (>1 chunk) is new: every chunk answers the SAME
        request.question over its own absolute temporal window, each chunk
        can fail independently (failed_chunks/partial, same mechanism
        summarize_video already uses), and the per-chunk VideoAnswer objects
        are combined deterministically in _aggregate_ask_video_answers - no
        second LLM call synthesizes them.
        """
        self._check_availability()
        if not self.capabilities.ask_video:
            raise VideoIntelligenceError(
                VideoIntelligenceErrorCode.CAPABILITY_UNAVAILABLE,
                f"Provider {self.name} does not support ask_video",
            )

        plan = analysis_plan or plan_analysis(request)
        requested_frames = plan.total_planned_frames

        if plan.strategy == AnalysisStrategy.FULL_SCAN:
            chunk = plan.chunks[0]
            frames = await self._extract_frames(
                request.asset_id, chunk.start_us, chunk.end_us, sampling_fps=chunk.sampling_fps
            )
            if not frames:
                raise VideoIntelligenceError(
                    VideoIntelligenceErrorCode.FRAME_EXTRACTION_FAILED,
                    f"No frames extracted for asset {request.asset_id} in range "
                    f"{request.start_us}-{request.end_us}us",
                )

            prompt = self._build_ask_prompt(request.question)
            system_prompt = self._build_system_prompt()
            raw_response = self._call_vlm_with_frames(system_prompt, prompt, frames)

            analyzed_frames = len(frames)
            return self._parse_response(
                request, raw_response, requested_frames, analyzed_frames, frames
            )

        # TARGETED_SCAN: multiple AnalysisChunk, one Director-visible result.
        # Phase 4.3A: chunks execute via chunk_executor.run_chunks_bounded
        # (bounded concurrency) instead of a sequential loop; aggregation
        # below is otherwise unchanged from Phase 4.1/3.5B.2 - it iterates
        # the ChunkOutcome[] in chunk.index order (run_chunks_bounded's own
        # ordering guarantee), never something a worker mutated directly.
        async def _ask_chunk_worker(chunk: AnalysisChunk) -> dict:
            chunk_id = f"chunk_{chunk.index}"
            frames = await self._extract_frames(
                request.asset_id,
                chunk.start_us,
                chunk.end_us,
                sampling_fps=chunk.sampling_fps,
            )
            if not frames:
                raise VideoIntelligenceError(
                    VideoIntelligenceErrorCode.FRAME_EXTRACTION_FAILED,
                    f"No frames extracted for {chunk_id}",
                )
            prompt = self._build_ask_prompt(request.question, chunk=chunk)
            system_prompt = self._build_system_prompt()
            raw_response = await asyncio.to_thread(
                self._call_vlm_with_frames, system_prompt, prompt, frames
            )
            parsed = self._parse_ask_chunk_response(request, chunk, raw_response, frames)
            return {"frames_len": len(frames), "parsed": parsed}

        outcomes = await run_chunks_bounded(plan.chunks, _ask_chunk_worker)

        chunk_answers: list[dict] = []
        all_observations: list[dict] = []
        all_events: list[dict] = []
        failed_chunks: list[dict] = []
        total_frames = 0

        for outcome in outcomes:
            chunk_id = f"chunk_{outcome.index}"
            if not outcome.ok:
                failed_chunks.append({"chunk_index": outcome.index, "error": outcome.error})
                continue

            # Only counted once extraction + VLM call + parse all succeed -
            # same rule summarize_video already applies to analyzed_frames.
            total_frames += outcome.value["frames_len"]
            parsed = outcome.value["parsed"]
            chunk_answers.append(parsed["answer"])
            for obs in parsed["observations"]:
                obs["chunk_id"] = chunk_id
                all_observations.append(obs)
            for ev in parsed["events"]:
                ev["chunk_id"] = chunk_id
                all_events.append(ev)

        coverage_ratio, coverage_status = VideoProcessingMetadata.compute_coverage(
            requested_frames, total_frames
        )
        metadata = VideoProcessingMetadata(
            chunks_processed=len(plan.chunks) - len(failed_chunks),
            chunks_failed=len(failed_chunks),
            total_frames_analyzed=total_frames,
            provider=self._vision_provider or "snipra_vlm",
            model=self._detector_model(),
            processing_time_ms=0,
            partial=len(failed_chunks) > 0,
            failed_chunks=failed_chunks,
            requested_frames=requested_frames,
            analyzed_frames=total_frames,
            coverage_ratio=coverage_ratio,
            coverage_status=coverage_status,
        )

        # No chunk produced a visual answer - do not fabricate one. answer stays
        # None (VideoIntelligenceResult.answer is Optional), same "no synthetic
        # response" rule summarize_video already applies when every chunk fails.
        answer = self._aggregate_ask_video_answers(chunk_answers) if chunk_answers else None

        return VideoIntelligenceResult(
            asset_id=request.asset_id,
            start_us=request.start_us,
            end_us=request.end_us,
            mode=request.mode,
            question=request.question,
            objective=request.objective,
            summary=None,
            answer=answer,
            observations=all_observations,
            events=all_events,
            scenes=[],
            objects=[],
            metadata=metadata,
            partial=len(failed_chunks) > 0,
            contract_version="via_v2",
        )

    async def summarize_video(
        self,
        request: VideoIntelligenceRequest,
        analysis_plan: VideoAnalysisPlan | None = None,
    ) -> VideoIntelligenceResult:
        """Execute a VideoAnalysisPlan for summarize_video.

        Chunk count, chunk duration and sampling density are NOT decided
        here - they come from analysis_plan (Phase 4.1's AnalysisPlanner),
        the sole technical-sampling authority. When called directly without
        a plan (e.g. from tests, or any caller that isn't
        VideoIntelligenceService), one is computed via plan_analysis(request)
        - the same neutral function, not a second competing decision path.
        In production, VideoIntelligenceService always supplies the plan
        itself so PLAN_INFEASIBLE fails before any frame extraction happens.
        """
        self._check_availability()
        from app.video_intelligence.models import (
            VideoIntelligenceResult,
            VideoProcessingMetadata,
        )

        if not self.capabilities.summarize_video:
            raise VideoIntelligenceError(
                VideoIntelligenceErrorCode.CAPABILITY_UNAVAILABLE,
                f"Provider {self.name} does not support summarize_video",
            )

        plan = analysis_plan or plan_analysis(request)

        # Requested frames now come exclusively from the plan's own budget,
        # not a naive duration*1fps estimate - see plan_analysis() for how
        # that budget is derived (Phase 4.1).
        requested_frames = plan.total_planned_frames

        # Process each chunk. Phase 4.3A: chunks execute via
        # chunk_executor.run_chunks_bounded (bounded concurrency) instead of
        # a sequential loop; aggregation below is otherwise unchanged - it
        # iterates the ChunkOutcome[] in chunk.index order (run_chunks_
        # bounded's own ordering guarantee), never something a worker
        # mutated directly.
        async def _summarize_chunk_worker(chunk: AnalysisChunk) -> dict:
            chunk_id = f"chunk_{chunk.index}"
            frames = await self._extract_frames(
                request.asset_id,
                chunk.start_us,
                chunk.end_us,
                sampling_fps=chunk.sampling_fps,
            )

            # _analyze_chunk/_parse_chunk_response take the legacy plain-dict
            # chunk shape - translate the plan's AnalysisChunk rather than
            # changing that internal contract for this slice.
            chunk_dict = {
                "index": chunk.index,
                "start_us": chunk.start_us,
                "end_us": chunk.end_us,
            }
            result = await self._analyze_chunk(request, chunk_dict, frames, chunk_id)
            return {"frames_len": len(frames), "result": result, "frames": frames}

        outcomes = await run_chunks_bounded(plan.chunks, _summarize_chunk_worker)

        all_observations = []
        all_events = []
        all_scenes = []
        all_objects = []
        failed_chunks = []
        total_frames = 0

        for outcome in outcomes:
            chunk_id = f"chunk_{outcome.index}"
            if not outcome.ok:
                failed_chunks.append({"chunk_index": outcome.index, "error": outcome.error})
                continue

            # Only counted once extraction + VLM call + parse all succeeded -
            # a failed chunk's frames never inflate analyzed_frames/coverage_ratio.
            total_frames += outcome.value["frames_len"]
            result = outcome.value["result"]

            if result.get("observations"):
                for obs in result["observations"]:
                    obs["chunk_id"] = chunk_id
                    all_observations.append(obs)
            if result.get("events"):
                for ev in result["events"]:
                    ev["chunk_id"] = chunk_id
                    all_events.append(ev)
            if result.get("scenes"):
                for sc in result["scenes"]:
                    sc["chunk_id"] = chunk_id
                    all_scenes.append(sc)
            if result.get("objects"):
                for obj in result["objects"]:
                    obj["chunk_id"] = chunk_id
                    all_objects.append(obj)

        # Aggregate results
        deduplicated_events = self._deduplicate_events(all_events)
        merged_scenes = self._merge_scenes(all_scenes)
        merged_objects = self._merge_objects(all_objects)

        # Build summary
        summary = self._build_summary(
            request, all_observations, deduplicated_events, merged_scenes
        )
        # This is SUMMARIZE_VIDEO, so there is no direct answer.

        # Compute coverage metadata
        coverage_ratio, coverage_status = VideoProcessingMetadata.compute_coverage(
            requested_frames, total_frames
        )

        metadata = VideoProcessingMetadata(
            chunks_processed=len(plan.chunks) - len(failed_chunks),
            chunks_failed=len(failed_chunks),
            total_frames_analyzed=total_frames,
            provider=self._vision_provider or "snipra_vlm",
            model=self._detector_model(),
            processing_time_ms=0,
            partial=len(failed_chunks) > 0,
            failed_chunks=failed_chunks,
            # Coverage metadata (Phase 3.4)
            requested_frames=requested_frames,
            analyzed_frames=total_frames,
            coverage_ratio=coverage_ratio,
            coverage_status=coverage_status,
        )

        return VideoIntelligenceResult(
            asset_id=request.asset_id,
            start_us=request.start_us,
            end_us=request.end_us,
            mode=request.mode,
            question=request.question,
            objective=request.objective,
            summary=summary,
            answer=None,
            observations=all_observations,
            events=deduplicated_events,
            scenes=merged_scenes,
            objects=merged_objects,
            metadata=metadata,
            partial=len(failed_chunks) > 0,
            contract_version="via_v2",
        )

    async def _extract_frames(
        self,
        asset_id: str,
        start_us: int,
        end_us: int,
        sampling_fps: float = 1.0,
    ) -> list[ExtractedFrame]:
        """Extract frames from video segment using FFmpeg.

        Returns ExtractedFrame objects with frame_id, timestamp_us, and data.
        Frame IDs are deterministic: frame_0000, frame_0001, etc. within each chunk.
        Timestamps are computed from chunk start + frame index / sampling_fps.
        """
        return await asyncio.to_thread(
            self._extract_frames_blocking, asset_id, start_us, end_us, sampling_fps
        )

    def _extract_frames_blocking(
        self,
        asset_id: str,
        start_us: int,
        end_us: int,
        sampling_fps: float = 1.0,
    ) -> list[ExtractedFrame]:
        """Synchronous frame extraction body - the actual blocking boundary.
        Never call this directly with `await` or wrap it in asyncio.to_thread
        from anywhere except _extract_frames() itself.

        Uses app.core.media.run() (Phase 4.3A) instead of a raw
        subprocess.run() call so a real cancellation (see
        chunk_executor.run_chunks_bounded's cancel_check wiring) can
        actually terminate the ffmpeg child process instead of abandoning
        it. Frame extraction range/sampling_fps/JPEG quality/output
        semantics are otherwise unchanged from the pre-4.3A implementation.

        Returns ExtractedFrame objects with deterministic frame_id and timestamp_us.
        """
        import os
        import tempfile

        from sqlmodel import Session

        from app.core.db import engine
        from app.editor.models import EditorAsset

        # Get asset path
        with Session(engine) as session:
            asset = session.get(EditorAsset, asset_id)
            if not asset:
                raise ValueError(f"Asset {asset_id} not found")
            video_path = asset.original_path

        # Calculate timestamps
        start_s = start_us / 1_000_000
        duration_s = (end_us - start_us) / 1_000_000

        # Extract frames using FFmpeg
        with tempfile.TemporaryDirectory() as tmpdir:
            output_pattern = os.path.join(tmpdir, "frame_%04d.jpg")
            cmd = [
                self.settings.ffmpeg_path, "-y",
                "-ss", str(start_s),
                "-t", str(duration_s),
                "-i", video_path,
                "-vf", f"fps={sampling_fps}",
                "-q:v", "2",
                output_pattern
            ]
            media.run(cmd)

            # Collect frame paths, then read bytes before the tempdir is removed
            frame_paths = sorted([
                os.path.join(tmpdir, f)
                for f in os.listdir(tmpdir)
                if f.endswith(".jpg")
            ])
            frames: list[ExtractedFrame] = []
            for idx, path in enumerate(frame_paths):
                with open(path, "rb") as handle:
                    data = handle.read()
                # Deterministic frame ID within chunk
                frame_id = f"frame_{idx:04d}"
                # Timestamp from sampling (chunk start + frame_index / fps)
                if sampling_fps > 0:
                    timestamp_us = start_us + int((idx / sampling_fps) * 1_000_000)
                else:
                    timestamp_us = start_us
                frames.append(
                    ExtractedFrame(frame_id=frame_id, timestamp_us=timestamp_us, data=data)
                )
            return frames

    def _build_system_prompt(self) -> str:
        return (
            "You are a video analysis AI. Analyze the video frames and answer "
            "the question accurately.\n"
            "Return a JSON object with:\n"
            "- answer: direct answer to the question\n"
            "- confidence: 0.0 to 1.0 (optional, only if genuinely estimable)\n"
            "- observations: list of observable facts with timestamps, nature, "
            "confidence, and frame_ids\n"
            "- events: list of events with timestamps, nature, confidence, "
            "frame_ids, and descriptions\n"
            "- confidence: overall confidence 0.0-1.0 (optional)\n"
            "\n"
            "Each observation/event must include:\n"
            '- start: offset in seconds (e.g., "1.5s")\n'
            '- end: offset in seconds (e.g., "2.0s")\n'
            "- description: what is observed\n"
            '- nature: "observation" | "inference" (required)\n'
            "- confidence: 0.0-1.0 (optional, only if genuinely estimable)\n"
            "- frame_ids: list of frame identifiers this evidence refers to "
            '(e.g., ["frame_0001", "frame_0002"])'
        )

    def _build_ask_prompt(self, question: str, chunk: AnalysisChunk | None = None) -> str:
        range_note = ""
        if chunk is not None:
            # Only used for TARGETED_SCAN's multiple AnalysisChunk (chunk is
            # None for FULL_SCAN's single call, preserving its exact original
            # prompt). Telling the model the chunk's real absolute window keeps
            # its own answer text coherent with the [start-end] label
            # _aggregate_ask_video_answers prepends when combining chunks.
            start_s = chunk.start_us / 1_000_000
            end_s = chunk.end_us / 1_000_000
            range_note = (
                f"\nThis clip covers seconds {start_s:.1f}-{end_s:.1f} of a "
                f"longer recording; answer using only what is visible in these frames."
            )
        return (
            f"Analyze the video frames and answer this question: {question}\n"
            f"{range_note}\n"
            "Return JSON:\n"
            "{\n"
            '  "answer": "direct answer",\n'
            '  "confidence": 0.0-1.0,\n'
            '  "observations": [\n'
            '    {"start": "0.0s", "end": "0.0s", "description": "observable fact", '
            '"nature": "observation", "confidence": 0.0-1.0, '
            '"frame_ids": ["frame_0000"]}\n'
            "  ],\n"
            '  "events": [\n'
            '    {"start": "0.0s", "end": "5.0s", "type": "action", '
            '"description": "event description",\n'
            '      "nature": "observation", "confidence": 0.0-1.0, '
            '"frame_ids": ["frame_0000", "frame_0001"]}\n'
            "  ]\n"
            "}\n"
            "\n"
            'Offsets ("start"/"end") are seconds relative to the first frame '
            "shown (0.0s).\n"
            "Base the answer and every observation/event only on what is visible "
            "in the frames.\n"
            "Include frame_ids for each observation/event to ground them in "
            "specific frames.\n"
            "Confidence and nature are optional but encouraged when the model "
            "can genuinely estimate them.\n"
            "If confidence cannot be genuinely estimated, omit the field.\n"
            "If nature is unclear, omit the field."
        )

    def _build_summarize_system_prompt(self) -> str:
        return """You are a video analysis AI. Analyze the given video frames, sampled in \
chronological order from one segment of a longer video, and describe only what is \
visually observable. Do not infer beyond what is shown, and do not describe frames \
you were not given. Return a single JSON object matching the requested schema exactly.

Each observation/event/scene/object must include:
- nature: "observation" | "inference" (required)
- confidence: 0.0-1.0 (optional, only if genuinely estimable)
- frame_ids: list of frame identifiers (e.g., ["frame_0001", "frame_0002"])"""

    def _build_chunk_prompt(self, request: VideoIntelligenceRequest, chunk: dict) -> str:
        objective = request.objective or "Describe what happens in this video segment"
        return (
            f"{objective}\n"
            "\n"
            "Return JSON:\n"
            "{\n"
            '  "observations": [\n'
            '    {"start": "0.0s", "end": "0.0s", "description": "observable fact", '
            '"nature": "observation", "confidence": 0.0-1.0, '
            '"frame_ids": ["frame_0000"]}\n'
            "  ],\n"
            '  "events": [\n'
            '    {"start": "0.0s", "end": "0.0s", "type": "action", '
            '"description": "event description",\n'
            '      "nature": "observation", "confidence": 0.0-1.0, '
            '"frame_ids": ["frame_0000", "frame_0001"],\n'
            '      "observed_action": "short action label", "subjects": ["..."], '
            '"objects": ["..."]}\n'
            "  ],\n"
            '  "scenes": [\n'
            '    {"description": "scene description", "nature": "observation", '
            '"confidence": 0.0-1.0, "frame_ids": ["frame_0000"],\n'
            '      "scene_type": "e.g. gameplay, menu, cutscene, outdoor, indoor", '
            '"subjects": ["..."], "dominant_objects": ["..."]}\n'
            "  ],\n"
            '  "objects": [\n'
            '    {"label": "object name", "nature": "observation", '
            '"confidence": 0.0-1.0, "frame_ids": ["frame_0000"]}\n'
            "  ]\n"
            "}\n"
            "\n"
            'Offsets ("start"/"end") are seconds relative to the start of THIS '
            "segment (0.0s = first frame).\n"
            "Only include items you can actually see. Return an empty list for "
            "any category with nothing to report.\n"
            "Include frame_ids for each item to ground them in specific frames.\n"
            "Confidence and nature are optional but encouraged when the model "
            "can genuinely estimate them.\n"
            "If confidence cannot be genuinely estimated, omit the field.\n"
            "If nature is unclear, omit the field."
        )

    def _detector_model(self) -> str:
        """Modelo visual efectivo que recibe los frames (resuelto por
        _ensure_detector via resolve_vision)."""
        self._ensure_detector()
        return self._vision_model or "claude-haiku-4-5"

    @staticmethod
    def _frames_to_image_content(frames: list[ExtractedFrame]) -> list[dict]:
        """Encode real extracted JPEG frame bytes as Anthropic multimodal image blocks.

        Mirrors the transport mechanism proven in app/providers/vision.py::_score_claude
        (base64 image blocks + a trailing text block), reimplemented locally so VIA does
        not couple to that module's clip-scoring semantics/prompting.
        Also adds frame_id label before each image for grounding.
        Includes a frame manifest at the beginning for grounding.
        """
        import base64
        content = []
        # Add frame manifest at the beginning
        manifest_lines = []
        for frame in frames:
            ts_s = frame.timestamp_us / 1_000_000
            manifest_lines.append(
                f"FRAME {frame.frame_id} TIMESTAMP {ts_s:.3f}s FILE {frame.frame_id}.jpg"
            )
        content.append({
            "type": "text",
            "text": "FRAME MANIFEST:\n" + "\n".join(manifest_lines)
        })
        for frame in frames:
            content.append({
                "type": "text",
                "text": f"FRAME {frame.frame_id} TIMESTAMP {frame.timestamp_us / 1_000_000:.3f}s"
            })
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(frame.data).decode(),
                },
            })
        return content

    def _call_vlm_with_frames(self, system: str, prompt: str, frames: list[ExtractedFrame]) -> str:
        """Envia frames + prompt al proveedor VISUAL resuelto (independiente del
        proveedor textual). Fail-closed: sin capacidad visual real o sin
        credenciales, nunca se degrada a texto plano ni a otra API.
        """
        self._ensure_detector()
        provider = self._vision_provider
        if not provider:
            est = self._availability or {}
            raise VideoIntelligenceError(
                (est.get("code") or VideoIntelligenceErrorCode.VISUAL_PROVIDER_NOT_CONFIGURED),
                est.get("reason") or "No hay proveedor visual configurado",
            )

        try:
            if provider == "claude":
                return self._vlm_claude(system, prompt, frames)
            if provider == "openai":
                return self._vlm_openai_compatible("openai", system, prompt, frames)
            if provider == "nvidia":
                return self._vlm_openai_compatible("nvidia", system, prompt, frames)
            if provider == "claude_code":
                return self._vlm_claude_code(system, prompt, frames)
            if provider == "codex":
                return self._vlm_codex(system, prompt, frames)
        except VideoIntelligenceError:
            raise
        except Exception as exc:
            raise self._map_vlm_error(provider, exc) from exc
        raise VideoIntelligenceError(
            VideoIntelligenceErrorCode.VISUAL_PROVIDER_UNSUPPORTED,
            f"Vision provider {provider!r} no soportado por VIA",
        )

    @staticmethod
    def _map_vlm_error(provider: str, exc: Exception) -> VideoIntelligenceError:
        """Mapea errores del SDK/CLI a codigos deterministas. Nunca incluye
        credenciales: solo tipo + extracto acotado y sanitizado del mensaje."""
        from app.core.sanitize import sanitize_error_text

        module = type(exc).__module__
        name = type(exc).__name__
        snippet = sanitize_error_text(str(exc).strip()[:200])
        msg = f"fallo en proveedor visual {provider}: {name}: {snippet}"
        if name in {"APITimeoutError", "TimeoutError"} or "timeout" in name.lower():
            return VideoIntelligenceError(VideoIntelligenceErrorCode.TIMEOUT, msg)
        status = getattr(exc, "status_code", None)
        if status in (401, 403):
            return VideoIntelligenceError(
                VideoIntelligenceErrorCode.MISSING_PROVIDER_CREDENTIALS, msg
            )
        if status in (404, 429) or (status is not None and status >= 500):
            return VideoIntelligenceError(VideoIntelligenceErrorCode.PROVIDER_UNAVAILABLE, msg)
        if name == "_CliError" or module.endswith("cli_subscription"):
            return VideoIntelligenceError(VideoIntelligenceErrorCode.PROVIDER_UNAVAILABLE, msg)
        return VideoIntelligenceError(VideoIntelligenceErrorCode.INVALID_OUTPUT, msg)

    def _vlm_claude(self, system: str, prompt: str, frames: list[ExtractedFrame]) -> str:
        if not self.settings.anthropic_api_key:
            raise VideoIntelligenceError(
                VideoIntelligenceErrorCode.MISSING_PROVIDER_CREDENTIALS,
                "Vision 'claude' selected but ANTHROPIC_API_KEY is not configured",
            )
        from anthropic import Anthropic

        client = Anthropic(api_key=self.settings.anthropic_api_key)
        content = self._frames_to_image_content(frames) + [{"type": "text", "text": prompt}]
        resp = client.messages.create(
            model=self._detector_model(),
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    def _vlm_openai_compatible(
        self, provider: str, system: str, prompt: str, frames: list[ExtractedFrame]
    ) -> str:
        import base64

        if provider == "openai":
            key = self.settings.openai_api_key
            base_url = None
        else:  # nvidia
            key = self.settings.nvidia_api_key
            base_url = self.settings.nvidia_base_url
        if not key:
            raise VideoIntelligenceError(
                VideoIntelligenceErrorCode.MISSING_PROVIDER_CREDENTIALS,
                f"Vision '{provider}' selected but its API key is not configured",
            )
        from openai import OpenAI

        kwargs = {"api_key": key, "timeout": 120.0, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        content: list[dict] = [{"type": "text", "text": f"{system}\n\n{prompt}"}]
        for frame in frames:
            b64 = base64.b64encode(frame.data).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
            # Add frame label for grounding
            content.append({
                "type": "text",
                "text": f"[FRAME {frame.frame_id} @ {frame.timestamp_us / 1_000_000:.3f}s]"
            })
        resp = client.chat.completions.create(
            model=self._detector_model(),
            messages=[{"role": "user", "content": content}],
        )
        return resp.choices[0].message.content or ""

    def _vlm_claude_code(self, system: str, prompt: str, frames: list[ExtractedFrame]) -> str:
        """Claude Code CLI (suscripcion): frames como @archivo en cwd temporal.

        Includes frame manifest for grounding.
        """
        import subprocess
        import tempfile
        from pathlib import Path

        from app.providers.cli_subscription import (
            _CLI_TIMEOUT_SECONDS,
            _extract_claude_result,
            resolve_provider_cli,
        )

        exe = resolve_provider_cli("claude_code")
        if exe is None:
            raise VideoIntelligenceError(
                VideoIntelligenceErrorCode.VISUAL_PROVIDER_UNSUPPORTED,
                "El CLI de Claude Code no esta instalado o no esta en el PATH.",
            )
        with tempfile.TemporaryDirectory() as tmp:
            refs: list[str] = []
            manifest_lines: list[str] = []
            for frame in frames:
                # Use deterministic frame_id as filename
                filename = f"{frame.frame_id}.jpg"
                path = Path(tmp) / filename
                path.write_bytes(frame.data)
                refs.append(f"@{filename}")
                # Add to manifest for grounding
                ts_s = frame.timestamp_us / 1_000_000
                manifest_lines.append(
                    f"FRAME {frame.frame_id} TIMESTAMP {ts_s:.3f}s FILE {filename}"
                )

            # Include manifest in prompt for frame grounding
            manifest = "\n".join(manifest_lines)
            full_prompt = (
                f"{system}\n\n{prompt}\n\nFRAME MANIFEST:\n{manifest}\n\n"
                "Fotogramas: " + " ".join(refs)
            )
            cmd = [exe, "-p", "--model", self._detector_model(), "--output-format", "json"]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    input=full_prompt,
                    timeout=_CLI_TIMEOUT_SECONDS,
                    cwd=tmp,
                )
            except subprocess.TimeoutExpired as exc:
                raise VideoIntelligenceError(
                    VideoIntelligenceErrorCode.TIMEOUT, f"Claude Code CLI timeout: {exc}"
                ) from exc
            except OSError as exc:
                raise VideoIntelligenceError(
                    VideoIntelligenceErrorCode.PROVIDER_UNAVAILABLE, f"Claude Code CLI: {exc}"
                ) from exc
            if proc.returncode != 0:
                from app.core.sanitize import sanitize_error_text

                raise VideoIntelligenceError(
                    VideoIntelligenceErrorCode.PROVIDER_UNAVAILABLE,
                    f"Claude Code CLI fallo ({proc.returncode}): "
                    f"{sanitize_error_text((proc.stderr or '')[:200])}",
                )
            return _extract_claude_result(proc.stdout)

    def _vlm_codex(self, system: str, prompt: str, frames: list[ExtractedFrame]) -> str:
        """Codex CLI (suscripcion): prompt primero, luego -i <frames ...>.

        Includes frame manifest for grounding.
        """
        import subprocess
        import tempfile
        from pathlib import Path

        from app.providers.cli_subscription import resolve_provider_cli

        exe = resolve_provider_cli("codex")
        if exe is None:
            raise VideoIntelligenceError(
                VideoIntelligenceErrorCode.VISUAL_PROVIDER_UNSUPPORTED,
                "El CLI de Codex no esta instalado o no esta en el PATH.",
            )
        with tempfile.TemporaryDirectory() as tmp:
            paths: list[str] = []
            manifest_lines: list[str] = []
            for frame in frames:
                # Use deterministic frame_id as filename
                filename = f"{frame.frame_id}.jpg"
                path = Path(tmp) / filename
                path.write_bytes(frame.data)
                paths.append(str(path.resolve()))
                # Add to manifest for grounding
                ts_s = frame.timestamp_us / 1_000_000
                manifest_lines.append(
                    f"FRAME {frame.frame_id} TIMESTAMP {ts_s:.3f}s FILE {filename}"
                )
            
            # Include manifest in prompt for frame grounding
            manifest = "\n".join(manifest_lines)
            out_path = Path(tmp) / "out.txt"
            cmd = [
                exe,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-rules",
                "--model",
                self._detector_model(),
                "-o",
                str(out_path),
                f"{system}\n\n{prompt}\n\nFRAME MANIFEST:\n{manifest}",
                "-i",
                *paths,
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=300,
                    cwd=tmp,
                    stdin=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired as exc:
                raise VideoIntelligenceError(
                    VideoIntelligenceErrorCode.TIMEOUT, f"Codex CLI timeout: {exc}"
                ) from exc
            except OSError as exc:
                raise VideoIntelligenceError(
                    VideoIntelligenceErrorCode.PROVIDER_UNAVAILABLE, f"Codex CLI: {exc}"
                ) from exc
            if proc.returncode != 0:
                from app.core.sanitize import sanitize_error_text

                raise VideoIntelligenceError(
                    VideoIntelligenceErrorCode.PROVIDER_UNAVAILABLE,
                    f"Codex CLI fallo ({proc.returncode}): "
                    f"{sanitize_error_text((proc.stderr or '')[:200])}",
                )
            if not out_path.exists():
                raise VideoIntelligenceError(
                    VideoIntelligenceErrorCode.PROVIDER_UNAVAILABLE,
                    "Codex CLI no escribio ninguna respuesta visual.",
                )
            return out_path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _parse_optional_offset_us(value, floor_us: int, ceil_us: int) -> int | None:
        """Parse a '12.5s'-style offset relative to floor_us into an absolute,
        range-clamped microsecond timestamp. Returns None when the value
        is missing or unparseable - caller must handle missing temporal grounding."""
        if value is None:
            return None
        try:
            seconds = float(str(value).rstrip("s"))
        except (TypeError, ValueError):
            return None
        # NaN / Infinity are not valid temporal grounding (VIA v2 section 7).
        if not math.isfinite(seconds):
            return None
        absolute_us = floor_us + int(seconds * 1_000_000)
        return max(floor_us, min(ceil_us, absolute_us))

    @staticmethod
    def _has_explicit_offset(item: dict) -> bool:
        """Check if item has an explicit, parseable, finite start/end offset."""
        for key in ("start", "end"):
            val = item.get(key)
            if val is not None:
                # Validate it is actually parseable AND finite (NaN/Inf do not
                # count as explicit model grounding - VIA v2 section 7).
                try:
                    parsed = float(str(val).rstrip("s"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(parsed):
                    return True
        return False

    @staticmethod
    def _resolve_temporal_range(
        item: dict,
        valid_frame_ids: list[str],
        frame_id_to_ts: dict[str, int],
        floor_us: int,
        ceil_us: int,
    ) -> tuple[int, int, TemporalSource]:
        """Resolve (start_us, end_us, temporal_source) for one VIA v2 evidence item.

        Single source of truth shared by every VIA v2 parser
        (_parse_response / _parse_ask_chunk_response / _parse_chunk_response) so
        temporal grounding can never diverge between them. Rules (VIA v2
        sections 7-8 - never fabricate false precision):

        - valid frame ids           -> FRAME_MAPPED, span = [min, max] of their
                                       real timestamps
        - no frames, BOTH start and
          end parse to finite offsets -> MODEL_GROUNDED, span = parsed offsets
        - only one of start/end, or
          missing / malformed /
          NaN / Infinity             -> CHUNK_WINDOW, span = full analysis window
                                       (floor_us, ceil_us)
        """
        has_explicit = ExistingSnipraVLMProvider._has_explicit_offset(item)
        temporal_source = ExistingSnipraVLMProvider._determine_temporal_source(
            valid_frame_ids, frame_id_to_ts, has_explicit
        )

        if temporal_source == TemporalSource.FRAME_MAPPED:
            frame_ts = [
                frame_id_to_ts[fid] for fid in valid_frame_ids if fid in frame_id_to_ts
            ]
            if frame_ts:
                return min(frame_ts), max(frame_ts), TemporalSource.FRAME_MAPPED
            return floor_us, ceil_us, TemporalSource.CHUNK_WINDOW

        if temporal_source == TemporalSource.MODEL_GROUNDED:
            start_raw = item.get("start")
            end_raw = item.get("end")
            start_us = ExistingSnipraVLMProvider._parse_optional_offset_us(
                start_raw, floor_us, ceil_us
            )
            end_us = ExistingSnipraVLMProvider._parse_optional_offset_us(
                end_raw, floor_us, ceil_us
            )
            # MODEL_GROUNDED requires BOTH bounds to be present and finite.
            if start_raw is not None and end_raw is not None \
                    and start_us is not None and end_us is not None:
                if start_us > end_us:
                    start_us, end_us = end_us, start_us
                return start_us, end_us, TemporalSource.MODEL_GROUNDED
            return floor_us, ceil_us, TemporalSource.CHUNK_WINDOW

        # CHUNK_WINDOW / UNKNOWN -> full analysis window, never a synthetic point.
        return floor_us, ceil_us, TemporalSource.CHUNK_WINDOW

    @staticmethod
    def _parse_nature(nature_val: str | None) -> tuple[EvidenceNature, NatureSource]:
        if nature_val is None:
            return EvidenceNature.UNKNOWN, NatureSource.UNKNOWN
        try:
            return EvidenceNature(nature_val), NatureSource.MODEL_EXPLICIT
        except ValueError:
            return EvidenceNature.UNKNOWN, NatureSource.UNKNOWN

    @staticmethod
    def _parse_confidence(conf_val) -> tuple[float | None, ConfidenceSource]:
        if conf_val is None:
            return None, ConfidenceSource.UNKNOWN
        try:
            return float(conf_val), ConfidenceSource.MODEL_EXPLICIT
        except (TypeError, ValueError):
            return None, ConfidenceSource.UNKNOWN

    @staticmethod
    def _validate_frame_ids(frame_ids: list[str], frame_id_to_ts: dict[str, int]) -> list[str]:
        """Validate and filter frame_ids, keeping only known IDs in original order."""
        seen = set()
        valid = []
        for fid in frame_ids:
            if fid in frame_id_to_ts and fid not in seen:
                seen.add(fid)
                valid.append(fid)
        return valid

    @staticmethod
    def _determine_temporal_source(
        frame_ids: list[str], 
        frame_id_to_ts: dict[str, int], 
        has_explicit_offset: bool = False
    ) -> TemporalSource:
        """Centralized temporal source determination.
        
        Priority:
        1. Valid frame_ids that exist in frame_id_to_ts -> FRAME_MAPPED
        2. Explicit model offset/timestamps (has_explicit_offset=True) -> MODEL_GROUNDED
        3. Only analysis window known -> CHUNK_WINDOW
        4. No usable temporal context -> UNKNOWN
        
        Args:
            frame_ids: List of frame IDs referenced by the model
            frame_id_to_ts: Mapping of frame_id to timestamp_us
            has_explicit_offset: Whether the model provided explicit start/end offsets
            
        Returns:
            TemporalSource enum value
        """
        if not frame_ids:
            # No frame IDs - check for explicit offset
            if has_explicit_offset:
                return TemporalSource.MODEL_GROUNDED
            return TemporalSource.CHUNK_WINDOW
        
        # Check if all frame_ids are known
        known = all(fid in frame_id_to_ts for fid in frame_ids)
        if known:
            return TemporalSource.FRAME_MAPPED
        
        # Some frame_ids unknown - if explicit offset provided, use MODEL_GROUNDED
        if has_explicit_offset:
            return TemporalSource.MODEL_GROUNDED
        
        # Partial frame knowledge without explicit offset
        return TemporalSource.CHUNK_WINDOW

    @staticmethod
    def _build_provenance(
        frame_ids: list[str], 
        temporal_source: TemporalSource,
        confidence_source: ConfidenceSource, 
        nature_source: NatureSource,
        actual_provider: str,
        model_name: str,
        floor_us: int,
        ceil_us: int,
        frame_id_to_ts: dict[str, int],
    ) -> EvidenceProvenance:
        frame_ts = [frame_id_to_ts[fid] for fid in frame_ids if fid in frame_id_to_ts]
        return EvidenceProvenance(
            provider=actual_provider,
            model=model_name,
            frame_ids=frame_ids,
            frame_timestamps_us=frame_ts,
            analysis_window_start_us=floor_us,
            analysis_window_end_us=ceil_us,
            temporal_source=temporal_source,
            confidence_source=confidence_source,
            nature_source=nature_source,
            contract_version="via_v2",
        )

    def _parse_response(
        self,
        request,
        raw_response: str,
        requested_frames: int = 0,
        analyzed_frames: int = 0,
        extracted_frames: list[ExtractedFrame] | None = None,
    ) -> VideoIntelligenceResult:
        from app.video_intelligence.models import (
            ConfidenceSource,
            VideoAnswer,
            VideoProcessingMetadata,
        )

        try:
            # Extract JSON from response
            json_str = self._extract_json(raw_response)
            data = json.loads(json_str)

            # Parse answer with optional confidence
            answer_conf = data.get("confidence")
            answer_conf_source = (
                ConfidenceSource.MODEL_EXPLICIT
                if answer_conf is not None
                else ConfidenceSource.UNKNOWN
            )
            answer = VideoAnswer(
                answer=data.get("answer", ""),
                confidence=answer_conf,
                confidence_source=answer_conf_source,
                supporting_observation_ids=data.get("supporting_observation_ids", []),
            )

            model_name = self._detector_model()
            floor_us, ceil_us = request.start_us, request.end_us
            actual_provider = self._vision_provider or "snipra_vlm"

            # Build frame mapping for provenance
            frame_id_to_ts = {f.frame_id: f.timestamp_us for f in (extracted_frames or [])}

            observations = []
            for obs in data.get("observations", []):
                nature, nature_source = self._parse_nature(obs.get("nature"))
                confidence, confidence_source = self._parse_confidence(obs.get("confidence"))
                frame_ids = obs.get("frame_ids", [])
                valid_frame_ids = self._validate_frame_ids(frame_ids, frame_id_to_ts)
                start_us, end_us, temporal_source = self._resolve_temporal_range(
                    obs, valid_frame_ids, frame_id_to_ts, floor_us, ceil_us
                )
                prov = self._build_provenance(
                    valid_frame_ids, temporal_source, confidence_source, nature_source,
                    actual_provider, model_name, floor_us, ceil_us, frame_id_to_ts,
                )
                observations.append(VideoObservation(
                    asset_id=request.asset_id,
                    start_us=start_us,
                    end_us=end_us,
                    description=obs.get("description", ""),
                    nature=nature,
                    confidence=confidence,
                    confidence_source=confidence_source,
                    nature_source=nature_source,
                    source=actual_provider,
                    provider=actual_provider,
                    model=model_name,
                    provenance=prov,
                ))

            events = []
            for ev in data.get("events", []):
                nature, nature_source = self._parse_nature(ev.get("nature"))
                confidence, confidence_source = self._parse_confidence(ev.get("confidence"))
                frame_ids = ev.get("frame_ids", [])
                valid_frame_ids = self._validate_frame_ids(frame_ids, frame_id_to_ts)
                start_us, end_us, temporal_source = self._resolve_temporal_range(
                    ev, valid_frame_ids, frame_id_to_ts, floor_us, ceil_us
                )
                prov = self._build_provenance(
                    valid_frame_ids, temporal_source, confidence_source, nature_source,
                    actual_provider, model_name, floor_us, ceil_us, frame_id_to_ts,
                )
                events.append(VideoEventCandidate(
                    asset_id=request.asset_id,
                    start_us=start_us,
                    end_us=end_us,
                    type=ev.get("type", "event"),
                    description=ev.get("description", ""),
                    confidence=confidence,
                    confidence_source=confidence_source,
                    nature=nature,
                    nature_source=nature_source,
                    provider=actual_provider,
                    model=model_name,
                    observed_action=ev.get("observed_action"),
                    subjects=ev.get("subjects", []),
                    objects=ev.get("objects", []),
                    provenance=prov,
                ))

            # Compute coverage metadata
            coverage_ratio, coverage_status = VideoProcessingMetadata.compute_coverage(
                requested_frames, analyzed_frames
            )

            metadata = VideoProcessingMetadata(
                chunks_processed=1,
                chunks_failed=0,
                total_frames_analyzed=analyzed_frames,
                provider=actual_provider,
                model=model_name,
                processing_time_ms=0,
                partial=False,
                failed_chunks=[],
                # Coverage metadata (Phase 3.4)
                requested_frames=requested_frames if requested_frames > 0 else None,
                analyzed_frames=analyzed_frames if analyzed_frames > 0 else None,
                coverage_ratio=coverage_ratio,
                coverage_status=coverage_status,
            )

            return VideoIntelligenceResult(
                asset_id=request.asset_id,
                start_us=request.start_us,
                end_us=request.end_us,
                mode=request.mode,
                question=request.question,
                objective=request.objective,
                summary=None,
                answer=answer,
                observations=observations,
                events=events,
                scenes=[],
                objects=[],
                metadata=metadata,
                partial=False,
                contract_version="via_v2",
            )

        except Exception as e:
            raise ValueError(f"Failed to parse VLM response: {e}") from e

    def _parse_ask_chunk_response(
        self, request: VideoIntelligenceRequest, chunk: AnalysisChunk, raw_response: str,
        extracted_frames: list[ExtractedFrame] | None = None,
    ) -> dict:
        """Parse one TARGETED_SCAN ask_video chunk's VLM response.

        Same _extract_json / _resolve_temporal_range machinery
        _parse_chunk_response uses for summarize_video, but floor_us/ceil_us are
        this chunk's own absolute
        start_us/end_us (not request.start_us/end_us) - every chunk answers
        the same request.question, over its own real temporal window, so its
        observations/events must be grounded in that window's real absolute
        timestamps rather than misread as request-relative. Raises on empty/
        invalid response so the caller's failed_chunks/partial handling in
        ask_video() applies - never fabricates an answer for a chunk that
        wasn't really seen or answered.
        """
        from app.video_intelligence.models import (
            ConfidenceSource,
        )
        json_str = self._extract_json(raw_response)
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError("VLM chunk response was not a JSON object")

        model_name = self._detector_model()
        floor_us, ceil_us = chunk.start_us, chunk.end_us
        actual_provider = self._vision_provider or "snipra_vlm"

        # Build frame mapping for provenance
        frame_id_to_ts = {f.frame_id: f.timestamp_us for f in (extracted_frames or [])}

        observations = []
        for obs in data.get("observations", []):
            nature, nature_source = self._parse_nature(obs.get("nature"))
            confidence, confidence_source = self._parse_confidence(obs.get("confidence"))
            frame_ids = obs.get("frame_ids", [])
            valid_frame_ids = self._validate_frame_ids(frame_ids, frame_id_to_ts)
            start_us, end_us, temporal_source = self._resolve_temporal_range(
                obs, valid_frame_ids, frame_id_to_ts, floor_us, ceil_us
            )
            prov = self._build_provenance(
                valid_frame_ids, temporal_source, confidence_source, nature_source,
                actual_provider, model_name, floor_us, ceil_us, frame_id_to_ts,
            )
            observations.append({
                "asset_id": request.asset_id,
                "start_us": start_us,
                "end_us": end_us,
                "description": obs.get("description", ""),
                "nature": nature,
                "nature_source": nature_source,
                "confidence": confidence,
                "confidence_source": confidence_source,
                "source": actual_provider,
                "provider": actual_provider,
                "model": model_name,
                "provenance": prov,
            })

        events = []
        for ev in data.get("events", []):
            nature, nature_source = self._parse_nature(ev.get("nature"))
            confidence, confidence_source = self._parse_confidence(ev.get("confidence"))
            frame_ids = ev.get("frame_ids", [])
            valid_frame_ids = self._validate_frame_ids(frame_ids, frame_id_to_ts)
            start_us, end_us, temporal_source = self._resolve_temporal_range(
                ev, valid_frame_ids, frame_id_to_ts, floor_us, ceil_us
            )
            prov = self._build_provenance(
                valid_frame_ids, temporal_source, confidence_source, nature_source,
                actual_provider, model_name, floor_us, ceil_us, frame_id_to_ts,
            )
            events.append({
                "asset_id": request.asset_id,
                "start_us": start_us,
                "end_us": end_us,
                "type": ev.get("type", "event"),
                "description": ev.get("description", ""),
                "confidence": confidence,
                "confidence_source": confidence_source,
                "nature": nature,
                "nature_source": nature_source,
                "provider": actual_provider,
                "model": model_name,
                "observed_action": ev.get("observed_action"),
                "subjects": ev.get("subjects", []),
                "objects": ev.get("objects", []),
                "provenance": prov,
            })

        answer_conf = data.get("confidence")
        answer_conf_source = (
            ConfidenceSource.MODEL_EXPLICIT
            if answer_conf is not None
            else ConfidenceSource.UNKNOWN
        )
        answer = {
            "answer": data.get("answer", ""),
            "confidence": answer_conf,
            "confidence_source": answer_conf_source,
            "supporting_observation_ids": data.get("supporting_observation_ids", []),
            "citations": data.get("citations", []),
            "range_label": f"[{floor_us / 1_000_000:.1f}s-{ceil_us / 1_000_000:.1f}s] ",
        }

        return {"answer": answer, "observations": observations, "events": events}

    @staticmethod
    def _aggregate_ask_video_answers(chunk_answers: list[dict]) -> VideoAnswer:
        """Deterministic, LLM-free aggregation of N per-chunk ask_video answers
        into one VideoAnswer - no second LLM synthesis call.

        VideoAnswer.answer is a single str field (Phase 3.5B.2 contract, see
        app/video_intelligence/models.py); there is no per-segment sub-structure
        to aggregate into, so a single chunk's answer is used verbatim and, for
        multiple chunks, answers are joined with an explicit temporal-range
        label per chunk rather than blended or silently overwritten - the
        least-lossy option that does not invent a new public contract.
        confidence is the mean of the successful chunks' confidences (no other
        principled weighting is available). supporting_observation_ids and
        citations are unioned/concatenated - citations stays list[dict], never
        collapsed into a dict (that was the Phase 3.5B.2 bug).
        confidence_source is DERIVED when aggregated from multiple chunks with valid confidences."""
        from app.video_intelligence.models import ConfidenceSource
        if len(chunk_answers) == 1:
            a = chunk_answers[0]
            return VideoAnswer(
                answer=a["answer"],
                confidence=a["confidence"],
                confidence_source=a.get("confidence_source", ConfidenceSource.UNKNOWN),
                supporting_observation_ids=a["supporting_observation_ids"],
                citations=a["citations"],
            )

        parts: list[str] = []
        valid_confidences: list[float] = []
        supporting_observation_ids: list[str] = []
        citations: list[dict] = []
        for a in chunk_answers:
            text = (a["answer"] or "").strip()
            if text:
                parts.append(f"{a['range_label']}{text}")
            conf = a["confidence"]
            if conf is not None:
                valid_confidences.append(conf)
            for oid in a["supporting_observation_ids"]:
                if oid not in supporting_observation_ids:
                    supporting_observation_ids.append(oid)
            citations.extend(a["citations"])

        # Aggregated confidence is DERIVED if we have at least one valid confidence
        agg_confidence = (
            sum(valid_confidences) / len(valid_confidences) if valid_confidences else None
        )
        agg_confidence_source = (
            ConfidenceSource.DERIVED if valid_confidences else ConfidenceSource.UNKNOWN
        )

        return VideoAnswer(
            answer="\n".join(parts),
            confidence=agg_confidence,
            confidence_source=agg_confidence_source,
            supporting_observation_ids=supporting_observation_ids,
            citations=citations,
        )

    def _extract_json(self, raw: str) -> str:
        stripped = raw.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped
        m = re.search(r"^```json\s*(\{.*?\})\s*```$", stripped, re.DOTALL)
        if m:
            return m.group(1)
        m = re.search(r"^```\s*(\{.*?\})\s*```$", stripped, re.DOTALL)
        if m:
            return m.group(1)
        raise ValueError("NO_VALID_JSON_BLOCK")

    async def _analyze_chunk(
        self,
        request: VideoIntelligenceRequest,
        chunk: dict,
        frames: list[ExtractedFrame],
        chunk_id: str,
    ) -> dict:
        """Analyze one real chunk using its actually-extracted frames.

        Raises on empty frames or an invalid/unparseable VLM response so the caller's
        existing per-chunk failed_chunks/partial handling in summarize_video() applies -
        this never fabricates evidence for a chunk that wasn't really seen or answered.
        """
        if not frames:
            raise RuntimeError(f"No frames extracted for {chunk_id}")
        system_prompt = self._build_summarize_system_prompt()
        prompt = self._build_chunk_prompt(request, chunk)
        raw_response = await asyncio.to_thread(
            self._call_vlm_with_frames, system_prompt, prompt, frames
        )
        return self._parse_chunk_response(request, chunk, raw_response, frames)

    def _parse_chunk_response(
        self, request: VideoIntelligenceRequest, chunk: dict, raw_response: str,
        extracted_frames: list[ExtractedFrame] | None = None,
    ) -> dict:
        """Parse one chunk's VLM response into the plain-dict evidence shape that
        summarize_video()'s aggregation loop expects. Raises on invalid JSON/schema -
        the caller records that as a failed chunk instead of inventing evidence."""
        from app.video_intelligence.models import (
            TemporalSource,
        )
        json_str = self._extract_json(raw_response)
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError("VLM chunk response was not a JSON object")

        model_name = self._detector_model()
        floor_us, ceil_us = chunk["start_us"], chunk["end_us"]
        actual_provider = self._vision_provider or "snipra_vlm"

        # Build frame mapping for provenance
        frame_id_to_ts = {f.frame_id: f.timestamp_us for f in (extracted_frames or [])}

        observations = []
        for obs in data.get("observations", []):
            nature, nature_source = self._parse_nature(obs.get("nature"))
            confidence, confidence_source = self._parse_confidence(obs.get("confidence"))
            frame_ids = obs.get("frame_ids", [])
            valid_frame_ids = self._validate_frame_ids(frame_ids, frame_id_to_ts)
            start_us, end_us, temporal_source = self._resolve_temporal_range(
                obs, valid_frame_ids, frame_id_to_ts, floor_us, ceil_us
            )
            prov = self._build_provenance(
                valid_frame_ids, temporal_source, confidence_source, nature_source,
                actual_provider, model_name, floor_us, ceil_us, frame_id_to_ts,
            )
            observations.append({
                "asset_id": request.asset_id,
                "start_us": start_us,
                "end_us": end_us,
                "description": obs.get("description", ""),
                "nature": nature,
                "nature_source": nature_source,
                "confidence": confidence,
                "confidence_source": confidence_source,
                "source": actual_provider,
                "provider": actual_provider,
                "model": model_name,
                "provenance": prov,
            })

        events = []
        for ev in data.get("events", []):
            nature, nature_source = self._parse_nature(ev.get("nature"))
            confidence, confidence_source = self._parse_confidence(ev.get("confidence"))
            frame_ids = ev.get("frame_ids", [])
            valid_frame_ids = self._validate_frame_ids(frame_ids, frame_id_to_ts)
            start_us, end_us, temporal_source = self._resolve_temporal_range(
                ev, valid_frame_ids, frame_id_to_ts, floor_us, ceil_us
            )
            prov = self._build_provenance(
                valid_frame_ids, temporal_source, confidence_source, nature_source,
                actual_provider, model_name, floor_us, ceil_us, frame_id_to_ts,
            )
            events.append({
                "asset_id": request.asset_id,
                "start_us": start_us,
                "end_us": end_us,
                "type": ev.get("type", "event"),
                "description": ev.get("description", ""),
                "confidence": confidence,
                "confidence_source": confidence_source,
                "nature": nature,
                "nature_source": nature_source,
                "provider": actual_provider,
                "model": model_name,
                "observed_action": ev.get("observed_action"),
                "subjects": ev.get("subjects", []),
                "objects": ev.get("objects", []),
                "provenance": prov,
            })

        scenes = []
        for sc in data.get("scenes", []):
            nature, nature_source = self._parse_nature(sc.get("nature"))
            confidence, confidence_source = self._parse_confidence(sc.get("confidence"))
            frame_ids = sc.get("frame_ids", [])
            valid_frame_ids = self._validate_frame_ids(frame_ids, frame_id_to_ts)
            # Scenes follow the same VIA v2 rule as observations/events: frame
            # timestamps > both finite model offsets > full analysis window.
            start_us, end_us, temporal_source = self._resolve_temporal_range(
                sc, valid_frame_ids, frame_id_to_ts, floor_us, ceil_us
            )
            prov = self._build_provenance(
                valid_frame_ids, temporal_source, confidence_source, nature_source,
                actual_provider, model_name, floor_us, ceil_us, frame_id_to_ts,
            )
            scenes.append({
                "asset_id": request.asset_id,
                "start_us": start_us,
                "end_us": end_us,
                "description": sc.get("description", ""),
                "confidence": confidence,
                "confidence_source": confidence_source,
                "nature": nature,
                "nature_source": nature_source,
                "provider": actual_provider,
                "model": model_name,
                "subjects": sc.get("subjects", []),
                "scene_type": sc.get("scene_type"),
                "continuity": sc.get("continuity"),
                "dominant_objects": sc.get("dominant_objects", []),
                "provenance": prov,
            })

        objects = []
        for obj in data.get("objects", []):
            nature, nature_source = self._parse_nature(obj.get("nature"))
            confidence, confidence_source = self._parse_confidence(obj.get("confidence"))
            frame_ids = obj.get("frame_ids", [])
            valid_frame_ids = self._validate_frame_ids(frame_ids, frame_id_to_ts)
            raw_ts = obj.get("timestamps_us") or []
            has_explicit = bool(raw_ts)
            temporal_source = self._determine_temporal_source(
                valid_frame_ids, frame_id_to_ts, has_explicit
            )
            # Objects: use the model's own timestamps when given (dropping any
            # malformed / non-finite entry), else fall back to validated frame
            # timestamps, else carry no point timestamps at all.
            timestamps_us: list[int] = []
            if raw_ts:
                for ts in raw_ts:
                    parsed = self._parse_optional_offset_us(ts, floor_us, ceil_us)
                    if parsed is not None:
                        timestamps_us.append(parsed)
            elif valid_frame_ids:
                timestamps_us = [
                    frame_id_to_ts[fid] for fid in valid_frame_ids if fid in frame_id_to_ts
                ]
            first_seen = min(timestamps_us) if timestamps_us else None
            last_seen = max(timestamps_us) if timestamps_us else None
            if not timestamps_us and not valid_frame_ids:
                temporal_source = TemporalSource.CHUNK_WINDOW
            prov = self._build_provenance(
                valid_frame_ids, temporal_source, confidence_source, nature_source,
                actual_provider, model_name, floor_us, ceil_us, frame_id_to_ts,
            )
            objects.append({
                "asset_id": request.asset_id,
                "label": obj.get("label", ""),
                "timestamps_us": timestamps_us,
                "confidence": confidence,
                "confidence_source": confidence_source,
                "nature": nature,
                "nature_source": nature_source,
                "provider": actual_provider,
                "model": model_name,
                "first_seen_us": first_seen,
                "last_seen_us": last_seen,
                "provenance": prov,
            })

        return {
            "observations": observations,
            "events": events,
            "scenes": scenes,
            "objects": objects,
        }

    def _deduplicate_events(self, events: list) -> list:
        """Deduplicate events by temporal overlap and description similarity."""
        if not events:
            return []

        sorted_events = sorted(events, key=lambda x: x.get("start_us", 0))
        deduplicated = []

        for event in sorted_events:
            is_duplicate = False
            for existing in deduplicated:
                overlap = min(event.get("end_us", 0), existing.get("end_us", 0)) - max(
                    event.get("start_us", 0), existing.get("start_us", 0)
                )
                # Overlapping in time AND same description -> duplicate.
                if overlap > 0 and event.get("description") == existing.get("description"):
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduplicated.append(event)

        return deduplicated

    def _merge_scenes(self, scenes: list) -> list:
        if not scenes:
            return []
        # Simple merge: sort by start time, merge overlapping
        sorted_scenes = sorted(scenes, key=lambda x: x.get("start_us", 0))
        merged = []
        for scene in sorted_scenes:
            if not merged:
                merged.append(scene)
            else:
                last = merged[-1]
                if scene.get("start_us", 0) <= last.get("end_us", 0) + 1_000_000:
                    # Merge
                    last["end_us"] = max(last.get("end_us", 0), scene.get("end_us", 0))
                    last["description"] = (
                        last.get("description", "") + " " + scene.get("description", "")
                    )
                else:
                    merged.append(scene)
        return merged

    def _merge_objects(self, objects: list) -> list:
        if not objects:
            return []
        # Group by label, merge timestamps
        grouped = {}
        for obj in objects:
            label = obj.get("label", "")
            if label not in grouped:
                grouped[label] = obj
            else:
                grouped[label]["timestamps_us"].extend(obj.get("timestamps_us", []))
        return list(grouped.values())

    def _build_summary(self, request, observations: list, events: list, scenes: list) -> str:
        parts = []
        if observations:
            parts.append(f"Observations: {len(observations)} visual observations recorded.")
        if events:
            parts.append(f"Events detected: {len(events)} events.")
        if scenes:
            parts.append(f"Scenes: {len(scenes)} scenes identified.")
        return " ".join(parts) if parts else "Video analyzed."

    # NOTE: this class intentionally has no base-class `_extract_json`/`_parse_response`
    # duplicates left below - a prior version of this file shadowed the coverage-aware
    # implementations above with dead copies that were silently never executed (Python
    # keeps only the last definition of a method name in a class body). Do not re-add
    # a second definition of either name in this class.
