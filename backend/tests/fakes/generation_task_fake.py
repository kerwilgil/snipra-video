"""FakeLocalGenerativeRuntime for deterministic tests (SNIPRA 1.6D Block 03).

Implements the LocalGenerativeRuntime contract without GPU, network, or
external dependencies. Produces deterministic output for testing the
generation lifecycle.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from app.editor.generation import (
    ExternalTaskState,
    GenerationExternalTask,
    GenerationModel,
    GenerationProvider,
    GenerationRequest,
    LocalGenerativeRuntime,
    LocalRuntimeState,
)

if TYPE_CHECKING:
    pass


class FakeLocalGenerativeRuntime(LocalGenerativeRuntime):
    """Deterministic fake runtime for testing generation lifecycle.

    - No GPU, no network, no model loading
    - Produces deterministic output video
    - Configurable behaviors for testing all lifecycle paths
    """

    def __init__(
        self,
        *,
        state: LocalRuntimeState = LocalRuntimeState.READY,
        submit_success: bool = True,
        submit_delay: float = 0.0,
        poll_sequence: list[str] | None = None,
        fail_sequence: list[str] | None = None,
        cancel_supported: bool = True,
    ):
        self._state = state
        self._submit_success = submit_success
        self._submit_delay = submit_delay
        self._poll_sequence = poll_sequence or [
            ExternalTaskState.PENDING.value,
            ExternalTaskState.RUNNING.value,
            ExternalTaskState.RUNNING.value,
            ExternalTaskState.SUCCEEDED.value,
        ]
        self._fail_sequence = fail_sequence
        self._cancel_supported = cancel_supported

        self._external_task: GenerationExternalTask | None = None
        self._poll_index = 0
        self._output_path: str | None = None
        self._submit_count = 0

    @property
    def state(self) -> LocalRuntimeState:
        return self._state

    async def probe(self) -> bool:
        return self._state == LocalRuntimeState.READY

    def availability(self) -> LocalRuntimeState:
        return self._state

    def compile_request(self, request: GenerationRequest) -> dict:
        """Compile the canonical request into provider-specific payload."""
        # For fake runtime, just return a simple dict
        return {
            "prompt": request.prompt_intent,
            "duration": request.duration_seconds,
            "aspect_ratio": request.aspect_ratio.value,
            "resolution": request.resolution.value,
            "model": self._state.value if hasattr(self._state, 'value') else str(self._state),
        }

    async def submit(self, compiled_request: dict) -> GenerationExternalTask:
        # For testing, just call the sync version
        return self._submit_sync(compiled_request)

    def _submit_sync(self, compiled_request: dict) -> GenerationExternalTask:
        if self._submit_delay:
            time.sleep(self._submit_delay)

        if not self._submit_success:
            from app.editor.generation import LocalGenerationUnavailable
            raise LocalGenerationUnavailable("fake runtime submit failure")

        self._submit_count += 1

        self._external_task = GenerationExternalTask(
            external_task_id=f"fake-gen-task-{int(time.time() * 1000)}",
            provider=GenerationProvider.LOCAL_LTX,
            model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8,
            state=ExternalTaskState.PENDING,
        )
        self._poll_index = 0
        return self._external_task

    def get_status(self, external_task: GenerationExternalTask) -> GenerationExternalTask:
        # Use the passed external_task if our internal one is not set (e.g., when polling after restart)
        if self._external_task is None:
            self._external_task = external_task
        
        if self._fail_sequence and self._poll_index < len(self._fail_sequence):
            state = self._fail_sequence[self._poll_index]
        elif self._poll_index < len(self._poll_sequence):
            state = self._poll_sequence[self._poll_index]
        else:
            state = ExternalTaskState.SUCCEEDED.value

        self._poll_index += 1
        self._external_task.state = ExternalTaskState(state)

        if state == ExternalTaskState.SUCCEEDED.value:
            self._external_task.completed_at = None  # set by caller if needed
            self._external_task.output_url = self._generate_deterministic_output()

        return self._external_task

    def cancel(self, external_task: GenerationExternalTask) -> GenerationExternalTask:
        if not self._cancel_supported:
            from app.editor.generation import LocalGenerationUnavailable
            raise LocalGenerationUnavailable("cancel not supported")

        self._external_task.state = ExternalTaskState.CANCELLED
        self._external_task.completed_at = None
        return self._external_task

    def fetch_result(self, external_task: GenerationExternalTask) -> bytes:
        """Fetch generated artifact bytes."""
        output_path = self._generate_deterministic_output()
        return Path(output_path).read_bytes()

    def fetch_result_to_path(self, external_task: GenerationExternalTask, output_path: str) -> None:
        """Fetch and write result to local path."""
        src = self._generate_deterministic_output()
        Path(output_path).write_bytes(Path(src).read_bytes())

    def _generate_deterministic_output(self) -> str:
        """Generate a small deterministic MP4 file for testing."""
        if self._output_path and Path(self._output_path).exists():
            return self._output_path

        # Create a minimal valid MP4 (1x1 black frame, 1 second)
        import subprocess
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name

        # Use ffmpeg to generate a tiny test video
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=c=black:size=16x16:rate=30:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-f", "mp4", output_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0:
            # Fallback: create a minimal MP4 manually
            Path(output_path).write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")

        return output_path

    def configure_for_success(self) -> None:
        """Configure to succeed quickly."""
        self._poll_sequence = [
            ExternalTaskState.PENDING.value,
            ExternalTaskState.RUNNING.value,
            ExternalTaskState.SUCCEEDED.value,
        ]
        self._poll_index = 0

    def configure_for_failure(self, error: str = "fake failure") -> None:
        """Configure to fail."""
        self._fail_sequence = [error]
        self._poll_sequence = []

    def configure_for_cancellation(self) -> None:
        """Configure for cancellation path."""
        self._poll_sequence = [
            ExternalTaskState.PENDING.value,
            ExternalTaskState.RUNNING.value,
            ExternalTaskState.CANCELLED.value,
        ]
        self._poll_index = 0


# Export for tests
__all__ = ["FakeLocalGenerativeRuntime"]