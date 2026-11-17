"""Detector basado en Claude (Anthropic)."""

from app.core.config import settings
from app.providers._llm import run_detection
from app.providers.base import ClipCandidate, ClipDetector, DetectParams


class ClaudeDetector(ClipDetector):
    name = "claude"

    def __init__(self, model: str = "claude-haiku-4-5"):
        self.model = model

    def _complete(self, system: str, user: str) -> str:
        if not settings.anthropic_api_key:
            raise RuntimeError("Falta ANTHROPIC_API_KEY en .env")
        from anthropic import Anthropic

        client = Anthropic(api_key=settings.anthropic_api_key)
        # Sin `temperature`: los modelos actuales (sonnet-5, opus-5/4.8/4.7,
        # fable-5) ya no aceptan parametros de sampling y devuelven
        # 400 "`temperature` is deprecated for this model". Ese 400 hacia caer
        # la deteccion al heuristico local en silencio: el job terminaba "done"
        # con clips, pero elegidos sin IA. El default del modelo sirve igual.
        resp = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    def detect(self, segments: list[dict], params: DetectParams) -> list[ClipCandidate]:
        return run_detection(self._complete, segments, params)
