"""Detector basado en Nvidia NIM (endpoint compatible con OpenAI). Modelos gratis.

Soporta native JSON Schema para modelos compatibles (Nemotron 3.5 Lightning, Super).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import jsonschema

from app.core.config import settings
from app.providers._llm import run_detection
from app.providers.base import ClipCandidate, ClipDetector, DetectParams
from app.providers.capabilities import get_provider_capabilities
from app.providers.structured_completion import (
    CompletionResult,
    NativeSchemaMixin,
    StructuredCompletionAdapter,
    build_schema_prompt,
    validate_json,
)

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


class NvidiaDetector(ClipDetector, NativeSchemaMixin, StructuredCompletionAdapter):
    name = "nvidia"

    def __init__(self, model: str = "auto"):
        from app.providers.nvidia_catalog import resolve_model

        self.model = resolve_model(model)
        self._caps = get_provider_capabilities("nvidia", self.model)

    @property
    def capabilities(self):
        return self._caps

    def _client(self) -> OpenAI:
        if not settings.nvidia_api_key:
            raise RuntimeError("Falta NVIDIA_API_KEY en .env")
        from openai import OpenAI

        return OpenAI(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
            timeout=120.0,
            max_retries=0,
        )

    def _complete(self, system: str, user: str) -> str:
        """Legacy completion for detection pipeline (no schema)."""
        resp = self._client().chat.completions.create(
            model=self.model,
            temperature=0.4,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def _complete_native_schema(
        self,
        system: str,
        user: str,
        json_schema: dict,
        temperature: float,
        max_tokens: int | None,
    ) -> CompletionResult:
        """Complete using native JSON Schema (response_format)."""
        client = self._client()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "director_output",
                    "strict": True,
                    "schema": json_schema,
                },
            },
        )
        content = resp.choices[0].message.content or ""
        reasoning_content = getattr(resp.choices[0].message, "reasoning_content", None)
        return CompletionResult(
            content=content,
            provider="nvidia",
            model=self.model,
            structured=True,
            input_tokens=resp.usage.prompt_tokens if resp.usage else None,
            output_tokens=resp.usage.completion_tokens if resp.usage else None,
            finish_reason=resp.choices[0].finish_reason,
            reasoning_present=reasoning_content is not None,
            metadata={},
        )

    def _complete_fallback(
        self,
        system: str,
        user: str,
        json_schema: dict | None,
        temperature: float,
        max_tokens: int | None,
    ) -> CompletionResult:
        """Fallback completion with schema in prompt."""
        prompt = system
        if json_schema:
            prompt += build_schema_prompt(json_schema)

        client = self._client()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content or ""
        reasoning_content = getattr(resp.choices[0].message, "reasoning_content", None)

        # Validate if schema was provided
        if json_schema:
            try:
                validate_json(content, json_schema)
            except jsonschema.exceptions.ValidationError:
                logger.warning("Fallback response failed schema validation for %s", self.model)
            except (jsonschema.exceptions.SchemaError, ImportError, ModuleNotFoundError):
                # Infrastructure/validator errors should NOT be swallowed
                raise

        return CompletionResult(
            content=content,
            provider="nvidia",
            model=self.model,
            structured=False,
            input_tokens=resp.usage.prompt_tokens if resp.usage else None,
            output_tokens=resp.usage.completion_tokens if resp.usage else None,
            finish_reason=resp.choices[0].finish_reason,
            reasoning_present=reasoning_content is not None,
            metadata={},
        )

    def detect(self, segments: list[dict], params: DetectParams) -> list[ClipCandidate]:
        return run_detection(self._complete, segments, params)
