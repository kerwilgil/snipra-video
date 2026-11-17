"""Detector basado en la API de OpenAI (chat completions).

Soporta native JSON Schema (gpt-4o, gpt-4o-mini).
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


class OpenAIDetector(ClipDetector, NativeSchemaMixin, StructuredCompletionAdapter):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._caps = get_provider_capabilities("openai", model)

    @property
    def capabilities(self):
        return self._caps

    def _client(self) -> OpenAI:
        if not settings.openai_api_key:
            raise RuntimeError("Falta OPENAI_API_KEY en .env")
        from openai import OpenAI

        return OpenAI(api_key=settings.openai_api_key)

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
        return CompletionResult(
            content=content,
            provider="openai",
            model=self.model,
            structured=True,
            input_tokens=resp.usage.prompt_tokens if resp.usage else None,
            output_tokens=resp.usage.completion_tokens if resp.usage else None,
            finish_reason=resp.choices[0].finish_reason,
            reasoning_present=False,
        )

    def _complete_fallback(
        self,
        system: str,
        user: str,
        json_schema: dict | None,
        temperature: float,
        max_tokens: int | None,
    ) -> CompletionResult:
        """Fallback completion with schema in prompt (JSON mode)."""
        client = self._client()

        if json_schema:
            # Use JSON mode for fallback
            resp = client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system + build_schema_prompt(json_schema)},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        else:
            resp = client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
            )

        content = resp.choices[0].message.content or ""

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
            provider="openai",
            model=self.model,
            structured=False,
            input_tokens=resp.usage.prompt_tokens if resp.usage else None,
            output_tokens=resp.usage.completion_tokens if resp.usage else None,
            finish_reason=resp.choices[0].finish_reason,
            reasoning_present=False,
        )

    def detect(self, segments: list[dict], params: DetectParams) -> list[ClipCandidate]:
        return run_detection(self._complete, segments, params)