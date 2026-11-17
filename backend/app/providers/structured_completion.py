"""Structured Completion Adapter - provider-neutral interface for LLM completion
with optional JSON schema."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jsonschema

from app.providers.capabilities import (
    CompletionResult,
    ProviderCapabilities,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class StructuredCompletionAdapter(ABC):
    """Abstract base for structured completion adapters.

    Providers implement this to support native JSON Schema or fallback modes.
    """

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Capabilities of this provider/model."""
        ...

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict | None = None,
        temperature: float = 0.4,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Complete a chat conversation, optionally with native JSON Schema enforcement.

        Args:
            system: System prompt
            user: User prompt
            json_schema: Optional JSON Schema to enforce. If None, free-form completion.
            temperature: Sampling temperature
            max_tokens: Maximum output tokens

        Returns:
            CompletionResult with content and metadata
        """
        ...


class NativeSchemaMixin:
    """Mixin for providers that support native JSON Schema (OpenAI-compatible)."""

    def _complete_native_schema(
        self,
        system: str,
        user: str,
        json_schema: dict,
        temperature: float,
        max_tokens: int | None,
    ) -> CompletionResult:
        """Complete using native JSON Schema (response_format)."""
        raise NotImplementedError

    def _complete_fallback(
        self,
        system: str,
        user: str,
        json_schema: dict | None,
        temperature: float,
        max_tokens: int | None,
    ) -> CompletionResult:
        """Complete using schema-in-prompt fallback."""
        raise NotImplementedError

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict | None = None,
        temperature: float = 0.4,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Complete with automatic native schema or fallback."""
        if json_schema and self.capabilities.supports_native_json_schema:
            return self._complete_native_schema(system, user, json_schema, temperature, max_tokens)
        return self._complete_fallback(system, user, json_schema, temperature, max_tokens)


class JsonModeMixin:
    """Mixin for providers that support JSON mode but not native schema."""

    def _complete_json_mode(
        self,
        system: str,
        user: str,
        json_schema: dict | None,
        temperature: float,
        max_tokens: int | None,
    ) -> CompletionResult:
        """Complete using JSON mode (response_format={type: 'json_object'})."""
        raise NotImplementedError

    def _complete_fallback(
        self,
        system: str,
        user: str,
        json_schema: dict | None,
        temperature: float,
        max_tokens: int | None,
    ) -> CompletionResult:
        raise NotImplementedError

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict | None = None,
        temperature: float = 0.4,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        if json_schema and self.capabilities.supports_json_mode:
            return self._complete_json_mode(system, user, json_schema, temperature, max_tokens)
        return self._complete_fallback(system, user, json_schema, temperature, max_tokens)


def build_schema_prompt(json_schema: dict) -> str:
    """Build a strict schema instruction for fallback mode."""
    schema_str = json.dumps(json_schema, ensure_ascii=False)
    return (
        f"\n\nOUTPUT CONTRACT:\n"
        f"Return exactly one JSON object matching this JSON Schema.\n"
        f"No Markdown. No code fences. No commentary.\n"
        f"Schema: {schema_str}\n"
    )


def extract_json(raw: str) -> str:
    """Extract JSON from raw response. Accepts pure JSON or a single ```json fence."""
    import re
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


def validate_json(raw: str, json_schema: dict) -> dict:
    """Extract and validate JSON against schema (basic structure only)."""
    json_str = extract_json(raw)
    data = json.loads(json_str)
    jsonschema.validate(data, json_schema)
    return data