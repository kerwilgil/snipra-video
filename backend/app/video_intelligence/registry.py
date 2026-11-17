"""Video Intelligence provider registry."""

from __future__ import annotations

from app.video_intelligence.providers import (
    ExistingSnipraVLMProvider,
    VideoIntelligenceProvider,
)


class VIAProviderRegistry:
    """Registry for Video Intelligence providers."""

    def __init__(self):
        self._providers: dict[str, VideoIntelligenceProvider] = {}
        self._register_default_providers()

    def _register_default_providers(self) -> None:
        # Register the existing Snipra VLM provider
        snipra_provider = ExistingSnipraVLMProvider()
        self.register(snipra_provider)

    def register(self, provider) -> None:
        """Register a provider."""
        if not hasattr(provider, "name"):
            raise ValueError("Provider must have a 'name' attribute")
        if not hasattr(provider, "capabilities"):
            raise ValueError("Provider must have a 'capabilities' attribute")
        self._providers[provider.name] = provider

    def get(self, name: str) -> VideoIntelligenceProvider | None:
        """Get a provider by name."""
        return self._providers.get(name)

    def get_for_mode(self, mode: str) -> list:
        """Get all providers that support a specific mode."""
        providers = []
        for provider in self._providers.values():
            if mode == "ask_video" and provider.capabilities.ask_video or mode == "summarize_video" and provider.capabilities.summarize_video:
                providers.append(provider)
        return providers

    def get_available(self, mode: str) -> list:
        """Get available providers for a mode."""
        return self.get_for_mode(mode)

    def select(self, mode: str, preferred: str | None = None):
        """Select the best provider for a mode."""
        providers = self.get_for_mode(mode)

        if not providers:
            raise ValueError(f"No providers available for mode: {mode}")

        if preferred:
            for p in providers:
                if p.name == preferred:
                    return p

        # Return first available (could be smarter: prefer visual over text-only)
        return providers[0]


# Global registry instance
_registry = VIAProviderRegistry()


def get_registry() -> VIAProviderRegistry:
    """Get the global VIA provider registry."""
    return _registry


def get_provider(name: str):
    """Get a provider by name."""
    return _registry.get(name)


def get_provider_for_mode(mode: str, preferred: str | None = None):
    """Get the best provider for a mode."""
    return _registry.select(mode, preferred)


def list_providers_for_mode(mode: str):
    """List all providers supporting a mode."""
    return _registry.get_for_mode(mode)