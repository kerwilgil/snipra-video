"""Decision registry independent of detection/VIA providers."""

from app.decision.engines import DecisionEngine, RulesDecisionEngine

_ENGINES: dict[str, DecisionEngine] = {}


def register_decision_engine(engine: DecisionEngine) -> None:
    name = engine.id()
    if not name or name in _ENGINES:
        raise ValueError(f"Empty or duplicate decision engine: {name!r}")
    _ENGINES[name] = engine


def get_decision_engine(name: str) -> DecisionEngine | None:
    return _ENGINES.get(name)


def resolve_decision_engine(
    name: str, *, job_provider: str | None = None, job_model: str | None = None,
):
    provider = (job_provider or "naive") if name == "auto" else name
    if provider in {"naive", "codex", "claude_code"}:
        return get_decision_engine("rules")
    if provider in {"openai", "claude", "nvidia"}:
        from app.decision.ai import ExistingAIDecisionEngine

        return ExistingAIDecisionEngine(provider, job_model if provider == job_provider else None)
    return get_decision_engine(provider)


def list_decision_engines() -> list[str]:
    return sorted(_ENGINES)


register_decision_engine(RulesDecisionEngine())
