"""P1.1 structured visual analysis; shadow-only and fail-closed."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

PROMPT_VERSION = "visual_structured_v1"
ANALYSIS_VERSION = "p1.1"
FRAME_SAMPLING_VERSION = "grid4_v1"
SCORING_PROFILE_VERSION = "gaming_v1"


class StructuredEventFeatures(BaseModel):
    event_present: bool
    event_type: Literal["combat", "resolution", "reaction", "menu_or_idle", "other", "uncertain"]
    event_strength: Literal["none", "low", "medium", "high"]
    payoff_present: bool
    outcome: Literal[
        "kill", "victory", "death", "neutral", "menu", "scoreboard", "idle", "uncertain"
    ]
    reaction_present: bool
    visual_novelty: Literal["none", "low", "medium", "high"]
    menu_or_idle: bool
    peak_frame: int = Field(ge=1, le=4)
    evidence_frames: list[int] = Field(min_length=1, max_length=4)
    evidence_quality: Literal["insufficient", "partial", "clear"]
    observed_evidence: str = Field(max_length=240)


GAMING_V1 = {
    "event_strength": {"none": 0, "low": 12, "medium": 25, "high": 38},
    "visual_novelty": {"none": 0, "low": 3, "medium": 7, "high": 11},
    "outcome": {
        "kill": 16, "victory": 14, "neutral": 0, "uncertain": 0, "death": -18,
        "menu": -35, "scoreboard": -28, "idle": -30,
    },
}


def deterministic_visual_score(
    features: StructuredEventFeatures, profile: dict = GAMING_V1
) -> float:
    if features.menu_or_idle:
        return 0.0
    score = profile["event_strength"][features.event_strength]
    score += profile["visual_novelty"][features.visual_novelty]
    score += profile["outcome"][features.outcome]
    score += 14 if features.payoff_present else 0
    score += 5 if features.reaction_present else 0
    score -= 20 if not features.event_present else 0
    score -= 8 if features.evidence_quality == "insufficient" else 0
    return round(max(0, min(100, score)) / 100, 3)


def structured_prompt(title: str) -> str:
    properties = StructuredEventFeatures.model_json_schema()["properties"]
    schema = json.dumps(properties, ensure_ascii=False)
    instructions = (
        "Describe evidencia observable y sus frames antes de clasificar. Outcome solo con "
        "evidencia inequívoca; no infieras audio, intención ni actor. Ante duda usa uncertain. "
    )
    return f"Eres un verificador visual. Candidato {title!r}. {instructions}SOLO JSON: {schema}"


def parse_structured_response(text: str) -> StructuredEventFeatures:
    try:
        document = text[text.index("{") : text.rindex("}") + 1]
        return StructuredEventFeatures.model_validate_json(document)
    except (ValueError, TypeError) as exc:
        raise ValueError("La respuesta structured visual no cumple el schema.") from exc


def inference_cache_key(
    video: str, start: float, end: float, times: list[float], provider: str, model: str
) -> str:
    parts = [video, f"{start:.3f}", f"{end:.3f}", ",".join(map(str, times))]
    parts += [provider, model, PROMPT_VERSION, ANALYSIS_VERSION, FRAME_SAMPLING_VERSION]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()


cache_key = inference_cache_key  # Backwards-compatible experimental helper alias.


class StructuredVisualCache:
    def __init__(self, path: Path): self.path = path
    def get(self, key: str) -> dict | None:
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS visual_analysis_cache "
                "(cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            row = db.execute(
                "SELECT payload FROM visual_analysis_cache WHERE cache_key=?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None
    def put(self, key: str, payload: dict) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS visual_analysis_cache "
                "(cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            db.execute(
                "INSERT OR REPLACE INTO visual_analysis_cache VALUES (?, ?)",
                (key, json.dumps(payload)),
            )
