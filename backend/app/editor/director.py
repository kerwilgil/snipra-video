"""Canonical Skill loading and safe, selective editorial context for Editor IA."""

from __future__ import annotations

import hashlib

from app.core.config import PROJECT_DIR

SKILL_ROOT = PROJECT_DIR / ".agents" / "skills" / "snipra-video-director"
CORE_REFERENCES = ("editorial-core.md", "execution-safety.md", "verification.md")
INTENT_REFERENCES = {
    "hook": ("hook-director.md", "event-model.md", "best-event-pool.md"),
    "highlight": ("highlight-director.md", "event-coherence.md", "best-event-pool.md"),
    "montage": ("montage-director.md", "event-coherence.md", "best-event-pool.md"),
    "rough_cut": ("rough-cut.md", "continuity.md", "pacing.md"),
    "captions": ("captions.md",),
    "audio": ("audio.md",),
    "b_roll": ("b-roll.md",),
}


def _read(relative: str) -> str:
    path = (SKILL_ROOT / relative).resolve()
    if SKILL_ROOT not in path.parents:
        raise ValueError("Invalid Skill reference")
    return path.read_text(encoding="utf-8")


def skill_fingerprint() -> str:
    files = sorted(path for path in SKILL_ROOT.rglob("*") if path.suffix in {".md", ".yaml"})
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(SKILL_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_director_context(intent: str | None = None) -> dict:
    if not (SKILL_ROOT / "SKILL.md").is_file():
        raise FileNotFoundError("Canonical snipra-video-director Skill is unavailable")
    refs = list(CORE_REFERENCES) + list(INTENT_REFERENCES.get(intent or "", ()))
    unique = list(dict.fromkeys(refs))
    return {
        "skill_name": "snipra-video-director",
        "skill_fingerprint": skill_fingerprint(),
        "workflow": _read("SKILL.md"),
        "references": {name: _read(f"references/{name}") for name in unique},
    }


def untrusted_evidence(value: object) -> dict:
    """Mark content-derived text as data before any provider prompt construction."""
    return {"kind": "untrusted_media_evidence", "content": str(value)}
