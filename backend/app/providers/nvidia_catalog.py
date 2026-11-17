"""Local-first NIM discovery. IDs come exclusively from the provider catalog.

The optional storage/nvidia_policy.json is re-read on every resolution. It can
override modalities and ranking without a release; it cannot add availability.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_retry_at = 0.0
_retry_source = ""
_usage: ContextVar[list | None] = ContextVar("nvidia_usage", default=None)


@contextmanager
def capture_model_usage():
    usage: list[dict] = []
    token = _usage.set(usage)
    try:
        yield usage
    finally:
        _usage.reset(token)


class NvidiaCatalogError(RuntimeError):
    pass


def _policy() -> dict:
    path = settings.storage_dir / "nvidia_policy.json"
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(policy, dict):
            raise ValueError("La política debe ser un objeto JSON")
        return policy
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise NvidiaCatalogError("No se pudo leer nvidia_policy.json") from exc


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _classify(model: dict, policy: dict) -> dict:
    model_id = model["id"]
    override = policy.get("models", {}).get(model_id, {})
    # /models currently often omits modalities. Surface this uncertainty rather
    # than claiming that a name-derived capability has been verified by NVIDIA.
    modalities = override.get("modalities", model.get("modalities"))
    if modalities is None:
        vision = bool(re.search(
            policy.get("vision_pattern", r"vision|(?:^|[-/])vl(?:[-/]|$)|multimodal|omni|cosmos|"
                       r"fuyu|kosmos|vila|neva|deplot|gemma-[34]"),
            model_id, re.I,
        ))
        excluded = bool(re.search(
            policy.get("exclude_pattern", r"embed|rerank|retriev|diffusion|guard|safety|"
                       r"detector|calibration|reward|parse|nvclip|translate|"
                       r"deplot|kosmos|fuyu|paligemma|vila|neva"),
            model_id, re.I,
        ))
        modalities = [] if excluded else (["text", "vision"] if vision else ["text"])
    return {
        "id": model_id,
        "text": "text" in modalities,
        "vision": "vision" in modalities or "image" in modalities,
        "compatibility_source": "policy" if "modalities" in override else (
            "metadata" if model.get("modalities") is not None else "inferred"
        ),
        "priority": override.get("priority", 0),
    }


def get_catalog(*, refresh: bool = False) -> dict:
    global _retry_at, _retry_source
    policy = _policy()
    ttl = max(30, min(86400, int(policy.get("ttl_seconds", 300))))
    max_age = max(ttl, min(604800, int(policy.get("max_stale_seconds", 86400))))
    source = settings.nvidia_base_url.rstrip("/")
    path = settings.storage_dir / "nvidia_catalog.json"
    with _lock:
        now = time.time()
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if (cached.get("source") != source
                or not isinstance(cached.get("fetched_at"), (int, float))
                or not isinstance(cached.get("models"), list)
                or any(not isinstance(m, dict) or not isinstance(m.get("id"), str)
                       for m in cached["models"])):
                cached = {}
        except (OSError, ValueError, AttributeError):
            cached = {}
        age = now - cached.get("fetched_at", 0)
        stale = age >= ttl
        failed = False
        if refresh or stale:
            if refresh or source != _retry_source or now >= _retry_at:
                try:
                    headers = {"Authorization": f"Bearer {settings.nvidia_api_key}"} if (
                        settings.nvidia_api_key
                    ) else {}
                    response = httpx.get(f"{source}/models", headers=headers, timeout=12)
                    response.raise_for_status()
                    models = response.json()["data"]
                    if not isinstance(models, list) or any(
                        not isinstance(m, dict) or not isinstance(m.get("id"), str)
                        or not m["id"] for m in models
                    ):
                        raise ValueError("Catálogo inválido")
                    cached = {"source": source, "fetched_at": now, "models": models}
                    _write(path, cached)
                    age, stale = 0, False
                    _retry_at = 0
                except (httpx.HTTPError, OSError, ValueError, KeyError, TypeError):
                    failed = True
                    _retry_at, _retry_source = now + 30, source
            else:
                failed = True
        usable = bool(cached) and 0 <= age <= max_age
        classified = [_classify(m, policy) for m in cached["models"]] if usable else []
        models = sorted(
            {m["id"]: m for m in classified if m["text"] or m["vision"]}.values(),
            key=lambda m: m["id"],
        )
        return {
            "models": models,
            "stale": stale or failed,
            "available": usable,
            "fetched_at": cached.get("fetched_at") if usable else None,
            "expires_at": cached.get("fetched_at", 0) + ttl if usable else None,
            "warning": ("No se pudo actualizar NVIDIA; el catálogo puede estar desactualizado."
                        if usable and (stale or failed) else
                        "Catálogo NVIDIA no disponible. Reintenta cuando tengas conexión."
                        if not usable else None),
        }


def resolve_model(value: str | None, *, vision: bool = False) -> str:
    selected = (value or "auto").strip() or "auto"
    catalog = get_catalog()
    if not catalog["available"]:
        raise NvidiaCatalogError(catalog["warning"])
    kind = "vision" if vision else "text"
    compatible = [m for m in catalog["models"] if m[kind]]
    if selected != "auto":
        if not any(m["id"] == selected for m in compatible):
            raise NvidiaCatalogError(
                f"El modelo NVIDIA guardado '{selected}' no está disponible para {kind}. "
                "Elige otro modelo o Automático en Ajustes."
                + (" El catálogo en caché puede estar desactualizado." if catalog["stale"] else "")
            )
        resolved = selected
    else:
        policy = _policy()
        preferred = policy.get(f"{kind}_priority", [])
        # Stable, generic policy; no model ID is compiled into the recommendation.
        compatible.sort(key=lambda m: (
            preferred.index(m["id"]) if m["id"] in preferred else len(preferred),
            -m["priority"],
            m["compatibility_source"] == "inferred",
            not bool(re.search(r"instruct|chat", m["id"], re.I)),
            m["vision"] if not vision else False,
            m["id"],
        ))
        if not compatible:
            raise NvidiaCatalogError(f"NVIDIA no ofrece modelos compatibles para {kind}.")
        resolved = compatible[0]["id"]
    logger.info("NVIDIA selection=%s modality=%s resolved=%s stale=%s",
                selected, kind, resolved, catalog["stale"])
    if (usage := _usage.get()) is not None:
        usage.append({"provider": "nvidia", "selection": selected, "model": resolved,
                      "modality": kind, "catalog_stale": catalog["stale"]})
    return resolved
