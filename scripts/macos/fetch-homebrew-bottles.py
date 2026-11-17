#!/usr/bin/env python3
"""Descarga bottles de Homebrew (macOS 15 / arm64_sequoia) para ffmpeg-full y su cierre de dependencias.

Objetivo: obtener FFmpeg y todas sus bibliotecas compiladas con un deployment target de macOS 15,
en un prefijo propio (no modifica el Homebrew del usuario). Cada bottle se verifica con el sha256
publicado en la API de formulae.brew.sh antes de extraerse.

Salida:
  <out>/Cellar/<formula>/<version>/...   contenido de cada bottle (placeholders @@HOMEBREW_*@@ sin tocar)
  <out>/opt/<formula>                    enlace simbólico, para resolver rutas de dependencias
  <out>/bottles.json                     formula, versión, tag, sha256 y URL de cada bottle

Uso: fetch-homebrew-bottles.py --root ffmpeg-full --tag arm64_sequoia --out DIR [--cache DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

API = "https://formulae.brew.sh/api/formula/{}.json"


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def closure(root: str) -> dict[str, dict]:
    """Cierre de dependencias runtime (no build) con su metadata de API."""
    meta: dict[str, dict] = {}
    queue = [root]
    while queue:
        name = queue.pop()
        if name in meta:
            continue
        data = get_json(API.format(name))
        meta[name] = data
        queue.extend(d for d in data.get("dependencies", []) if d not in meta)
    return meta


def bottle_entry(data: dict, tag: str) -> tuple[str, str, str] | None:
    files = (data.get("bottle", {}).get("stable", {}) or {}).get("files", {})
    for candidate in (tag, "all"):
        if candidate in files:
            info = files[candidate]
            return info["url"], info["sha256"], candidate
    return None


def ghcr_repo(bottle_url: str) -> str:
    """Repositorio OCI del bottle, p. ej. 'homebrew/core/icu4c/78' (las fórmulas con @ usan '/')."""
    match = re.search(r"/v2/(homebrew/core/[^/]+(?:/[^/]+)?)/blobs/", bottle_url)
    if not match:
        raise SystemExit(f"URL de bottle no reconocida: {bottle_url}")
    return match.group(1)


def ghcr_token(repo: str) -> str:
    url = f"https://ghcr.io/token?service=ghcr.io&scope=repository:{repo}:pull"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)["token"]


def download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {ghcr_token(ghcr_repo(url))}"})
    with urllib.request.urlopen(request, timeout=600) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="ffmpeg-full")
    parser.add_argument("--tag", default="arm64_sequoia")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()

    out = args.out.resolve()
    cache = (args.cache or out / "downloads").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    (out / "Cellar").mkdir(parents=True, exist_ok=True)
    (out / "opt").mkdir(parents=True, exist_ok=True)

    meta = closure(args.root)
    records = []
    for name in sorted(meta):
        found = bottle_entry(meta[name], args.tag)
        if not found:
            raise SystemExit(f"Sin bottle {args.tag} para {name}")
        url, expected, tag = found
        version = meta[name]["versions"]["stable"]
        revision = meta[name].get("revision", 0)
        keg_version = f"{version}_{revision}" if revision else version
        archive = cache / f"{name}-{keg_version}.{tag}.bottle.tar.gz"
        if not archive.exists() or sha256(archive) != expected:
            download(url, archive)
        actual = sha256(archive)
        if actual != expected:
            raise SystemExit(f"sha256 no coincide para {name}: {actual} != {expected}")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(out / "Cellar", filter="data")
        # El directorio real del keg es el que trae el tar (versión + _revisión de la fórmula).
        kegs = [p for p in (out / "Cellar" / name).iterdir() if p.is_dir()]
        if len(kegs) != 1:
            raise SystemExit(f"Se esperaba un único keg para {name}, encontrados: {[k.name for k in kegs]}")
        keg = kegs[0]
        keg_version = keg.name
        if keg_version != (f"{version}_{revision}" if revision else version):
            print(f"  aviso: keg {name} es {keg_version} (API: {version}, revisión {revision})")
        link = out / "opt" / name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(f"../Cellar/{name}/{keg_version}", link)
        records.append({"formula": name, "version": keg_version, "tag": tag, "sha256": expected, "url": url,
                        "keg": str(keg.relative_to(out))})
        print(f"ok {name} {keg_version} ({tag})")
    (out / "bottles.json").write_text(json.dumps({"root": args.root, "tag": args.tag, "bottles": records},
                                                  indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"bottles: {len(records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
