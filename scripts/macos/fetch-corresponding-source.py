#!/usr/bin/env python3
"""Descarga el código fuente correspondiente a los componentes copyleft de Snipra.app.

Lee `inventory.json` (generado por license_inventory.py) y, para cada componente nativo con
licencia GPL/LGPL/MPL, descarga la URL exacta de la fórmula
Homebrew usada, verifica su sha256 y guarda los parches declarados en la fórmula.
Los componentes obtenidos de git se clonan en la revisión indicada por la fórmula.

Uso: fetch-corresponding-source.py --inventory FILE --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

COPYLEFT = re.compile(r"(A?GPL|LGPL|MPL)", re.I)
PATCH_BLOCK = re.compile(r"patch do\s*\n(.*?)\n\s*end", re.S)
PATCH_URL = re.compile(r'url\s+"([^"]+)"')
PATCH_SHA = re.compile(r'sha256\s+"([0-9a-f]{64})"')
REVISION = re.compile(r'revision:?\s+"([0-9a-f]{7,40})"|revision\s+"([^"]+)"')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=180) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def formula_patches(formula_file: Path) -> list[tuple[str, str]]:
    if not formula_file.exists():
        return []
    text = formula_file.read_text(encoding="utf-8", errors="ignore")
    out = []
    for block in PATCH_BLOCK.findall(text):
        url, sha = PATCH_URL.search(block), PATCH_SHA.search(block)
        if url and sha:
            out.append((url.group(1), sha.group(1)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--formulae", type=Path, help="Carpeta con las fórmulas (inventory/formulae)")
    args = parser.parse_args()

    data = json.loads(args.inventory.read_text(encoding="utf-8"))
    formulae_dir = args.formulae or args.inventory.parent / "formulae"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    selected = [c for c in data["components"]
                if c["category"] == "native" and COPYLEFT.search(c["license"])]
    results, failures = [], []
    for comp in selected:
        name, version = comp["name"], comp["version"]
        folder = out / f"{name}-{version}".replace("/", "_")
        folder.mkdir(parents=True, exist_ok=True)
        formula = formulae_dir / Path(comp.get("formula_file", "")).name if comp.get("formula_file") else None
        url, sha = comp.get("source_url", ""), comp.get("source_sha256", "")
        entry = {"component": name, "version": version, "license": comp["license"], "url": url,
                 "sha256": sha, "files": [], "status": "PENDING"}
        try:
            if url.endswith(".git"):
                revision = ""
                if formula and formula.exists():
                    m = REVISION.search(formula.read_text(encoding="utf-8", errors="ignore"))
                    revision = (m.group(1) or m.group(2)) if m else ""
                repo = folder / "src"
                subprocess.run(["git", "clone", "--quiet", url, str(repo)], check=True)
                if revision:
                    subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", revision], check=True)
                entry.update({"status": "GIT_CLONED", "revision": revision})
            elif url:
                archive = folder / Path(url).name
                download(url, archive)
                actual = sha256_file(archive)
                if sha and actual != sha:
                    raise RuntimeError(f"sha256 no coincide: {actual} != {sha}")
                entry.update({"status": "VERIFIED" if sha else "DOWNLOADED_NO_SHA", "sha256_actual": actual})
                entry["files"].append(archive.name)
            for patch_url, patch_sha in formula_patches(formula) if formula else []:
                patch_file = folder / ("patch-" + Path(patch_url).name)
                download(patch_url, patch_file)
                actual = sha256_file(patch_file)
                if actual != patch_sha:
                    raise RuntimeError(f"sha256 del parche no coincide: {patch_url}")
                entry["files"].append(patch_file.name)
            if formula and formula.exists():
                shutil.copy2(formula, folder / formula.name)
                entry["files"].append(formula.name)
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "FAILED"
            entry["error"] = str(exc)[:300]
            failures.append(name)
        results.append(entry)
        print(f"{entry['status']:12} {name} {version}")
    (out / "SOURCE_INDEX.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"componentes: {len(results)} · fallos: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
