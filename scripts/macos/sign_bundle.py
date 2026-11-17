#!/usr/bin/env python3
"""Firma del bundle Snipra.app: ad hoc, Apple Development o Developer ID Application.

Orden (de dentro hacia fuera): todos los Mach-O anidados (dylibs, .so, ejecutables auxiliares),
después el ejecutable principal y, por último, el bundle. Así cada firma cubre el contenido ya firmado.

Modos:
  --identity -                  firma ad hoc (sin credenciales; NO distribuible)
  --identity "<nombre>"         identidad del Keychain. Para distribución: "Developer ID Application: ..."
  --runtime                     activa Hardened Runtime (--options runtime). Obligatorio para notarizar.
  --entitlements FILE           entitlements (solo los estrictamente necesarios; ver docs/MACOS_SIGNING.md)
  --timestamp                   sello de tiempo seguro (requerido por Developer ID y notarización)

Nunca imprime ni registra credenciales: el Keychain las gestiona codesign directamente.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

MACHO_MAGICS = {
    b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",
}


def is_macho(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) in MACHO_MAGICS
    except OSError:
        return False


def codesign(path: Path, args: argparse.Namespace, entitlements: bool) -> None:
    cmd = ["codesign", "--force", "--sign", args.identity]
    if args.runtime:
        cmd += ["--options", "runtime"]
    cmd += ["--timestamp" if args.timestamp else "--timestamp=none"]
    if entitlements and args.entitlements:
        cmd += ["--entitlements", str(args.entitlements)]
    cmd.append(str(path))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"codesign falló en {path.name}: {result.stderr.strip()[-400:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("app", type=Path)
    parser.add_argument("--identity", default="-")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--entitlements", type=Path)
    parser.add_argument("--timestamp", action="store_true")
    args = parser.parse_args()
    app = args.app.resolve()
    if args.entitlements and not args.entitlements.exists():
        raise SystemExit(f"No existe el entitlements {args.entitlements}")

    nested = []
    for dirpath, _dirs, files in os.walk(app):
        for name in files:
            path = Path(dirpath) / name
            if path.is_symlink() or not is_macho(path):
                continue
            nested.append(path)
    # Más profundos primero: las dylibs antes que quien las carga.
    nested.sort(key=lambda p: (-len(p.parts), str(p)))
    for path in nested:
        codesign(path, args, entitlements=False)
    main_exec = app / "Contents" / "MacOS" / "Snipra"
    codesign(main_exec, args, entitlements=True)
    codesign(app, args, entitlements=True)
    mode = "ad hoc" if args.identity == "-" else f"identidad '{args.identity.split(':')[0]}'"
    print(f"firmados {len(nested)} Mach-O anidados y el bundle ({mode}, runtime={'sí' if args.runtime else 'no'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
