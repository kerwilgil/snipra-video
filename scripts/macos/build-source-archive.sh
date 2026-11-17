#!/usr/bin/env bash
# Genera el paquete de código fuente correspondiente (GPL/LGPL/MPL) de Snipra.app y su sha256.
# Requiere haber ejecutado antes scripts/macos/build-macos-app.sh (lee el inventario del bundle).
# Uso: scripts/macos/build-source-archive.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/backend/pyproject.toml" | head -1)"
OUT="${SNIPRA_MACOS_OUT:-$ROOT/dist-release}"
LIC="$OUT/macos-work/snipra-macos-1.0.0-arm64/Snipra.app/Contents/Resources/licenses"
NAME="snipra-macos-${VERSION}-corresponding-source"
SRC_DIR="$OUT/$NAME"
ZIP="$OUT/$NAME.zip"
[[ -f "$LIC/inventory.json" ]] || { echo "Falta $LIC/inventory.json: ejecuta build-macos-app.sh primero." >&2; exit 1; }
rm -rf "$SRC_DIR" && mkdir -p "$SRC_DIR"
python3 "$ROOT/scripts/macos/fetch-corresponding-source.py" --inventory "$LIC/inventory.json" \
  --formulae "$LIC/formulae" --out "$SRC_DIR/sources"
cp "$LIC/SOURCES.md" "$LIC/THIRD_PARTY_NOTICES.md" "$LIC/inventory.json" "$SRC_DIR/"
cp -R "$LIC/formulae" "$SRC_DIR/formulae"
cp "$ROOT/LICENSE" "$SRC_DIR/LICENSE"
cat > "$SRC_DIR/README.md" <<TXT
# Código fuente correspondiente de Snipra $VERSION (macOS arm64)

- sources/: código fuente de cada componente GPL, LGPL y MPL, verificado por sha256
  (o clonado por git en la revisión de la fórmula). Índice en sources/SOURCE_INDEX.json.
- formulae/: fórmulas Homebrew que describen las opciones de compilación y parches.
- SOURCES.md: URL y sha256 de cada componente.
- Estado: LEGAL_REVIEW_REQUIRED. Ver docs/LICENSE_COMPLIANCE.md del repositorio.
TXT
rm -f "$ZIP" && ditto -c -k --noextattr --norsrc --keepParent "$SRC_DIR" "$ZIP"
(cd "$OUT" && shasum -a 256 "$(basename "$ZIP")" > "$NAME.sha256")
echo "ZIP: $ZIP"; echo "SHA256: $(cut -d' ' -f1 "$OUT/$NAME.sha256")"; echo "Tamaño: $(du -h "$ZIP" | cut -f1)"
