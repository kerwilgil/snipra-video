#!/usr/bin/env bash
# Gemelo en bash de package-snipra-release.ps1, para empaquetar desde macOS.
# Mismo contenido y misma estructura que el zip de Windows; lo unico distinto es
# el empaquetador: se usa ditto en vez de Compress-Archive porque zip/Compress
# pierden el bit de ejecucion de Snipra.app/Contents/MacOS/Snipra y el .app
# queda inservible al descomprimir.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-dist-release}"
# La version sale del pyproject para no tener que sincronizarla a mano.
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/backend/pyproject.toml" | head -1)"
VERSION="${VERSION:-0.0.0}"

OUT_ROOT="$ROOT/$OUTPUT_DIR"
STAGE="$OUT_ROOT/snipra"
ZIP="$OUT_ROOT/snipra-macos-$VERSION.zip"

INCLUDE=(
  backend
  frontend
  scripts
  extension
  Snipra.app
  # Lanzador sin consola de Windows (equivalente a Snipra.app). Ambos scripts
  # arman el mismo bundle multiplataforma, solo cambia la etiqueta del zip.
  Snipra.exe
  Start-Snipra.bat
  Start-Snipra.ps1
  Start-Snipra.command
  Stop-Snipra.bat
  README.md
  LICENSE
  CHANGELOG.md
  docs
  snipra.ico
  snipra.icns
)

# Artefactos de build y, sobre todo, secretos: backend/.env lleva las API keys
# y backend/cookies.txt la sesion real del navegador (YT_DLP_COOKIES_PATH) --
# ninguno de los dos puede viajar en un release.
REMOVE=(
  backend/.env
  backend/cookies.txt
  backend/.venv
  frontend/node_modules
  extension/node_modules
  frontend/dist
  frontend/.vite
  storage
)

rm -rf "$STAGE"
mkdir -p "$STAGE"

# Only tracked source files can enter a release, including the canonical skill.
git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null
while IFS= read -r -d '' item; do
  case "/$item" in
    */.env.example) ;;
    */.env|*/.env.*|*/cookies.txt|*/Cookies|*/*.cookies.*|*/.git/*|*/.venv/*|*/venv/*|*/node_modules/*|*/storage/*|*/logs/*|*/cache/*|*/.cache/*|*/temp/*|*/tmp/*|*/artifacts/*|*/coverage/*|*/dist/*|*/build/*|*/__pycache__/*|*/.pytest_cache/*|*/.ruff_cache/*|*/.vite/*|*.db|*.sqlite|*.sqlite3|*.log|*.tmp|*.temp|*.bak|*.mp4|*.mkv|*.mov|*.webm|*.mp3|*.wav|*.m4a|*.flac|*.opus) continue ;;
  esac
  [[ -L "$ROOT/$item" ]] && continue
  mkdir -p "$STAGE/$(dirname "$item")"
  cp "$ROOT/$item" "$STAGE/$item"
done < <(git -C "$ROOT" ls-files -z -- "${INCLUDE[@]}" .agents/skills/snipra-video-director)

# El Agent Runtime necesita el Skill canonico en tiempo de ejecucion
# (backend/app/editor/director.py resuelve SKILL_ROOT = PROJECT_DIR/.agents/skills/
# snipra-video-director). No se copia todo .agents/ -- solo esta carpeta, para no
# arrastrar otros skills sueltos (p.ej. snipra-editorial-director2.zip). Fail closed:
# si el Skill canonico no existe en el repo, no tiene sentido seguir empaquetando.
SKILL_SRC="$ROOT/.agents/skills/snipra-video-director"
if [[ ! -f "$SKILL_SRC/SKILL.md" ]]; then
  echo "Falta el Skill canonico .agents/skills/snipra-video-director/SKILL.md en el repo -- abortando empaquetado." >&2
  exit 1
fi
if [[ ! -f "$STAGE/.agents/skills/snipra-video-director/SKILL.md" ]]; then
  echo "SKILL.md no llego al staging tras copiar el Skill canonico -- abortando empaquetado." >&2
  exit 1
fi

for item in "${REMOVE[@]}"; do
  rm -rf "${STAGE:?}/$item"
done

# __pycache__, .pytest_cache, .ruff_cache y .DS_Store pueden aparecer en
# cualquier subcarpeta (no solo backend/), por ejemplo si pytest se corrio
# alguna vez con cwd=frontend por error -- se limpian por nombre en todo el
# arbol en vez de una ruta fija.
find "$STAGE" \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.ruff_cache' \) \
  -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name '.DS_Store' -type f -delete 2>/dev/null || true

mkdir -p "$STAGE/storage"

# Guarda de regresion: evita repetir el bug de v1.3.0 (Agent Runtime empaquetado sin
# Skill). Corre justo antes de ditto, sobre el staging ya armado.
STAGED_SKILLS_DIR="$STAGE/.agents/skills"
STAGED_SKILL_DIR="$STAGED_SKILLS_DIR/snipra-video-director"
if [[ ! -f "$STAGED_SKILL_DIR/SKILL.md" ]]; then
  echo "Guarda de empaquetado: falta SKILL.md canonico en el staging -- abortando." >&2
  exit 1
fi
# Referencias que backend/app/editor/director.py lee por nombre (CORE_REFERENCES +
# todos los valores de INTENT_REFERENCES). Si falta alguna, el Agent Runtime falla
# en tiempo de ejecucion al pedir un intent que la necesite.
REQUIRED_REFERENCES=(
  editorial-core.md execution-safety.md verification.md
  hook-director.md event-model.md best-event-pool.md
  highlight-director.md event-coherence.md
  montage-director.md
  continuity.md pacing.md
  captions.md
  audio.md
  b-roll.md
)
for ref in "${REQUIRED_REFERENCES[@]}"; do
  if [[ ! -f "$STAGED_SKILL_DIR/references/$ref" ]]; then
    echo "Guarda de empaquetado: falta la referencia requerida references/$ref del Skill canonico -- abortando." >&2
    exit 1
  fi
done
if [[ -e "$STAGED_SKILLS_DIR/snipra-editorial-director2.zip" ]]; then
  echo "Guarda de empaquetado: snipra-editorial-director2.zip no debe viajar en el release -- abortando." >&2
  exit 1
fi
AGENTS_ENTRY_COUNT=$(find "$STAGE/.agents" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')
if [[ "$AGENTS_ENTRY_COUNT" != "1" || ! -d "$STAGED_SKILLS_DIR" ]]; then
  echo "Guarda de empaquetado: .agents en staging trae algo distinto de skills/ -- abortando." >&2
  exit 1
fi
SKILLS_ENTRY_COUNT=$(find "$STAGED_SKILLS_DIR" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')
if [[ "$SKILLS_ENTRY_COUNT" != "1" || ! -d "$STAGED_SKILL_DIR" ]]; then
  echo "Guarda de empaquetado: .agents/skills en staging trae algo distinto de snipra-video-director/ -- abortando." >&2
  exit 1
fi

rm -f "$ZIP"
ditto -c -k --sequesterRsrc --keepParent "$STAGE" "$ZIP"

echo "Release package created: $ZIP"
echo "Tamano: $(du -h "$ZIP" | cut -f1)"
