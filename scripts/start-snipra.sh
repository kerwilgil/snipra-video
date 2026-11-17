#!/usr/bin/env bash
set -euo pipefail

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
STORAGE="$ROOT/storage"
LOGS="$STORAGE/logs"

# Finder-launched .app bundles do not inherit the user's shell PATH.
export PATH="/opt/homebrew/opt/ffmpeg-full/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$STORAGE/.uv-cache}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Falta '$1'. $2" >&2
    exit 1
  fi
}

ensure_brew() {
  if command -v brew >/dev/null 2>&1; then return 0; fi
  echo "Instalando Homebrew (gestor de paquetes de macOS)..."
  NONINTERACTIVE=1 /bin/bash -c \
    "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
}

# Instala una herramienta con brew si falta (auto-setup en Mac limpia).
ensure_cmd() {
  local name="$1" pkg="$2"
  if command -v "$name" >/dev/null 2>&1; then return 0; fi
  ensure_brew
  echo "Instalando $name (brew install $pkg)..."
  brew install "$pkg"
}

# Chequeo de puerto sin depender de python (bash /dev/tcp).
port_open() {
  (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1 && exec 3>&- 3<&-
}

restart_port() {
  local port="$1"
  local name="$2"
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi

  # Fail closed if any listener belongs to another program/checkout.
  while IFS= read -r pid; do
    if ! snipra_port_owner "$pid" "$name"; then
      echo "Puerto $port ocupado por otro proceso (PID $pid)." >&2
      return 1
    fi
  done <<< "$pids"
  echo "Reiniciando $name de esta carpeta (puerto $port)..."
  while IFS= read -r pid; do
    if [[ -n "$pid" ]] && snipra_port_owner "$pid" "$name"; then
      kill "$pid" 2>/dev/null || true
    fi
  done <<< "$pids"

  for _ in $(seq 1 20); do
    if ! port_open "$port"; then
      return 0
    fi
    sleep 0.2
  done

  while IFS= read -r pid; do
    if [[ -n "$pid" ]] && snipra_port_owner "$pid" "$name"; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done <<< "$pids"
}

snipra_port_owner() {
  local command
  command="$(ps -p "$1" -o command= 2>/dev/null || true)"
  case "$2" in
    backend) [[ "$command" == "$BACKEND/.venv/bin/python -m uvicorn app.main:app "* ]] ;;
    frontend) [[ "$command" == *" $FRONTEND/node_modules/vite/bin/vite.js "* ]] ;;
    *) return 1 ;;
  esac
}

wait_port() {
  local port="$1"
  local name="$2"
  for _ in $(seq 1 40); do
    if port_open "$port"; then
      echo "$name listo en puerto $port"
      return 0
    fi
    sleep 0.5
  done
  echo "$name no respondio en puerto $port. Revisa logs en $LOGS" >&2
  return 1
}

start_detached() {
  local dir="$1"
  local stdout_log="$2"
  local stderr_log="$3"
  shift 3
  (
    cd "$dir"
    nohup "$@" > "$stdout_log" 2> "$stderr_log" < /dev/null &
    disown
  )
}

echo "== Snipra launcher =="
echo "Carpeta: $ROOT"

echo "Verificando dependencias (se instalan automaticamente si faltan)..."
ensure_cmd uv uv
ensure_cmd node node
ensure_cmd ffmpeg ffmpeg
require_cmd npm "Viene con Node: brew install node"
require_cmd ffprobe "Viene con FFmpeg: brew install ffmpeg"

mkdir -p "$LOGS" "$UV_CACHE_DIR"

if [[ ! -f "$BACKEND/.env" && -f "$BACKEND/.env.example" ]]; then
  cp "$BACKEND/.env.example" "$BACKEND/.env"
  echo "Creado backend/.env desde .env.example"
fi

if [[ "${NO_VISION:-0}" == "1" ]]; then
  echo "Instalando backend (base, sin tracking)..."
  (cd "$BACKEND" && uv sync)
else
  echo "Instalando backend (con 'vision' para tracking de sujeto)..."
  if ! (cd "$BACKEND" && uv sync --extra vision); then
    echo "Aviso: no se pudo instalar 'vision'. Instalacion base; reframe usara center/blur." >&2
    (cd "$BACKEND" && uv sync)
  fi
fi

if [[ ! -d "$FRONTEND/node_modules" ]]; then
  echo "Instalando frontend..."
  (cd "$FRONTEND" && npm install)
else
  echo "Frontend ya tiene node_modules"
fi

PYTHON="$BACKEND/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "No existe $PYTHON despues de uv sync." >&2
  exit 1
fi

restart_port "$BACKEND_PORT" "backend"
start_detached "$BACKEND" "$LOGS/backend-api.log" "$LOGS/backend-api.err.log" \
  "$PYTHON" -m uvicorn app.main:app --port "$BACKEND_PORT"

# Worker persistente: toma jobs de SQLite y sobrevive al reinicio de la API.
# El lock interno evita duplicados si el launcher se ejecuta otra vez.
start_detached "$BACKEND" "$LOGS/backend-worker.log" "$LOGS/backend-worker.err.log" \
  "$PYTHON" -m app.worker.runner

restart_port "$FRONTEND_PORT" "frontend"
start_detached "$FRONTEND" "$LOGS/frontend-vite.log" "$LOGS/frontend-vite.err.log" \
  npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT"

ready_status=0
wait_port "$BACKEND_PORT" "Backend" || ready_status=1
wait_port "$FRONTEND_PORT" "Frontend" || ready_status=1
if [[ "$ready_status" -ne 0 ]]; then
  exit "$ready_status"
fi

URL="http://localhost:$FRONTEND_PORT"
echo ""
echo "Snipra listo: $URL"
echo "Logs: $LOGS"

if command -v open >/dev/null 2>&1; then
  if ! open "$URL"; then
    if ! open -a Safari "$URL" 2>/dev/null; then
      if ! open -a "Google Chrome" "$URL" 2>/dev/null; then
        echo "Aviso: no se pudo abrir el navegador automaticamente. Abre $URL manualmente." >&2
      fi
    fi
  fi
fi
