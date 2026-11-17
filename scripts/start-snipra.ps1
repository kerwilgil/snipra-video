param(
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5173,
  [switch]$NoVision,
  [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"
$Storage = Join-Path $Root "storage"
$Logs = Join-Path $Storage "logs"
$EnvFile = Join-Path $Backend ".env"
$EnvExample = Join-Path $Backend ".env.example"
. (Join-Path $PSScriptRoot 'process-ownership.ps1')
# Check both services before installations or process changes.
Assert-SnipraPort $BackendPort $Root 'backend'
Assert-SnipraPort $FrontendPort $Root 'frontend'

function Require-Command($Name, $Hint) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Falta '$Name'. $Hint"
  }
}

function Refresh-Path {
  $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" +
              [System.Environment]::GetEnvironmentVariable("Path", "Machine")
}

# Instala una herramienta con winget si falta (auto-setup en PC limpia).
function Ensure-Command($Name, $WingetId, $Hint) {
  if (Get-Command $Name -ErrorAction SilentlyContinue) { return }
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw "Falta '$Name' y winget no esta disponible. Instala manualmente: $Hint"
  }
  Write-Host "Instalando $Name ($WingetId) con winget..."
  winget install --id $WingetId -e --accept-package-agreements --accept-source-agreements --silent
  Refresh-Path
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Se instalo $Name pero no aparece en PATH. Cierra y reabre la terminal, luego reintenta. ($Hint)"
  }
}

function Test-Port($Port) {
  $client = New-Object Net.Sockets.TcpClient
  try {
    $client.Connect("127.0.0.1", $Port)
    return $true
  } catch {
    return $false
  } finally {
    $client.Close()
  }
}

function Wait-Port($Port, $Name) {
  for ($i = 0; $i -lt 40; $i++) {
    if (Test-Port $Port) {
      Write-Host "$Name listo en puerto $Port"
      return
    }
    Start-Sleep -Milliseconds 500
  }
  Write-Warning "$Name no respondio en puerto $Port. Revisa logs en $Logs"
}

Write-Host "== Snipra launcher =="
Write-Host "Carpeta: $Root"

Write-Host "Verificando dependencias (se instalan automaticamente si faltan)..."
Ensure-Command "uv" "astral-sh.uv" "https://docs.astral.sh/uv/"
Ensure-Command "node" "OpenJS.NodeJS.LTS" "https://nodejs.org"
Ensure-Command "ffmpeg" "Gyan.FFmpeg" "https://www.gyan.dev/ffmpeg/builds/"
Require-Command "npm" "Viene con Node LTS: winget install OpenJS.NodeJS.LTS"
Require-Command "ffprobe" "Viene con FFmpeg: winget install Gyan.FFmpeg"

New-Item -ItemType Directory -Force -Path $Logs | Out-Null

if (!(Test-Path $EnvFile) -and (Test-Path $EnvExample)) {
  Copy-Item $EnvExample $EnvFile
  Write-Host "Creado backend\.env desde .env.example"
}

Push-Location $Backend
try {
  if ($NoVision) {
    Write-Host "Instalando backend (base, sin tracking)..."
    & uv sync
    if ($LASTEXITCODE -ne 0) { throw "uv sync fallo." }
  } else {
    Write-Host "Instalando backend (con 'vision' para tracking de sujeto)..."
    & uv sync --extra vision
    if ($LASTEXITCODE -ne 0) {
      Write-Warning "No se pudo instalar el extra 'vision'. Continuo con base; el reframe usara center/blur."
      & uv sync
      if ($LASTEXITCODE -ne 0) { throw "uv sync fallo." }
    }
  }
} finally {
  Pop-Location
}

Push-Location $Frontend
try {
  if (!(Test-Path (Join-Path $Frontend "node_modules"))) {
    Write-Host "Instalando frontend..."
    & npm install
  } else {
    Write-Host "Frontend ya tiene node_modules"
  }
} finally {
  Pop-Location
}

$Python = Join-Path $Backend ".venv\Scripts\python.exe"
if (!(Test-Path $Python)) {
  throw "No existe $Python despues de uv sync."
}

Restart-SnipraPort $BackendPort $Root 'backend'
Start-Process -WindowStyle Hidden -FilePath $Python -ArgumentList @(
  "-m", "uvicorn", "app.main:app", "--port", "$BackendPort"
) -WorkingDirectory $Backend `
  -RedirectStandardOutput (Join-Path $Logs "backend-api.log") `
  -RedirectStandardError (Join-Path $Logs "backend-api.err.log")

# Worker persistente: toma jobs de SQLite y sobrevive al reinicio de la API.
# El lock interno evita duplicados si el launcher se ejecuta otra vez.
Start-Process -WindowStyle Hidden -FilePath $Python -ArgumentList @(
  "-m", "app.worker.runner"
) -WorkingDirectory $Backend `
  -RedirectStandardOutput (Join-Path $Logs "backend-worker.log") `
  -RedirectStandardError (Join-Path $Logs "backend-worker.err.log")

Restart-SnipraPort $FrontendPort $Root 'frontend'
Start-Process -WindowStyle Hidden -FilePath "npm.cmd" -ArgumentList @(
  "run", "dev", "--", "--host", "127.0.0.1", "--port", "$FrontendPort"
) -WorkingDirectory $Frontend `
  -RedirectStandardOutput (Join-Path $Logs "frontend-vite.log") `
  -RedirectStandardError (Join-Path $Logs "frontend-vite.err.log")

Wait-Port $BackendPort "Backend"
Wait-Port $FrontendPort "Frontend"

$Url = "http://localhost:$FrontendPort"
Write-Host ""
Write-Host "Snipra listo: $Url"
Write-Host "Logs: $Logs"

if (!$NoOpen) {
  Start-Process $Url
}
