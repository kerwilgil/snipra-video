param(
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5173
)

# Detener por puertos NO alcanza: el worker (app.worker.runner) no escucha
# ningun puerto, asi que sobrevivia a cada "detener", se quedaba con
# storage/snipra-worker.lock y ningun worker nuevo podia arrancar despues
# ("Ya existe un worker activo"). Los jobs quedaban encolados en silencio.

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
$LockPath = Join-Path $Root "storage\snipra-worker.lock"
. (Join-Path $PSScriptRoot 'process-ownership.ps1')

function Stop-ByPid([int]$ProcessId, [string]$What) {
  if (-not $ProcessId) { return }
  $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
  if (-not $proc) { return }
  $service = if ($What -like 'worker*') { 'worker' } elseif ($What -eq 'frontend') { 'frontend' } else { 'backend' }
  if (-not (Test-SnipraProcess $ProcessId $Root $service)) {
    Write-Warning "No se detiene PID ${ProcessId}: no pertenece a Snipra de esta carpeta."
    return
  }
  try {
    Stop-Process -Id $ProcessId -Force -ErrorAction Stop
    Write-Host "  detenido: $What (PID $ProcessId)"
  } catch {
    Write-Warning "  no se pudo detener $What (PID $ProcessId): $($_.Exception.Message)"
    Write-Warning "  si persiste, abri PowerShell como administrador y repeti."
  }
}

Write-Host "Deteniendo Snipra..."

# 1) API y frontend, por puerto.
foreach ($port in @($BackendPort, $FrontendPort)) {
  $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
  foreach ($conn in $conns) {
    $service = if ($port -eq $BackendPort) { 'backend' } else { 'frontend' }
    Stop-ByPid $conn.OwningProcess $service
  }
}

# 2) Worker segun el PID del lock.
$lockPid = 0
if (Test-Path $LockPath) {
  try { $lockPid = [int](Get-Content $LockPath -Raw).Trim() } catch { $lockPid = 0 }
  Stop-ByPid $lockPid "worker (lock)"
}

# 3) Por si el lock quedo desincronizado: workers de ESTA instalacion.
#    El filtro por ExecutablePath es imprescindible: sin el, detener una copia
#    (p.ej. un release descomprimido para probar) mataba el worker de la otra,
#    dejandola con API y frontend vivos pero sin nadie procesando la cola.
$venvPython = Join-Path $Root "backend\.venv"
$workers = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
  Where-Object {
    Test-SnipraProcess $_.ProcessId $Root 'worker'
  }
foreach ($worker in $workers) {
  if ($worker.ProcessId -ne $lockPid) {
    Stop-ByPid $worker.ProcessId "worker (huerfano)"
  }
}

# 4) El lock solo se borra si ya no queda ningun worker vivo, para no dejar
#    arrancar un segundo worker mientras el viejo sigue procesando.
Start-Sleep -Milliseconds 500
$alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
  Where-Object {
    Test-SnipraProcess $_.ProcessId $Root 'worker'
  }
if ($alive) {
  Write-Warning "Sigue habiendo un worker vivo; no se borra el lock."
} elseif (Test-Path $LockPath) {
  Remove-Item $LockPath -Force -ErrorAction SilentlyContinue
  Write-Host "  lock liberado"
}

Write-Host "Snipra detenido."
