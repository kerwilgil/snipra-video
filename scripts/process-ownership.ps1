# Shared fail-closed identity checks for this checkout (never trust a port/PID alone).
function Test-SnipraProcess([int]$ProcessId, [string]$Root, [string]$Service) {
  $process = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction SilentlyContinue
  if (-not $process -or -not $process.ExecutablePath -or -not $process.CommandLine) { return $false }
  $python = [IO.Path]::GetFullPath((Join-Path $Root 'backend\.venv\Scripts\python.exe'))
  $command = $process.CommandLine.Replace('/', '\')
  if ($Service -in @('backend', 'worker')) {
    $pattern = if ($Service -eq 'backend') { '(?i)\s-m\s+uvicorn\s+app\.main:app(?:\s|$)' } else { '(?i)\s-m\s+app\.worker\.runner(?:\s|$)' }
    if ($command -notmatch $pattern) { return $false }
    if ($process.ExecutablePath.Equals($python, [StringComparison]::OrdinalIgnoreCase)) { return $true }
    # Windows venv python.exe delegates to the base interpreter. Verify the
    # live parent is this checkout's venv launcher with the same service command.
    if ([IO.Path]::GetFileName($process.ExecutablePath) -ne 'python.exe') { return $false }
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($process.ParentProcessId)" -ErrorAction SilentlyContinue
    return ($parent -and $parent.ExecutablePath -and $parent.CommandLine -and
      $parent.ExecutablePath.Equals($python, [StringComparison]::OrdinalIgnoreCase) -and
      $parent.CommandLine -match $pattern)
  }
  if ($Service -eq 'frontend') {
    $vite = [IO.Path]::GetFullPath((Join-Path $Root 'frontend\node_modules\vite\bin\vite.js'))
    if ([IO.Path]::GetFileName($process.ExecutablePath) -ne 'node.exe') { return $false }
    # npm's Windows shim uses node_modules/.bin/../vite/bin/vite.js.
    # Parse the script argument and normalize it before comparing exact paths.
    if ($command -match '^\s*(?:"[^"]+"|\S+)\s+(?:"(?<script>[^"]+)"|(?<script>\S+))(?:\s|$)') {
      try {
        return [IO.Path]::GetFullPath($Matches.script).Equals($vite, [StringComparison]::OrdinalIgnoreCase)
      } catch { return $false }
    }
    return $false
  }
  return $false
}

function Assert-SnipraPort([int]$Port, [string]$Root, [string]$Service) {
  $owners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  foreach ($owner in $owners) {
    if (-not (Test-SnipraProcess $owner.OwningProcess $Root $Service)) {
      throw "Puerto $Port ocupado por otro proceso (PID $($owner.OwningProcess))."
    }
  }
}

function Stop-SnipraOwnedProcess([int]$ProcessId, [string]$Root, [string]$Service) {
  if (Test-SnipraProcess $ProcessId $Root $Service) {
    Stop-Process -Id $ProcessId -Force -ErrorAction Stop
  } else {
    Write-Warning "No se detiene PID ${ProcessId}: no pertenece a Snipra de esta carpeta."
  }
}

function Restart-SnipraPort([int]$Port, [string]$Root, [string]$Service) {
  Assert-SnipraPort $Port $Root $Service
  $owners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  foreach ($owner in $owners) { Stop-SnipraOwnedProcess $owner.OwningProcess $Root $Service }
  if ($owners) { Start-Sleep -Milliseconds 600 }
  if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Puerto $Port sigue ocupado; no se inicia $Service."
  }
}
