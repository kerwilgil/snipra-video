param(
  [string]$ShortcutName = "Start Snipra.lnk",
  [switch]$IconOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ExeLauncher = Join-Path $Root "Snipra.exe"
$BatLauncher = Join-Path $Root "Start-Snipra.bat"
$Launcher = if (Test-Path $ExeLauncher) { $ExeLauncher } else { $BatLauncher }
$IcoIcon = Join-Path $Root "snipra.ico"
$Exporter = Join-Path $PSScriptRoot "export-app-icons.py"
$ShortcutPath = Join-Path $Root $ShortcutName

if (!(Test-Path $Launcher)) {
  throw "No existe $Launcher"
}
if (!(Test-Path $Exporter)) {
  throw "No existe el exportador Brand V2: $Exporter"
}

# Un único exportador mantiene PNG e ICO sincronizados. La generación explícita
# (-IconOnly) acepta un Python de tooling mediante SNIPRA_ICON_PYTHON.
if ($IconOnly) {
  $Python = $env:SNIPRA_ICON_PYTHON
  $PythonArgs = @()
  if (!$Python) {
    $PythonCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($PythonCommand) {
      $Python = $PythonCommand.Source
      $PythonArgs = @("-3")
    } else {
      $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
      if ($PythonCommand) { $Python = $PythonCommand.Source }
    }
  }
  if (!$Python) {
    throw "Python no está disponible. Ejecuta scripts\export-app-icons.py en un entorno con Pillow."
  }
  & $Python @PythonArgs $Exporter
  if ($LASTEXITCODE -ne 0) {
    throw "Falló el exportador Brand V2. Instala Pillow en el entorno indicado por SNIPRA_ICON_PYTHON."
  }
  Write-Host "Icono Windows actualizado desde Brand V2: $IcoIcon"
  exit 0
}
if (!(Test-Path $IcoIcon)) {
  throw "No existe el ICO canónico $IcoIcon. Ejecuta scripts\export-app-icons.py en un entorno con Pillow."
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($ShortcutPath)
$shortcut.TargetPath = $Launcher
$shortcut.WorkingDirectory = $Root
$shortcut.IconLocation = $IcoIcon
$shortcut.Description = "Iniciar Snipra"
$shortcut.Save()

Write-Host "Shortcut creado: $ShortcutPath"
Write-Host "Icono canónico usado: $IcoIcon"
