param(
  [string]$AppName = "Snipra",
  [switch]$IconOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$IconDirectory = Join-Path $Root "frontend\image\snipra-brand-v2\app-icons"
$IcnsIcon = Join-Path $Root "snipra.icns"
$AppRoot = Join-Path $Root "$AppName.app"
$Contents = Join-Path $AppRoot "Contents"
$MacOS = Join-Path $Contents "MacOS"
$Resources = Join-Path $Contents "Resources"
$Executable = Join-Path $MacOS $AppName
$BundleIcon = Join-Path $Resources "snipra.icns"
$InfoPlist = Join-Path $Contents "Info.plist"

if (!(Test-Path $IconDirectory)) {
  throw "No existe el pack oficial de iconos $IconDirectory"
}

function Write-Ascii([System.IO.BinaryWriter]$Writer, [string]$Value) {
  $Writer.Write([System.Text.Encoding]::ASCII.GetBytes($Value))
}

function Write-UInt32BE([System.IO.BinaryWriter]$Writer, [int]$Value) {
  $bytes = [System.BitConverter]::GetBytes([UInt32]$Value)
  [Array]::Reverse($bytes)
  $Writer.Write($bytes)
}

$chunks = @(
  @{ Type = "icp4"; Size = 16 },
  @{ Type = "icp5"; Size = 32 },
  @{ Type = "icp6"; Size = 64 },
  @{ Type = "ic07"; Size = 128 },
  @{ Type = "ic08"; Size = 256 },
  @{ Type = "ic09"; Size = 512 },
  @{ Type = "ic10"; Size = 1024 }
)

$iconChunks = foreach ($chunk in $chunks) {
  $variant = Join-Path $IconDirectory "snipra-app-icon-$($chunk.Size).png"
  if (!(Test-Path $variant)) { throw "Falta la variante oficial $variant" }
  [byte[]]$data = [System.IO.File]::ReadAllBytes($variant)
  [pscustomobject]@{ Type = $chunk.Type; Data = $data }
}

$totalLength = 8
foreach ($chunk in $iconChunks) {
  $totalLength += 8 + $chunk.Data.Length
}

$stream = [System.IO.File]::Create($IcnsIcon)
try {
  $writer = New-Object System.IO.BinaryWriter $stream
  try {
    Write-Ascii $writer "icns"
    Write-UInt32BE $writer $totalLength
    foreach ($chunk in $iconChunks) {
      Write-Ascii $writer $chunk.Type
      Write-UInt32BE $writer (8 + $chunk.Data.Length)
      $writer.Write($chunk.Data)
    }
  } finally {
    $writer.Dispose()
  }
} finally {
  $stream.Dispose()
}

New-Item -ItemType Directory -Force -Path $MacOS, $Resources | Out-Null
Copy-Item $IcnsIcon $BundleIcon -Force
foreach ($size in @(16, 32, 48, 128, 256, 512)) {
  Copy-Item (Join-Path $IconDirectory "snipra-app-icon-$size.png") (Join-Path $Resources "icon_$($size)x$($size).png") -Force
}
Copy-Item (Join-Path $IconDirectory "snipra-app-icon-1024.png") (Join-Path $Resources "icon_512x512@2x.png") -Force

if ($IconOnly) {
  Write-Host "Icono macOS actualizado: $IcnsIcon"
  exit 0
}

$NativeLauncher = Join-Path $Root "scripts\macos_launcher.c"
if ((Test-Path $NativeLauncher) -and (Get-Command cc -ErrorAction SilentlyContinue)) {
  & cc $NativeLauncher -o $Executable
  if ($LASTEXITCODE -ne 0) {
    throw "No se pudo compilar $NativeLauncher"
  }
} else {
  $launcher = @'
#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOG_DIR="$ROOT/storage/logs"
mkdir -p "$LOG_DIR"

exec >> "$LOG_DIR/macos-app-launcher.log" 2>&1
echo ""
echo "== Snipra.app launch $(date) =="

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

bash "$ROOT/scripts/start-snipra.sh"
status=$?
if [[ "$status" -ne 0 ]]; then
  echo "Snipra launcher failed with status $status"
  if (exec 3<>"/dev/tcp/127.0.0.1/5173") >/dev/null 2>&1; then
    exec 3>&- 3<&-
    echo "Frontend is already available; suppressing failure alert."
    exit 0
  fi
  if command -v osascript >/dev/null 2>&1; then
    osascript -e 'display alert "Snipra no pudo abrir" message "Revisa storage/logs/macos-app-launcher.log dentro de la carpeta Snipra." as critical' >/dev/null 2>&1 || true
  fi
  exit "$status"
fi
'@
  $launcher -replace "`r`n", "`n" | Set-Content -Path $Executable -Encoding ASCII
}

$plist = @"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>$AppName</string>
  <key>CFBundleIconFile</key>
  <string>snipra.icns</string>
  <key>CFBundleIdentifier</key>
  <string>com.snipra.launcher</string>
  <key>CFBundleName</key>
  <string>$AppName</string>
  <key>CFBundleDisplayName</key>
  <string>$AppName</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
</dict>
</plist>
"@
$plist -replace "`r`n", "`n" | Set-Content -Path $InfoPlist -Encoding ASCII

Write-Host "Icono macOS creado: $IcnsIcon"
Write-Host "App macOS creada: $AppRoot"
Write-Host "En macOS ejecuta: chmod +x '$Executable'"
