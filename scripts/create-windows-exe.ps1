param(
  [string]$AppName = "Snipra"
)

# Equivalente Windows de create-macos-app.ps1: compila scripts/windows_launcher.ps1
# en un .exe sin consola (ps2exe -noConsole) que reemplaza a Start-Snipra.bat como
# forma de arrancar Snipra con doble clic, sin ventana de cmd ni "pause" final.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$IcoIcon = Join-Path $Root "snipra.ico"
$Wrapper = Join-Path $PSScriptRoot "windows_launcher.ps1"
$OutExe = Join-Path $Root "$AppName.exe"

if (!(Test-Path $Wrapper)) {
  throw "No existe $Wrapper"
}
if (!(Test-Path $IcoIcon)) {
  throw "No existe $IcoIcon. Corre primero scripts\export-app-icons.py en un entorno con Pillow."
}

if (!(Get-Module -ListAvailable -Name ps2exe)) {
  Write-Host "Instalando modulo ps2exe (una sola vez, sin admin)..."
  if (!(Get-PackageProvider -Name NuGet -ErrorAction SilentlyContinue)) {
    Install-PackageProvider -Name NuGet -MinimumVersion 2.8.5.201 -Force -Scope CurrentUser | Out-Null
  }
  Install-Module -Name ps2exe -Scope CurrentUser -Force -AllowClobber
}
Import-Module ps2exe

Invoke-ps2exe `
  -inputFile $Wrapper `
  -outputFile $OutExe `
  -iconFile $IcoIcon `
  -noConsole `
  -title $AppName `
  -product $AppName `
  -company "Kerwil Gil" `
  -version "1.0.0.0" `
  -description "Lanza Snipra (backend + worker + frontend) y abre el navegador."

Write-Host "Exe creado: $OutExe"
Write-Host "Doble clic en $OutExe para arrancar Snipra sin consola (logs en storage\logs\windows-app-launcher.log)."
