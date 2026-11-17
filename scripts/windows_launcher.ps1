# Fuente que scripts/create-windows-exe.ps1 compila a Snipra.exe (ps2exe -noConsole).
# Equivalente Windows de scripts/macos_launcher.c: sin consola visible, corre
# start-snipra.ps1 (que ya arranca backend+worker+frontend ocultos y abre el navegador)
# y solo avisa con un dialogo si el frontend nunca llega a levantar.
#
# start-snipra.ps1 se lanza como proceso hijo real (Start-Process + redireccion de
# archivo a nivel de SO), NO con "*>>"/"2>&1" de PowerShell: uv/npm escriben progreso
# por stderr, y bajo $ErrorActionPreference=Stop (que start-snipra.ps1 ya declara)
# una redireccion de stream de PowerShell convierte esas lineas de stderr en errores
# terminantes aunque el comando termine con exit code 0.
#
# No se usa "-Wait": en pruebas reales el proceso hijo terminaba su trabajo (se ve
# completo en el log de salida) pero Start-Process -Wait a veces tardaba mucho en
# detectar la salida real del proceso -- un problema conocido y flaky de .NET con
# procesos hijos redirigidos en Windows. La señal de exito real, igual que en el
# launcher nativo de macOS, es que el puerto del frontend responda -- no el exit
# code del script.

$ErrorActionPreference = "Continue"

$ExePath = (Get-Process -Id $PID).Path
$Root = Split-Path -Parent $ExePath
$LogsDir = Join-Path $Root "storage\logs"
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
$LogPath = Join-Path $LogsDir "windows-app-launcher.log"
$StdOutLog = Join-Path $LogsDir "windows-app-launcher-out.log"
$StdErrLog = Join-Path $LogsDir "windows-app-launcher-err.log"

try { Add-Content -Path $LogPath -Value "" } catch {}
try { Add-Content -Path $LogPath -Value "== Snipra.exe native launch $(Get-Date) ==" } catch {}
try { Add-Content -Path $LogPath -Value "Root: $Root" } catch {}

$StartScript = Join-Path $Root "scripts\start-snipra.ps1"
try {
  Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"' + $StartScript + '"')
  ) -WorkingDirectory $Root -WindowStyle Hidden `
    -RedirectStandardOutput $StdOutLog -RedirectStandardError $StdErrLog
} catch {
  try { Add-Content -Path $LogPath -Value "Snipra launcher failed to spawn: $_" } catch {}
}

function Test-FrontendPort {
  try {
    $client = New-Object System.Net.Sockets.TcpClient
    $client.Connect("127.0.0.1", 5173)
    $ok = $client.Connected
    $client.Close()
    return $ok
  } catch {
    return $false
  }
}

# Primera corrida instala/actualiza uv+npm; dar margen generoso antes de rendirse.
$deadline = (Get-Date).AddSeconds(90)
$portOpen = $false
while ((Get-Date) -lt $deadline) {
  if (Test-FrontendPort) { $portOpen = $true; break }
  Start-Sleep -Seconds 1
}

try { Add-Content -Path $LogPath -Value "Frontend arriba: $portOpen" } catch {}
try { Get-Content -Path $StdOutLog -ErrorAction SilentlyContinue | Add-Content -Path $LogPath -ErrorAction SilentlyContinue } catch {}
try { Get-Content -Path $StdErrLog -ErrorAction SilentlyContinue | Add-Content -Path $LogPath -ErrorAction SilentlyContinue } catch {}

if (-not $portOpen) {
  try {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show(
      "Revisa storage\logs\windows-app-launcher.log dentro de la carpeta Snipra.",
      "Snipra no pudo abrir",
      [System.Windows.Forms.MessageBoxButtons]::OK,
      [System.Windows.Forms.MessageBoxIcon]::Error
    ) | Out-Null
  } catch {}
}
