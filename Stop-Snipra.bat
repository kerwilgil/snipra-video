@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop-snipra.ps1"
REM Sin "pause" en el camino normal: la ventana se cierra sola al detener.
REM Solo se mantiene abierta si algo fallo, para poder leer el error (si no, se
REM cerraria de golpe y no quedaria rastro). stop-snipra.ps1 ya avisa cuando no
REM puede cerrar un proceso y sugiere reintentar como administrador.
if errorlevel 1 (
  echo.
  echo Snipra no se detuvo del todo. Revisa el error de arriba.
  pause
)
