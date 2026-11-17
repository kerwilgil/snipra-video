@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-snipra.ps1"
REM Sin "pause" en el camino normal: la ventana se cierra sola al terminar.
REM Solo se mantiene abierta si el arranque fallo, para poder leer el error
REM (antes se cerraba de golpe y no quedaba rastro en pantalla).
if errorlevel 1 (
  echo.
  echo Snipra no pudo arrancar. Revisa el error de arriba y storage\logs\.
  pause
)
