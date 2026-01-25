@echo off
REM Local python wrapper: prefer .venv\Scripts\python.exe when present
setlocal
set VENV=%~dp0.venv\Scripts\python.exe
if exist "%VENV%" (
  "%VENV%" %*
) else (
  py %*
)
endlocal
