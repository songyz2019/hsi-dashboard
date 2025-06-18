@echo off
echo Starting...
setlocal
set "DIR=%~dp0"
cd /d "%DIR%"
%DIR%/bin/uv.exe run main.py