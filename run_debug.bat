@echo off
chcp 65001 >nul
rem Запуск с консолью: видны сообщения и ошибки SimPEG
cd /d "%~dp0"
.venv\Scripts\python.exe run.py %*
pause
