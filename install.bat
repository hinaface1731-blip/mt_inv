@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Установка MT-Inv ===
where python >nul 2>nul
if errorlevel 1 (
  echo Не найден Python. Установите Python 3.11 или 3.12 с https://www.python.org/downloads/
  echo и при установке отметьте галочку "Add python.exe to PATH".
  pause
  exit /b 1
)
if not exist .venv (
  echo Создаю виртуальное окружение .venv ...
  python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Ошибка установки зависимостей.
  pause
  exit /b 1
)
echo Пробую поставить быстрый решатель Pardiso (не обязательно)...
python -m pip install pydiso || echo Pardiso не установился - будет использован более медленный решатель.
echo.
echo Готово. Запускайте run.bat
pause
