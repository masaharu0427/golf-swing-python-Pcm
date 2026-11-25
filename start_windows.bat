@echo off
setlocal
cd /d "%~dp0"
if not exist "app.py" goto missing_files
if not exist "requirements.txt" goto missing_files

set "GOLF_PYTHON=.venv\Scripts\python.exe"
if exist "%GOLF_PYTHON%" goto dependencies

py -3.11-64 -V >nul 2>&1
if not errorlevel 1 (
    py -3.11-64 -m venv .venv
    goto dependencies
)
py -3.12-64 -V >nul 2>&1
if not errorlevel 1 (
    py -3.12-64 -m venv .venv
    goto dependencies
)
python -c "import struct, sys; sys.exit(not (sys.version_info[:2] in ((3,11),(3,12)) and struct.calcsize('P') == 8))" >nul 2>&1
if errorlevel 1 goto missing_python
python -m venv .venv

:dependencies
if not exist "%GOLF_PYTHON%" goto failed
if not exist ".venv\golf-dependencies-installed" (
    echo Installing the Golf Swing Analysis packages. This can take several minutes.
    "%GOLF_PYTHON%" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 goto failed
    type nul > ".venv\golf-dependencies-installed"
)
"%GOLF_PYTHON%" -m streamlit run app.py --server.address 127.0.0.1
if errorlevel 1 goto failed
exit /b 0

:missing_files
echo Extract all files from the ZIP before starting the app.
goto failed

:missing_python
echo Install 64-bit Python 3.11 or 3.12 from https://www.python.org/downloads/windows/
echo Then run start_windows.bat again.
goto failed

:failed
echo The app stopped. See the message above for details.
pause
exit /b 1
