@echo off
setlocal

:: ──────────────────────────────────────────────────
::  ALMA - Portable EXE Build
::  Produces a single-file dist\ALMA.exe that runs on
::  any Windows machine without Python installed.
:: ──────────────────────────────────────────────────

set "VENV_DIR=%~dp0.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

echo.
echo [0/3] Checking virtual environment...
if not exist "%VENV_PYTHON%" (
    echo  ERROR: .venv not found. Create it first with:
    echo         uv venv .venv
    pause
    exit /b 1
)
echo  Using: %VENV_PYTHON%

echo.
echo [1/3] Installing runtime + build dependencies...
uv pip install --python "%VENV_PYTHON%" -r "%~dp0requirements.txt" -r "%~dp0requirements-build.txt"
if errorlevel 1 (
    echo  ERROR: Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo [2/3] Building protein_physics extension...
"%VENV_PYTHON%" "%~dp0setup.py" build_ext --inplace
if errorlevel 1 (
    echo  ERROR: Extension build failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Building portable ALMA.exe with PyInstaller...
"%VENV_PYTHON%" -m PyInstaller "%~dp0alma.spec" --noconfirm --clean
if errorlevel 1 (
    echo  ERROR: PyInstaller build failed.
    pause
    exit /b 1
)

echo.
echo Done. Portable executable: dist\ALMA.exe
endlocal
pause
