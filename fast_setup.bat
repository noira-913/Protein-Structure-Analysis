@echo off
setlocal

:: ──────────────────────────────────────────
::  Protein Physics Engine - Fast Setup
:: ──────────────────────────────────────────

set "VENV_DIR=%~dp0.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

:: [0] Check venv exists
echo.
echo [0/3] Checking virtual environment...
if not exist "%VENV_PYTHON%" (
    echo  ERROR: .venv not found. Create it first with:
    echo         uv venv .venv
    pause
    exit /b 1
)
echo  Using: %VENV_PYTHON%

:: Ensure data directory exists for downloaded structures
if not exist "%~dp0data\" mkdir "%~dp0data"

:: [1] Install dependencies from requirements.txt
echo.
echo [1/3] Installing dependencies from requirements.txt...
uv pip install --python "%VENV_PYTHON%" -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo  ERROR: Dependency installation failed.
    pause
    exit /b 1
)

:: [2] Build physics engines
::   - CPU: setuptools finds MSVC automatically (no vcvarsall needed)
::   - GPU: setup.py locates cl.exe via vswhere for nvcc
echo.
echo [2/3] Building physics engines...
"%VENV_PYTHON%" setup.py build_ext --inplace
if errorlevel 1 (
    echo  ERROR: CPU build failed. Ensure Visual Studio Build Tools are installed.
    pause
    exit /b 1
)

:: Report GPU build result
set "NVCC_FOUND=0"
where nvcc >nul 2>&1
if not errorlevel 1 set "NVCC_FOUND=1"
if "%NVCC_FOUND%"=="0" if defined CUDA_PATH (
    if exist "%CUDA_PATH%\bin\nvcc.exe" set "NVCC_FOUND=1"
)
if "%NVCC_FOUND%"=="0" (
    for /d %%d in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
        if exist "%%d\bin\nvcc.exe" set "NVCC_FOUND=1"
    )
)
if "%NVCC_FOUND%"=="1" (
    echo  CUDA found - GPU engine attempted ^(see build output above^).
) else (
    echo  CUDA not found - GPU engine skipped.
)

:: [3] Launch application
echo.
echo [3/3] Launching application...
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
"%VENV_PYTHON%" python/gui_main.py

endlocal
pause
