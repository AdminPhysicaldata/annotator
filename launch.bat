@echo off
title VIVE Labeler
echo ============================================================
echo    VIVE LABELER - Lancement
echo ============================================================
echo.

:: Find Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python non trouve dans le PATH.
    echo Installez Python 3.10+ depuis https://www.python.org
    pause
    exit /b 1
)

:: Check Python version
python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>nul
if %errorlevel% neq 0 (
    echo [ERREUR] Python 3.10+ requis.
    pause
    exit /b 1
)

:: Move to script directory
cd /d "%~dp0"

echo [1/2] Verification des dependances...
python -c "import PyQt6, cv2, numpy, pandas" 2>nul
if %errorlevel% neq 0 (
    echo Installation des dependances...
    pip install -r requirements.txt
)
echo OK.

echo [2/2] Lancement de VIVE Labeler...
echo.
python -m src.main
if %errorlevel% neq 0 (
    echo.
    echo [ERREUR] L'application s'est terminee avec une erreur.
    pause
)
