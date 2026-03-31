@echo off
:: ============================================================
::  Cree un raccourci "VIVE Labeler" sur le Bureau Windows
::  avec l'icone personnalisee.
::
::  Usage : double-cliquer sur ce fichier.
:: ============================================================

echo Creation du raccourci sur le Bureau...

set "SCRIPT_DIR=%~dp0"
set "ICON=%SCRIPT_DIR%assets\icon.ico"
set "TARGET=%SCRIPT_DIR%launch.bat"
set "SHORTCUT=%USERPROFILE%\Desktop\VIVE Labeler.lnk"

:: Use PowerShell to create the .lnk shortcut
powershell -NoProfile -Command ^
  "$ws = New-Object -ComObject WScript.Shell; " ^
  "$sc = $ws.CreateShortcut('%SHORTCUT%'); " ^
  "$sc.TargetPath = '%TARGET%'; " ^
  "$sc.WorkingDirectory = '%SCRIPT_DIR%'; " ^
  "$sc.IconLocation = '%ICON%'; " ^
  "$sc.Description = 'VIVE Labeler - Annotation multi-camera'; " ^
  "$sc.WindowStyle = 7; " ^
  "$sc.Save()"

if exist "%SHORTCUT%" (
    echo.
    echo ============================================================
    echo   Raccourci cree sur le Bureau !
    echo   Vous pouvez maintenant lancer VIVE Labeler depuis
    echo   l'icone sur votre Bureau.
    echo ============================================================
) else (
    echo.
    echo [ERREUR] Impossible de creer le raccourci.
    echo Essayez de lancer ce script en tant qu'administrateur.
)

echo.
pause
