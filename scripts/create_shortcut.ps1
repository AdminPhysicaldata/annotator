# create_shortcut.ps1
# Crée un raccourci sur le Bureau pour lancer VIVE Labeler
# Le raccourci : git stash, git pull, puis lance l'app

param(
    [string]$RepoPath = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonPath = "python"
)

# ── Résolution des chemins ──────────────────────────────────────────────────

$repoPath   = (Resolve-Path $RepoPath).Path
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath "VIVE Labeler.lnk"

# ── Script de lancement (launcher.bat dans le repo) ─────────────────────────

$launcherPath = Join-Path $repoPath "launch_windows.bat"

$launcherContent = @"
@echo off
title VIVE Labeler
cd /d "$repoPath"

echo ============================================
echo   VIVE Labeler — Mise a jour du depot...
echo ============================================

echo.
echo [1/3] git stash (sauvegarde modifications locales)...
git stash
if %ERRORLEVEL% neq 0 (
    echo AVERTISSEMENT: git stash a echoue, on continue quand meme.
)

echo.
echo [2/3] git pull (mise a jour depuis le serveur)...
git pull
if %ERRORLEVEL% neq 0 (
    echo ERREUR: git pull a echoue.
    echo Verifiez votre connexion reseau et vos acces au depot.
    pause
    exit /b 1
)

echo.
echo [3/3] Lancement de l'application...
echo ============================================
echo.

$PythonPath -m src
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERREUR: L'application s'est terminee avec une erreur.
    pause
)
"@

Set-Content -Path $launcherPath -Value $launcherContent -Encoding ASCII
Write-Host "✔  Script de lancement créé : $launcherPath"

# ── Création du raccourci .lnk ──────────────────────────────────────────────

$shell    = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)

$shortcut.TargetPath       = $launcherPath
$shortcut.WorkingDirectory = $repoPath
$shortcut.Description      = "Lancer VIVE Labeler (avec git pull)"
$shortcut.WindowStyle      = 1   # Normal window

# Icône : si une icône existe dans le repo, on l'utilise
$iconCandidates = @(
    (Join-Path $repoPath "icon.ico"),
    (Join-Path $repoPath "assets\icon.ico"),
    (Join-Path $repoPath "scripts\icon.ico")
)
foreach ($ico in $iconCandidates) {
    if (Test-Path $ico) {
        $shortcut.IconLocation = $ico
        Write-Host "✔  Icône utilisée : $ico"
        break
    }
}

$shortcut.Save()
Write-Host "✔  Raccourci créé sur le Bureau : $shortcutPath"
Write-Host ""
Write-Host "Terminé. Double-cliquez sur 'VIVE Labeler' depuis le Bureau pour lancer l'app."
