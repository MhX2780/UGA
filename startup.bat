@echo off
cls
title UGA Advanced Installer & Runner
color 0B

echo =======================================================
echo              UGA AUTOMATED SETUP SCRIPT                
echo =======================================================
echo.

:: 1. ENVIRONMENT VERIFICATION
echo [1/7] Verifying system dependencies...

where git >nul 2>nul
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Git is missing!
    echo Download link: https://git-scm.com
    goto error_exit
)
echo [OK] Git is available.

where python >nul 2>nul
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Python is missing!
    echo Install Python and check "Add Python to PATH".
    goto error_exit
)
echo [OK] Python is available.

:: 2. FORCED CLEANUP (Delete folder if exists)
echo.
echo [2/7] Checking for existing installations...
if exist UGA (
    echo [WARNING] Previous 'UGA' folder found. Wiping directory for a clean install...
    rmdir /s /q UGA
    if exist UGA (
        color 0C
        echo [ERROR] Could not delete the existing folder. File might be in use.
        goto error_exit
    )
    echo [OK] Cleaned old directory successfully.
) else (
    echo [OK] No conflicting directories found.
)

:: 3. REPOSITORY CLONING
echo.
echo [3/7] Cloning fresh repository from GitHub...
git clone https://github.com
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Repository cloning failed. Check your internet connection.
    goto error_exit
)
echo [OK] Repository downloaded.

:: 4. DIRECTORY TRANSITION
echo.
echo [4/7] Navigating into project directory...
cd UGA
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Failed to access the 'UGA' directory.
    goto error_exit
)

:: 5. INJECTING REQUIREMENTS
echo.
echo [5/7] Injecting required library packages...
if not exist requirements.txt (
    echo. > requirements.txt
)
echo google-genai^>=0.8.0 >> requirements.txt
echo [OK] Requirements tracking updated.

:: 6. PACKAGE INSTALLATION & UPGRADE
echo.
echo [6/7] Upgrading package managers and installing dependencies...
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip --user >nul 2>nul

echo [INFO] Running installer (this may take a moment)...
pip install -r requirements.txt --user --quiet
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Failed to install required Python modules.
    goto error_exit
)
echo [OK] All dependencies successfully initialized.

:: 7. APPLICATION LAUNCH
echo.
echo [7/7] Launching UGA Core Engine...
echo -------------------------------------------------------
color 0A
python cli.py
if %errorlevel% neq 0 (
    color 0E
    echo.
    echo [WARNING] Application terminated with an error code: %errorlevel%
)
goto end

:error_exit
echo.
echo -------------------------------------------------------
echo Setup aborted due to a critical error.
echo -------------------------------------------------------

:end
echo.
echo Operation finished. Press any key to close this terminal.
pause >nul
