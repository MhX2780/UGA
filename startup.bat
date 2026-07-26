@echo off
cls
title UGA Installer & Runner
color 0A

echo =========================================
echo       Initializing UGA Setup...
echo =========================================

:: 1. الفحص: التأكد من وجود Git
where git >nul 2>nul
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Git is not installed! 
    echo Please install Git from https://git-scm.com and try again.
    goto end
)

:: 2. الفحص: التأكد من وجود Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Python is not installed!
    echo Please install Python and check "Add Python to PATH".
    goto end
)

:: 3. تحميل المشروع
echo.
echo [1/4] Cloning repository...
if exist UGA (
    echo [INFO] Folder 'UGA' already exists. Skipping clone.
    cd UGA
    git pull
) else (
    git clone https://github.com
    if %errorlevel% neq 0 (
        color 0C
        echo [ERROR] Failed to clone repository. Check your internet.
        goto end
    )
    cd UGA
)

:: 4. تحديث ملف المتطلبات
echo.
echo [2/4] Updating requirements.txt...
:: الفحص لمنع تكرار السطر في الملف إذا تم تشغيل السكربت سابقاً
findstr /C:"google-genai" requirements.txt >nul 2>nul
if %errorlevel% neq 0 (
    echo google-genai^>=0.8.0 >> requirements.txt
)

:: 5. تثبيت المكتبات
echo.
echo [3/4] Installing dependencies...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Installation failed.
    goto end
)

:: 6. تشغيل الأداة
echo.
echo [4/4] Starting UGA Tool...
echo =========================================
echo.
python cli.py

:end
echo.
echo =========================================
echo Press any key to exit.
pause >nul
