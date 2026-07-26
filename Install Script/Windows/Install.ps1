# Clear terminal host screen
Clear-Host

# Set console title and encoding
$host.ui.RawUI.WindowTitle = "UGA Advanced PowerShell Installer & Runner"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "             UGA AUTOMATED POWERSHELL SETUP            " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# 1. ENVIRONMENT VERIFICATION
Write-Host "[1/7] Verifying system dependencies..." -ForegroundColor Cyan

# Check for Git
if ( -not (Get-Command git -ErrorAction SilentlyContinue) ) {
    Write-Host "[ERROR] Git is missing!" -ForegroundColor Red
    Write-Host "Please download and install Git from https://git-scm.com" -ForegroundColor Yellow
    Exit
}
Write-Host "[OK] Git is available." -ForegroundColor Green

# Check for Python
if ( -not (Get-Command python -ErrorAction SilentlyContinue) ) {
    Write-Host "[ERROR] Python is missing!" -ForegroundColor Red
    Write-Host "Please install Python and check 'Add Python to PATH'." -ForegroundColor Yellow
    Exit
}
Write-Host "[OK] Python is available." -ForegroundColor Green

# 2. FORCED CLEANUP (Delete folder if exists)
Write-Host ""
Write-Host "[2/7] Checking for existing installations..." -ForegroundColor Cyan
if (Test-Path -Path "UGA") {
    Write-Host "[WARNING] Previous 'UGA' folder found. Wiping directory for a clean install..." -ForegroundColor Yellow
    # Force recursive removal of the directory
    Remove-Item -Path "UGA" -Recurse -Force -ErrorAction SilentlyContinue
    
    if (Test-Path -Path "UGA") {
        Write-Host "[ERROR] Could not delete the existing folder. File might be in use." -ForegroundColor Red
        Exit
    }
    Write-Host "[OK] Cleaned old directory successfully." -ForegroundColor Green
} else {
    Write-Host "[OK] No conflicting directories found." -ForegroundColor Green
}

# 3. REPOSITORY CLONING
Write-Host ""
Write-Host "[3/7] Cloning fresh repository from GitHub..." -ForegroundColor Cyan
git clone https://github.com
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Repository cloning failed. Check your internet connection." -ForegroundColor Red
    Exit
}
Write-Host "[OK] Repository downloaded." -ForegroundColor Green

# 4. DIRECTORY TRANSITION
Write-Host ""
Write-Host "[4/7] Navigating into project directory..." -ForegroundColor Cyan
if (Test-Path -Path "UGA") {
    Set-Location -Path "UGA"
    Write-Host "[OK] Inside project folder." -ForegroundColor Green
} else {
    Write-Host "[ERROR] Failed to access the 'UGA' directory." -ForegroundColor Red
    Exit
}

# 5. INJECTING REQUIREMENTS
Write-Host ""
Write-Host "[5/7] Injecting required library packages..." -ForegroundColor Cyan
# Ensure requirements file exists
if (-not (Test-Path -Path "requirements.txt")) {
    New-Item -Path "requirements.txt" -ItemType File -Force | Out-Null
}
# Append requirements without duplicate trailing newlines
Add-Content -Path "requirements.txt" -Value "google-genai>=0.8.0"
Write-Host "[OK] Requirements tracking updated." -ForegroundColor Green

# 6. PACKAGE INSTALLATION & UPGRADE
Write-Host ""
Write-Host "[6/7] Upgrading package managers and installing dependencies..." -ForegroundColor Cyan
Write-Host "[INFO] Upgrading pip..." -ForegroundColor Gray
python -m pip install --upgrade pip --user --quiet 2>$null

Write-Host "[INFO] Running package installer (this may take a moment)..." -ForegroundColor Gray
pip install -r requirements.txt --user --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install required Python modules." -ForegroundColor Red
    Exit
}
Write-Host "[OK] All dependencies successfully initialized." -ForegroundColor Green

# 7. APPLICATION LAUNCH
Write-Host ""
Write-Host "[7/7] Launching UGA Core Engine..." -ForegroundColor Cyan
Write-Host "-------------------------------------------------------" -ForegroundColor Cyan
Write-Host ""

# Run the app python engine
python cli.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[WARNING] Application terminated with an error code: $LASTEXITCODE" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "-------------------------------------------------------" -ForegroundColor Cyan
Write-Host "Operation finished. Press any key to close this terminal."
[void][Console]::ReadKey($true)
