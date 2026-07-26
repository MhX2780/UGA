#!/data/data/com.termux/files/usr/bin/bash

# Clear terminal screen
clear

# Define professional ANSI color codes for Termux
RESET="\e[0m"
GREEN="\e[1;32m"
YELLOW="\e[1;33m"
RED="\e[1;31m"
CYAN="\e[1;36m"

echo -e "${CYAN}=======================================================${RESET}"
echo -e "${CYAN}              UGA AUTOMATED TERMUX SETUP               ${RESET}"
echo -e "${CYAN}=======================================================${RESET}"
echo ""

# 1. ENVIRONMENT VERIFICATION & AUTO-INSTALL
echo -e "${CYAN}[1/7] Verifying and updating system dependencies...${RESET}"

# Update Termux package repositories first
echo -e "${CYAN}[INFO]${RESET} Synchronizing package mirrors..."
apt update -y && apt upgrade -y

# Check/Install Git
if ! command -v git &> /dev/null; then
    echo -e "${YELLOW}[WARNING]${RESET} Git is missing. Installing Git..."
    apt install git -y
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Failed to install Git. Check your internet.${RESET}"
        exit 1
    fi
fi
echo -e "${GREEN}[OK]${RESET} Git is ready."

# Check/Install Python
if ! command -v python &> /dev/null; then
    echo -e "${YELLOW}[WARNING]${RESET} Python is missing. Installing Python..."
    apt install python -y
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Failed to install Python.${RESET}"
        exit 1
    fi
fi
echo -e "${GREEN}[OK]${RESET} Python is ready."

# 2. FORCED CLEANUP (Delete folder if exists)
echo ""
echo -e "${CYAN}[2/7] Checking for existing installations...${RESET}"
if [ -d "UGA" ]; then
    echo -e "${YELLOW}[WARNING] Previous 'UGA' folder found. Wiping directory for a clean install...${RESET}"
    rm -rf UGA
    if [ -d "UGA" ]; then
        echo -e "${RED}[ERROR] Could not delete the existing folder. Permission denied.${RESET}"
        exit 1
    fi
    echo -e "${GREEN}[OK]${RESET} Cleaned old directory successfully."
else
    echo -e "${GREEN}[OK]${RESET} No conflicting directories found."
fi

# 3. REPOSITORY CLONING
echo ""
echo -e "${CYAN}[3/7] Cloning fresh repository from GitHub...${RESET}"
git clone https://github.com
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Repository cloning failed. Check your connection.${RESET}"
    exit 1
fi
echo -e "${GREEN}[OK]${RESET} Repository downloaded."

# 4. DIRECTORY TRANSITION
echo ""
echo -e "${CYAN}[4/7] Navigating into project directory...${RESET}"
cd UGA || { echo -e "${RED}[ERROR] Failed to access the 'UGA' directory.${RESET}"; exit 1; }
echo -e "${GREEN}[OK]${RESET} Inside project folder."

# 5. INJECTING REQUIREMENTS
echo ""
echo -e "${CYAN}[5/7] Injecting required library packages...${RESET}"
if [ ! -f "requirements.txt" ]; then
    touch requirements.txt
fi
echo "google-genai>=0.8.0" >> requirements.txt
echo -e "${GREEN}[OK]${RESET} Requirements tracking updated."

# 6. PACKAGE INSTALLATION & UPGRADE
echo ""
echo -e "${CYAN}[6/7] Upgrading package managers and installing dependencies...${RESET}"
echo -e "${CYAN}[INFO]${RESET} Upgrading pip..."
python -m pip install --upgrade pip --quiet 2>/dev/null

echo -e "${CYAN}[INFO]${RESET} Running package installer (this may take a moment)...${RESET}"
pip install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Failed to install required Python modules.${RESET}"
    exit 1
fi
echo -e "${GREEN}[OK]${RESET} All dependencies successfully initialized."

# 7. APPLICATION LAUNCH
echo ""
echo -e "${CYAN}[7/7] Launching UGA Core Engine...${RESET}"
echo -e "${CYAN}-------------------------------------------------------${RESET}"
echo ""

# Execute the application
python cli.py

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${YELLOW}[WARNING] Application terminated with an error code.${RESET}"
fi

echo ""
echo -e "${CYAN}-------------------------------------------------------${RESET}"
echo "Operation finished."
