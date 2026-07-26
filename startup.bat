@echo off 
cls
echo Intilizing...
git clone https://github.com/MhX2780/UGA
cd UGA
pip install -r requirements.txt 
echo Done Installing Requirements
python cli.py
