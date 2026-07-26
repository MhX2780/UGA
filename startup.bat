@echo off 
cls
echo Intilizing...
echo "google-genai>=0.8.0" >> requirements.txt 
git clone https://github.com/MhX2780/UGA
cd UGA
pip install -r requirements.txt 
echo Done Installing Requirements
python cli.py
