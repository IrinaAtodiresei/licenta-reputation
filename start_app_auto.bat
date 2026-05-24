@echo off
title REPUTATION DASHBOARD - AUTO START (WINDOWS SAFE MODE)

echo =====================================================
echo     REPUTATION DASHBOARD - AUTO START (WINDOWS SAFE)
echo =====================================================

REM --- Set project directory
set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

echo [INFO] Activating virtual environment...
call venv\Scripts\activate

echo [INFO] Python version (should be 3.10):
python --version

echo [INFO] Checking Torch installation...
python check_torch.py

if %errorlevel%==1 (
    echo [WARNING] Torch not found. Installing Torch CPU...
    pip install --upgrade pip
    pip install torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu
) else (
    echo [INFO] Torch OK.
)

echo [INFO] Installing requirements...
pip install -r requirements.txt

echo [INFO] Checking VADER lexicon...
python check_vader.py

echo [INFO] Starting Streamlit using python.exe from venv...
venv\Scripts\python.exe -m streamlit run app/streamlit_app.py

echo.
echo =====================================================
echo   SCRIPT FINISHED. PRESS ANY KEY TO CLOSE WINDOW.
echo =====================================================
pause >nul
