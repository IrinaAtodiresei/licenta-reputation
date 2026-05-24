@echo off
echo ============================================
echo   STARTING REPUTATION DASHBOARD (SAFE MODE)
echo ============================================

REM --- Setează folderul proiectului (acest fișier trebuie să fie în root)
set PROJECT_DIR=%~dp0
cd /d %PROJECT_DIR%

echo [INFO] Activating virtual environment...
call venv\Scripts\activate

echo [INFO] Python version:
python --version

echo [INFO] Checking Torch installation...
python - <<EOF
import torch
print("Torch version:", torch.__version__)
EOF

echo [INFO] Starting Streamlit from venv...
venv\Scripts\streamlit.exe run app/streamlit_app.py

pause
