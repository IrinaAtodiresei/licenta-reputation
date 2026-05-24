@echo off
echo ============================================
echo   Reputation ^& Sentiment Dashboard
echo   Starting WITHOUT virtual environment...
echo ============================================
echo.

REM Navigate to app folder
if exist app\streamlit_app.py (
    cd app
) else (
    echo [ERROR] Folder "app" not found.
    echo Make sure your structure is:
    echo   ReputationDashboard\
    echo       run_no_venv.bat
    echo       app\
    echo           streamlit_app.py
    pause
    exit /b
)

echo Launching Streamlit...
echo.

python -m streamlit run streamlit_app.py

echo.
echo Application closed.
pause
