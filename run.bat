@echo off
echo ============================================
echo   Reputation & Sentiment Dashboard
echo   Starting Streamlit Application...
echo ============================================
echo.

REM Activate virtual environment
call venv\Scripts\activate

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Virtual environment not found.
    echo Please create it with:
    echo     python -m venv venv
    echo Then install dependencies:
    echo     venv\Scripts\activate
    echo     pip install -r requirements.txt
    pause
    exit /b
)

echo Virtual environment activated.
echo.

REM Navigate to app folder
cd app

echo Launching Streamlit...
echo.

python -m streamlit run streamlit_app.py

echo.
echo Application closed.
pause
