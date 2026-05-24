@echo off
echo ============================================
echo   Reputation & Sentiment Dashboard - Setup
echo   Initializing environment...
echo ============================================
echo.

REM Create assets and data folders if missing
if not exist assets (
    echo Creating assets folder...
    mkdir assets
)

if not exist data (
    echo Creating data folder...
    mkdir data
)

REM Create virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Could not activate virtual environment.
    echo Make sure Python is installed and added to PATH.
    pause
    exit /b
)

echo Installing required Python packages...
pip install --upgrade pip
pip install streamlit pandas scikit-learn nltk transformers torch requests

echo Downloading NLTK VADER lexicon...
python -m nltk.downloader vader_lexicon

echo.
echo ============================================
echo   Setup completed successfully!
echo   You can now run the app using:
echo       run.bat
echo ============================================
echo.
pause
