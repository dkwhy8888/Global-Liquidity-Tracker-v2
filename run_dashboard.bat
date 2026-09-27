@echo off
title Global Liquidity Tracker
cd /d "%~dp0"

if not exist venv (
    echo First run - creating environment and installing dependencies, please wait...
    python -m venv venv
    call venv\Scripts\activate.bat
    python -m pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

if not exist manual_inputs.csv copy manual_inputs_template.csv manual_inputs.csv >nul

if "%FRED_API_KEY%"=="" set /p FRED_API_KEY=Enter your FRED API key:

echo.
echo ============================================================
echo   Global Liquidity Tracker is starting...
echo   Your browser will open at http://localhost:8501
echo   (first load takes 1-2 minutes while it fetches live data)
echo.
echo   KEEP THIS WINDOW OPEN. Close it to stop the dashboard.
echo ============================================================
echo.

start "" cmd /c "timeout /t 8 /nobreak >nul & start http://localhost:8501"
python -m streamlit run dashboard.py --server.headless true --server.port 8501

pause
