@echo off
REM Local Development Setup Script for Windows
REM Sets up the environment for full local functionality

echo 🚀 Setting up PaPaGei Blood Pressure Predictor for local development...

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install local requirements
echo 📚 Installing local dependencies...
pip install -r requirements-local.txt

REM Create .streamlit directory for local configuration
if not exist ".streamlit" mkdir .streamlit

REM Create local configuration file
echo [browser] > .streamlit\config.toml
echo gatherUsageStats = false >> .streamlit\config.toml
echo. >> .streamlit\config.toml
echo [server] >> .streamlit\config.toml
echo port = 8501 >> .streamlit\config.toml
echo headless = false >> .streamlit\config.toml
echo. >> .streamlit\config.toml
echo [theme] >> .streamlit\config.toml
echo primaryColor = "#FF6B6B" >> .streamlit\config.toml
echo backgroundColor = "#FFFFFF" >> .streamlit\config.toml
echo secondaryBackgroundColor = "#F0F2F6" >> .streamlit\config.toml
echo textColor = "#262730" >> .streamlit\config.toml

echo ✅ Local setup complete!
echo.
echo 🎯 To run the app locally:
echo    venv\Scripts\activate
echo    streamlit run streamlit_app.py
echo.
echo 📹 Available camera modes locally:
echo    ✅ Real PPG Extraction (WebRTC) - No TURN servers needed
echo    ✅ Photo Analysis - Works everywhere  
echo    ✅ Traditional Camera - Direct OpenCV access
echo.
echo 🌐 For cloud deployment, use requirements.txt with Twilio configuration
pause