#!/bin/bash

# Local Development Setup Script
# Sets up the environment for full local functionality

echo "🚀 Setting up PaPaGei Blood Pressure Predictor for local development..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install local requirements
echo "📚 Installing local dependencies..."
pip install -r requirements-local.txt

# Create .streamlit directory for local configuration
mkdir -p .streamlit

# Create local configuration file
cat > .streamlit/config.toml << EOF
[browser]
gatherUsageStats = false

[server]
port = 8501
headless = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
EOF

echo "✅ Local setup complete!"
echo ""
echo "🎯 To run the app locally:"
echo "   source venv/bin/activate"
echo "   streamlit run streamlit_app.py"
echo ""
echo "📹 Available camera modes locally:"
echo "   ✅ Real PPG Extraction (WebRTC) - No TURN servers needed"
echo "   ✅ Photo Analysis - Works everywhere"
echo "   ✅ Traditional Camera - Direct OpenCV access"
echo ""
echo "🌐 For cloud deployment, use requirements.txt with Twilio configuration"