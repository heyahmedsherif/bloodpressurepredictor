#!/bin/bash

# Conda Environment Setup Script
# Fixes numpy/pandas compatibility issues with pinned versions

echo "🔧 Setting up PaPaGei Blood Pressure Predictor with Conda..."

# Remove existing environment if it exists
echo "🗑️ Removing existing bloodpressure environment (if exists)..."
conda env remove -n bloodpressure -y 2>/dev/null || true

# Create new environment from yml file
echo "📦 Creating new conda environment with compatible versions..."
conda env create -f environment.yml

# Activate the environment
echo "🔧 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bloodpressure

# Verify installation
echo "✅ Verifying installation..."
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Test streamlit-webrtc import
echo "🔬 Testing WebRTC import..."
python -c "
try:
    from streamlit_webrtc import VideoTransformerBase
    print('✅ streamlit-webrtc: OK')
except ImportError as e:
    print(f'❌ streamlit-webrtc: {e}')
"

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

echo ""
echo "✅ Conda setup complete!"
echo ""
echo "🎯 To run the app:"
echo "   conda activate bloodpressure"
echo "   streamlit run streamlit_app.py"
echo ""
echo "📹 Available camera modes locally:"
echo "   ✅ Real PPG Extraction (WebRTC) - No TURN servers needed"
echo "   ✅ Photo Analysis - Works everywhere"
echo "   ✅ Traditional Camera - Direct OpenCV access"
echo ""
echo "🔬 This setup uses pinned versions to avoid numpy/pandas compatibility issues"