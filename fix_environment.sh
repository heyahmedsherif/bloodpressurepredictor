#!/bin/bash

# Fix existing conda environment with compatible package versions
# This script fixes the numpy/pandas compatibility issue

echo "🔧 Fixing numpy/pandas compatibility in existing environment..."

# Activate the bloodpressure environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bloodpressure

# Check current Python version
echo "📍 Current environment info:"
python --version
which python

# Remove problematic packages first
echo "🗑️ Removing conflicting packages..."
pip uninstall -y numpy pandas scikit-learn matplotlib plotly scipy joblib streamlit opencv-python torch torchvision streamlit-webrtc aiortc 2>/dev/null || true

# Install compatible versions in correct order
echo "📦 Installing compatible core packages..."

# Install numpy first (critical base dependency)
pip install numpy==1.24.4

# Install pandas (depends on numpy)
pip install pandas==2.0.3

# Install other core packages
pip install scikit-learn==1.3.2
pip install matplotlib==3.8.2
pip install scipy==1.11.4
pip install joblib==1.3.2

# Install Streamlit and related
pip install streamlit==1.28.1
pip install plotly==5.17.0

# Install OpenCV
pip install opencv-python==4.8.1.78

# Install PyTorch CPU versions
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install tqdm==4.66.1
pip install PyYAML==6.0.1
pip install yacs==0.1.8
pip install pillow==10.1.0

# Install WebRTC components (these may have build issues, so install last)
echo "📡 Installing WebRTC components..."
pip install aiortc==1.6.0
pip install streamlit-webrtc==0.47.7
pip install twilio==8.10.3

# Verify installation
echo "✅ Verifying installation..."
python -c "import numpy; print(f'✅ NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'✅ Pandas: {pandas.__version__}')"
python -c "import streamlit; print(f'✅ Streamlit: {streamlit.__version__}')"
python -c "import cv2; print(f'✅ OpenCV: {cv2.__version__}')"

# Test the problematic import
echo "🔬 Testing pandas import that was failing..."
python -c "
try:
    import pandas as pd
    print('✅ Pandas import: SUCCESS')
    df = pd.DataFrame({'test': [1, 2, 3]})
    print(f'✅ Pandas functionality: {len(df)} rows created')
except Exception as e:
    print(f'❌ Pandas import failed: {e}')
"

# Test streamlit-webrtc import
echo "🔬 Testing WebRTC import..."
python -c "
try:
    from streamlit_webrtc import VideoTransformerBase
    print('✅ streamlit-webrtc: SUCCESS')
except ImportError as e:
    print(f'❌ streamlit-webrtc: {e}')
"

echo ""
echo "✅ Environment fix complete!"
echo ""
echo "🎯 Try running the app now:"
echo "   conda activate bloodpressure"
echo "   streamlit run streamlit_app.py"
echo ""
echo "If you still get errors, run the full conda setup:"
echo "   ./setup_conda.sh"