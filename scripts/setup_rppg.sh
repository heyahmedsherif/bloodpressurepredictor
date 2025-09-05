#!/bin/bash

# Setup script for rPPG-Toolbox integration
# This script initializes the rPPG-Toolbox submodule and installs dependencies

set -e  # Exit on any error

echo "🔧 Setting up rPPG-Toolbox for PaPaGei Blood Pressure Predictor"
echo "================================================================="

# Check if we're in the correct directory
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Initialize git submodules
echo "📦 Initializing rPPG-Toolbox submodule..."
git submodule update --init --recursive

# Check if rPPG-Toolbox directory exists
if [ ! -d "external/rppg-toolbox" ]; then
    echo "❌ Error: rPPG-Toolbox submodule not found"
    echo "Please run: git submodule add https://github.com/ubicomplab/rPPG-Toolbox.git external/rppg-toolbox"
    exit 1
fi

echo "✅ rPPG-Toolbox submodule initialized"

# Navigate to rPPG-Toolbox directory
cd external/rppg-toolbox

echo "🐍 Setting up rPPG-Toolbox environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Using conda for rPPG-Toolbox setup..."
    
    # Make setup script executable
    chmod +x setup.sh
    
    # Run the setup script
    bash setup.sh conda
    
    echo "✅ rPPG-Toolbox conda environment created"
    echo "💡 To activate: conda activate rppg-toolbox"
    
elif command -v python3 &> /dev/null; then
    echo "Using pip for rPPG-Toolbox setup..."
    
    # Install dependencies manually using pip
    pip3 install -r requirements.txt
    
    echo "✅ rPPG-Toolbox dependencies installed with pip"
    
else
    echo "❌ Error: Neither conda nor python3 found"
    echo "Please install Python or Anaconda first"
    exit 1
fi

# Return to project root
cd ../..

echo "🧪 Testing rPPG integration..."

# Test Python imports
python3 -c "
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'external', 'rppg-toolbox'))
try:
    import cv2
    import yaml
    import numpy as np
    print('✅ Core dependencies available')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

echo "🎉 rPPG-Toolbox setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Run the Streamlit app: streamlit run streamlit_app.py"
echo "2. Select '📹 Camera BP Predictor (NEW!)' from the dropdown"
echo "3. Allow camera access when prompted"
echo ""
echo "🔍 Troubleshooting:"
echo "- Ensure your camera is connected and not used by other apps"
echo "- Check camera permissions in your system settings"
echo "- For Streamlit Cloud deployment, camera features work in local mode only"
echo ""
echo "📚 For more info, see: https://github.com/ubicomplab/rPPG-Toolbox"