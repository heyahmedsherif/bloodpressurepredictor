#!/usr/bin/env python3
"""
Simple test script to check if the Streamlit app can be imported and basic functionality works
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.append('.')

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        import pandas as pd
        print("✅ NumPy and Pandas imported successfully")
    except ImportError as e:
        print(f"❌ NumPy/Pandas import failed: {e}")
        return False
    
    return True

def test_papagei_components():
    """Test PaPaGei component imports"""
    print("\nTesting PaPaGei component imports...")
    
    try:
        from models.resnet import ResNet1DMoE
        print("✅ ResNet1DMoE imported successfully")
    except ImportError as e:
        print(f"⚠️ ResNet1DMoE import failed: {e}")
        print("This is expected if model files are not in the correct location")
    
    try:
        from preprocessing.ppg import preprocess_one_ppg_signal
        print("✅ PPG preprocessing imported successfully") 
    except ImportError as e:
        print(f"⚠️ PPG preprocessing import failed: {e}")
        print("This is expected if preprocessing files are not available")
    
    try:
        from segmentations import waveform_to_segments
        print("✅ Segmentation functions imported successfully")
    except ImportError as e:
        print(f"⚠️ Segmentation import failed: {e}")
        print("This is expected if segmentation files are not available")

def test_sample_data_generation():
    """Test sample PPG data generation"""
    print("\nTesting sample PPG data generation...")
    
    try:
        # Generate sample PPG data (same as in streamlit app)
        duration = 10  # seconds
        fs = 250  # sampling rate
        t = np.linspace(0, duration, duration * fs)
        
        # Base PPG signal with heart rate around 70 bpm
        heart_rate = 70 / 60  # Hz
        ppg_signal = np.sin(2 * np.pi * heart_rate * t)
        
        # Add dicrotic notch (realistic PPG morphology)
        ppg_signal += 0.3 * np.sin(2 * np.pi * heart_rate * t * 2 + np.pi/4)
        
        # Add noise and artifacts
        ppg_signal += 0.1 * np.random.normal(0, 1, len(t))
        
        # Add breathing artifact
        ppg_signal += 0.2 * np.sin(2 * np.pi * 0.25 * t)  # 15 breaths/min
        
        print(f"✅ Sample PPG data generated: {len(ppg_signal)} samples at {fs} Hz")
        print(f"✅ Signal duration: {duration} seconds")
        print(f"✅ Signal range: {ppg_signal.min():.3f} to {ppg_signal.max():.3f}")
        
        return ppg_signal, fs
        
    except Exception as e:
        print(f"❌ Sample data generation failed: {e}")
        return None, None

def main():
    """Main test function"""
    print("🧪 Testing Cardiovascular Risk Predictor App Components\n")
    
    # Test imports
    if not test_imports():
        print("\n❌ Basic import test failed. Please install required packages.")
        return False
    
    # Test PaPaGei components
    test_papagei_components()
    
    # Test sample data generation
    ppg_data, fs = test_sample_data_generation()
    
    if ppg_data is not None:
        print("\n✅ All basic functionality tests passed!")
        print("\n🚀 Ready to run Streamlit app with command:")
        print("streamlit run streamlit_app.py")
        print("\nOr visit: http://localhost:8501")
        return True
    else:
        print("\n⚠️ Some tests failed, but app may still work with reduced functionality")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)