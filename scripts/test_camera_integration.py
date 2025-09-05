#!/usr/bin/env python3
"""
Test script for camera-based PPG integration
============================================

This script tests the rPPG-Toolbox integration without requiring Streamlit.
Use this to validate the core functionality before running the full app.

Usage: python scripts/test_camera_integration.py
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_core_imports():
    """Test that all core modules can be imported."""
    print("🧪 Testing core imports...")
    
    try:
        from src.core.preprocessing.ppg import preprocess_one_ppg_signal
        print("  ✅ PPG preprocessing")
        
        from src.core.segmentations import waveform_to_segments
        print("  ✅ PPG segmentation")
        
        from src.core.rppg_integration import rPPGToolboxIntegration
        print("  ✅ rPPG integration")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_rppg_initialization():
    """Test rPPG extractor initialization."""
    print("\n🔧 Testing rPPG initialization...")
    
    try:
        from src.core.rppg_integration import rPPGToolboxIntegration
        
        # Test different methods
        methods = ["CHROM", "POS", "ICA", "GREEN"]
        
        for method in methods:
            extractor = rPPGToolboxIntegration(method=method)
            print(f"  ✅ {method} method initialized")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Initialization failed: {e}")
        return False

def test_ppg_generation():
    """Test PPG signal generation and processing."""
    print("\n📊 Testing PPG signal generation...")
    
    try:
        from src.core.rppg_integration import rPPGToolboxIntegration
        
        extractor = rPPGToolboxIntegration("CHROM")
        
        # Generate test PPG signal
        ppg_signal = extractor._generate_realistic_ppg(30.0, 250)
        print(f"  ✅ Generated PPG: {len(ppg_signal)} samples")
        
        # Test conversion to PaPaGei format
        metadata = {
            'method': 'CHROM',
            'heart_rate': 75.0,
            'quality_score': 0.8,
            'sampling_rate': 250
        }
        
        papagei_data = extractor.convert_to_papagei_format(ppg_signal, metadata)
        print(f"  ✅ PaPaGei format: {papagei_data['sampling_rate']}Hz")
        
        return True
        
    except Exception as e:
        print(f"  ❌ PPG generation failed: {e}")
        return False

def test_camera_availability():
    """Test if camera is available (without actually using it)."""
    print("\n📹 Testing camera availability...")
    
    try:
        import cv2
        
        # Try to initialize camera (don't actually capture)
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("  ✅ Camera 0 is available")
            cap.release()
        else:
            print("  ⚠️ Camera 0 not available (may be in use)")
        
        print("  ✅ OpenCV available for camera operations")
        return True
        
    except ImportError:
        print("  ❌ OpenCV not available")
        return False
    except Exception as e:
        print(f"  ❌ Camera test failed: {e}")
        return False

def test_signal_processing():
    """Test basic signal processing functions."""
    print("\n🔄 Testing signal processing...")
    
    try:
        from src.core.preprocessing.ppg import preprocess_one_ppg_signal
        from src.core.segmentations import waveform_to_segments
        
        # Generate test signal
        fs = 250
        duration = 10
        t = np.linspace(0, duration, fs * duration)
        test_signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.normal(size=len(t))
        
        # Test preprocessing
        ppg_processed, _, _, _ = preprocess_one_ppg_signal(test_signal, fs)
        print(f"  ✅ PPG preprocessing: {len(ppg_processed)} samples")
        
        # Test segmentation
        segment_length = fs * 5  # 5-second segments
        segments = waveform_to_segments('ppg', segment_length, clean_signal=ppg_processed)
        print(f"  ✅ PPG segmentation: {segments.shape if segments.ndim > 1 else len(segments)} segments")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Signal processing failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Camera PPG Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_core_imports,
        test_rppg_initialization,
        test_ppg_generation,
        test_signal_processing,
        test_camera_availability
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📋 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Camera PPG integration is ready.")
        print("\n📝 Next steps:")
        print("   1. Run: streamlit run streamlit_app.py")
        print("   2. Select '📹 Camera BP Predictor (NEW!)'")
        print("   3. Allow camera access when prompted")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("\n🔧 Troubleshooting:")
        print("   - Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   - For rPPG-Toolbox: bash scripts/setup_rppg.sh")
        print("   - Check camera permissions in system settings")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)