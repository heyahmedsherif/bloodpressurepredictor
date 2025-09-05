#!/usr/bin/env python3
"""
Test camera recording functionality
===================================

This script tests the improved camera recording implementation
to ensure it doesn't hang and provides proper progress feedback.

Usage: python scripts/test_camera_recording.py
"""

import cv2
import time
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_camera_access(camera_id=0):
    """Test basic camera access."""
    print(f"🎥 Testing camera {camera_id} access...")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_id}")
        return False
    
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"✅ Camera {camera_id} working: {width}x{height} at {fps:.1f}fps")
    else:
        print(f"❌ Cannot read frame from camera {camera_id}")
        cap.release()
        return False
    
    cap.release()
    return True

def test_camera_recording(camera_id=0, duration=5.0):
    """Test camera recording with timeout protection."""
    print(f"\n🎬 Testing {duration}s camera recording...")
    
    # Create temporary video file
    temp_video_path = tempfile.mktemp(suffix='.avi')
    
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return False
        
        # Camera properties
        fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        total_frames = int(fps * duration)
        frames_recorded = 0
        start_time = time.time()
        failed_reads = 0
        max_failed_reads = 50
        
        print(f"📹 Recording {total_frames} frames at {fps}fps...")
        
        while frames_recorded < total_frames:
            ret, frame = cap.read()
            
            if not ret:
                failed_reads += 1
                if failed_reads > max_failed_reads:
                    print(f"❌ Too many failed reads ({failed_reads})")
                    break
                time.sleep(0.01)
                continue
            
            failed_reads = 0  # Reset on successful read
            out.write(frame)
            frames_recorded += 1
            
            # Progress update every 50 frames
            if frames_recorded % 50 == 0:
                elapsed = time.time() - start_time
                progress = frames_recorded / total_frames * 100
                print(f"  📊 Progress: {progress:.1f}% ({frames_recorded}/{total_frames} frames, {elapsed:.1f}s)")
            
            # Safety timeout
            if time.time() - start_time > duration + 5:
                print("⏰ Recording timeout reached")
                break
        
        total_time = time.time() - start_time
        print(f"✅ Recording complete: {frames_recorded} frames in {total_time:.1f}s")
        
        # Check if video file was created
        if os.path.exists(temp_video_path):
            file_size = os.path.getsize(temp_video_path)
            print(f"📁 Video file created: {file_size} bytes")
            
            # Try to read back the video
            test_cap = cv2.VideoCapture(temp_video_path)
            if test_cap.isOpened():
                test_frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"🔍 Video verification: {test_frame_count} frames readable")
                test_cap.release()
            else:
                print("❌ Cannot read back recorded video")
        
        return frames_recorded > 0
        
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return False
    finally:
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

def test_streamlit_integration():
    """Test that the camera app can import without errors."""
    print(f"\n🖥️ Testing Streamlit app integration...")
    
    try:
        # Test core imports
        from src.core.rppg_integration import rPPGToolboxIntegration
        print("  ✅ rPPG integration imports")
        
        # Test camera app imports (without running main)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "camera_bp_predictor", 
            "src/apps/camera_bp_predictor.py"
        )
        camera_module = importlib.util.module_from_spec(spec)
        
        # This will test imports without running streamlit
        spec.loader.exec_module(camera_module)
        print("  ✅ Camera BP predictor app imports")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False

def main():
    """Run all camera recording tests."""
    print("🚀 Camera Recording Test Suite")
    print("=" * 40)
    
    tests = []
    
    # Test 1: Camera access
    if test_camera_access():
        tests.append(True)
        
        # Test 2: Short recording (5 seconds)
        if test_camera_recording(duration=5.0):
            tests.append(True)
        else:
            tests.append(False)
    else:
        print("⚠️  Skipping recording tests - camera not accessible")
        tests.append(False)
        tests.append(False)
    
    # Test 3: Streamlit integration
    if test_streamlit_integration():
        tests.append(True)
    else:
        tests.append(False)
    
    print("\n" + "=" * 40)
    passed = sum(tests)
    total = len(tests)
    print(f"📋 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Camera recording should work properly.")
        print("\n📝 Try the camera feature:")
        print("   streamlit run streamlit_app.py")
        print("   → Select '📹 Camera BP Predictor (NEW!)'")
        print("   → Click 'Test Camera' first")
        print("   → Then 'Start Recording'")
    else:
        print("⚠️  Some tests failed.")
        if not tests[0]:
            print("   🔧 Camera not accessible - check permissions")
        if not tests[1]:
            print("   🔧 Recording failed - try different camera settings")
        if not tests[2]:
            print("   🔧 Import issues - check dependencies")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)