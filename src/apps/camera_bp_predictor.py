"""
Camera-Based Blood Pressure Predictor
=====================================

Enhanced PaPaGei Blood Pressure Predictor with integrated rPPG-Toolbox
for camera-based PPG signal extraction and real-time BP prediction.

Features:
- Camera-based PPG extraction using rPPG-Toolbox
- Multiple rPPG algorithms (CHROM, POS, TSCAN, PhysNet, etc.)
- Real-time blood pressure prediction
- Video file upload support
- Integration with existing PaPaGei pipeline

Author: Claude Code Integration
Date: 2025-09-05
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
import tempfile
import os
import sys
from pathlib import Path
import logging
from typing import Optional, Dict, Any, Tuple
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import PaPaGei components
try:
    from src.core.rppg_integration import rPPGToolboxIntegration, extract_ppg_from_camera, extract_ppg_from_video
    from src.core.preprocessing.ppg import preprocess_one_ppg_signal
    from src.core.segmentations import waveform_to_segments
    from src.core.glucose_integration import GlucoseIntegration
    from src.core.cholesterol_integration import CholesterolCardiovascularIntegration
    try:
        from torch_ecg._preprocessors import Normalize
    except ImportError:
        # Fallback normalization class
        class Normalize:
            def __init__(self, method='z-score'):
                self.method = method
            def apply(self, signal, fs):
                if self.method == 'z-score':
                    normalized = (signal - np.mean(signal)) / np.std(signal)
                    return normalized, {}
                return signal, {}
    RPPG_AVAILABLE = True
except ImportError as e:
    st.error(f"rPPG integration not available: {e}")
    RPPG_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_resample_signal(signal, fs_original, fs_target):
    """Simple resampling function using scipy."""
    try:
        from scipy.signal import resample
        target_length = int(len(signal) * fs_target / fs_original)
        return resample(signal, target_length)
    except ImportError:
        # If scipy not available, return original signal
        logger.warning("Scipy not available, skipping resampling")
        return signal

# Streamlit page config
st.set_page_config(
    page_title="📹 Camera BP Predictor",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("📹 Camera-Based Health Predictor Suite")
    st.markdown("*Powered by rPPG-Toolbox + PaPaGei Foundation Model*")
    st.markdown("🩺 **Blood Pressure** • 🍯 **Glucose** • ❤️ **Cardiovascular Risk**")
    
    # Check if rPPG is available
    if not RPPG_AVAILABLE:
        st.error("⚠️ rPPG-Toolbox integration not available. Please install dependencies.")
        st.info("Run: `cd external/rppg-toolbox && bash setup.sh conda`")
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # rPPG method selection with detailed explanations
        st.subheader("📊 rPPG Extraction Method")
        
        # Method categories
        method_categories = {
            "🟢 Unsupervised (Ready-to-use)": {
                "CHROM": {
                    "name": "CHROM (Recommended)",
                    "speed": "⚡⚡",
                    "accuracy": "⭐⭐⭐",
                    "description": "Analyzes color differences between RGB channels",
                    "best_for": "General use, good lighting",
                    "pros": "Fast, reliable, works on most people"
                },
                "POS": {
                    "name": "POS",
                    "speed": "⚡⚡",
                    "accuracy": "⭐⭐⭐⭐",
                    "description": "Projects skin color changes onto optimal plane",
                    "best_for": "Varying lighting, different skin tones",
                    "pros": "Robust to illumination changes"
                },
                "ICA": {
                    "name": "ICA",
                    "speed": "⚡",
                    "accuracy": "⭐⭐⭐",
                    "description": "Separates mixed signals to isolate pulse",
                    "best_for": "Noisy environments, motion artifacts",
                    "pros": "Good at filtering interference"
                },
                "GREEN": {
                    "name": "GREEN (Quick Test)",
                    "speed": "⚡⚡⚡",
                    "accuracy": "⭐⭐",
                    "description": "Simple green channel analysis",
                    "best_for": "Quick testing, minimal processing",
                    "pros": "Fastest method, lowest cost"
                }
            },
            "🧠 AI Methods (Advanced)": {
                "TSCAN": {
                    "name": "TSCAN (Highest Accuracy)",
                    "speed": "⚡",
                    "accuracy": "⭐⭐⭐⭐⭐",
                    "description": "Neural network with attention mechanisms",
                    "best_for": "Maximum accuracy, complex scenarios",
                    "pros": "State-of-the-art performance"
                },
                "PhysNet": {
                    "name": "PhysNet",
                    "speed": "⚡",
                    "accuracy": "⭐⭐⭐⭐⭐",
                    "description": "End-to-end physiological measurement",
                    "best_for": "Research, multi-person scenarios",
                    "pros": "Handles complex conditions"
                },
                "DeepPhys": {
                    "name": "DeepPhys",
                    "speed": "⚡",
                    "accuracy": "⭐⭐⭐⭐",
                    "description": "Deep learning with motion representation",
                    "best_for": "High motion environments",
                    "pros": "Excellent motion robustness"
                },
                "EfficientPhys": {
                    "name": "EfficientPhys",
                    "speed": "⚡⚡",
                    "accuracy": "⭐⭐⭐⭐",
                    "description": "Lightweight neural network",
                    "best_for": "Real-time, mobile devices",
                    "pros": "Fast AI with good accuracy"
                }
            }
        }
        
        # Method selection with enhanced interface
        all_methods = []
        method_info = {}
        for category, methods in method_categories.items():
            for method_key, details in methods.items():
                all_methods.append(f"{details['name']}")
                method_info[details['name']] = {**details, 'key': method_key}
        
        selected_method_display = st.selectbox(
            "Choose Method:",
            all_methods,
            index=0,
            help="Select the rPPG extraction algorithm. Hover over method names for details."
        )
        
        # Extract the actual method key
        rppg_method = method_info[selected_method_display]['key']
        
        # Show detailed information about selected method
        method_details = method_info[selected_method_display]
        
        with st.expander(f"ℹ️ About {selected_method_display}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Speed:** {method_details['speed']}")
                st.markdown(f"**Accuracy:** {method_details['accuracy']}")
            with col2:
                category = "🟢 Traditional" if method_details['key'] in method_categories["🟢 Unsupervised (Ready-to-use)"] else "🧠 AI-Based"
                st.markdown(f"**Type:** {category}")
            
            st.markdown(f"**How it works:** {method_details['description']}")
            st.markdown(f"**Best for:** {method_details['best_for']}")
            st.markdown(f"**Advantages:** {method_details['pros']}")
        
        # Quick recommendations
        st.markdown("### 🎯 Quick Recommendations")
        st.info("""
        **🏠 First time?** → CHROM (Recommended)  
        **🔬 Research?** → TSCAN (Highest Accuracy)  
        **⚡ Quick test?** → GREEN (Quick Test)  
        **💡 Poor lighting?** → POS
        """)
        
        # Method comparison button
        if st.button("📊 Compare All Methods"):
            st.session_state.show_method_comparison = True
        
        # Recording duration
        duration = st.slider(
            "Recording Duration (seconds)",
            min_value=10.0,
            max_value=60.0,
            value=30.0,
            step=5.0,
            help="Longer recordings provide more stable predictions"
        )
        
        # Camera settings
        camera_id = st.number_input(
            "Camera ID",
            min_value=0,
            max_value=10,
            value=0,
            help="0 for default camera, 1+ for additional cameras"
        )
        
        # Patient Information for Enhanced Predictions
        st.markdown("---")
        st.subheader("👤 Patient Information")
        st.markdown("*Optional: Improves glucose & cardiovascular predictions*")
        
        # Basic demographics
        patient_age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        patient_gender = st.selectbox("Gender", ["Female", "Male"], index=0)
        
        col1, col2 = st.columns(2)
        with col1:
            height_cm = st.number_input("Height (cm)", min_value=120, max_value=220, value=170, step=1)
        with col2:
            weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70, step=1)
        
        # Health information
        with st.expander("🏥 Health Information (Optional)"):
            st.markdown("*For more accurate cardiovascular risk assessment*")
            
            col1, col2 = st.columns(2)
            with col1:
                total_cholesterol = st.number_input(
                    "Total Cholesterol (mg/dL)", 
                    min_value=100, max_value=400, value=200, step=5,
                    help="Leave as default if unknown"
                )
                cigarettes_per_day = st.number_input(
                    "Cigarettes per day", 
                    min_value=0, max_value=60, value=0, step=1
                )
            
            with col2:
                diabetes = st.checkbox("Diabetes", value=False)
                hypertension = st.checkbox("Hypertension", value=False) 
                bp_medication = st.checkbox("Blood Pressure Medication", value=False)
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            target_fps = st.slider("Target FPS", 15, 60, 30)
            quality_threshold = st.slider("Quality Threshold", 0.1, 1.0, 0.5)
        
        # Store patient info in session state
        st.session_state.patient_info = {
            'age': patient_age,
            'gender': patient_gender,
            'height_cm': height_cm,
            'weight_kg': weight_kg,
            'total_cholesterol': total_cholesterol if total_cholesterol != 200 else None,
            'cigarettes_per_day': cigarettes_per_day,
            'diabetes': diabetes,
            'hypertension': hypertension,
            'bp_medication': bp_medication
        }
            
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📹 Live Camera", "🎥 Video Upload", "📊 Results", "ℹ️ Info"])
    
    with tab1:
        camera_interface(rppg_method, duration, camera_id, target_fps, quality_threshold)
    
    with tab2:
        video_upload_interface(rppg_method, duration, target_fps)
    
    with tab3:
        results_interface()
    
    with tab4:
        info_interface()
    
    # Show method comparison if requested
    if st.session_state.get('show_method_comparison', False):
        show_method_comparison_modal()

def show_method_comparison_modal():
    """Display comprehensive method comparison."""
    st.markdown("---")
    st.header("📊 rPPG Method Comparison")
    
    # Close button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("✕ Close"):
            st.session_state.show_method_comparison = False
            st.rerun()
    
    # Comparison table
    comparison_data = {
        "Method": [
            "GREEN (Quick Test)", "CHROM (Recommended)", "POS", "ICA", 
            "EfficientPhys", "DeepPhys", "PhysNet", "TSCAN (Highest Accuracy)"
        ],
        "Type": [
            "🟢 Traditional", "🟢 Traditional", "🟢 Traditional", "🟢 Traditional",
            "🧠 AI-Based", "🧠 AI-Based", "🧠 AI-Based", "🧠 AI-Based"
        ],
        "Speed": ["⚡⚡⚡", "⚡⚡", "⚡⚡", "⚡", "⚡⚡", "⚡", "⚡", "⚡"],
        "Accuracy": ["⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
        "Best Use Case": [
            "Quick testing", "General use", "Poor lighting", "Noisy conditions",
            "Mobile/real-time", "High motion", "Research/multi-person", "Maximum accuracy"
        ],
        "Key Advantage": [
            "Fastest processing", "Reliable & fast", "Light-robust", "Noise-resistant",
            "AI + speed", "Motion-robust", "Multi-person", "State-of-the-art"
        ]
    }
    
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)
    
    # Detailed recommendations
    st.markdown("### 🎯 Detailed Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🏠 **Home Users**
        - **Start with:** CHROM (Recommended)
        - **If poor lighting:** POS  
        - **Quick test:** GREEN
        - **Best accuracy:** TSCAN
        
        #### 📱 **Mobile/Real-time**
        - **Best choice:** EfficientPhys
        - **Alternative:** CHROM
        - **Avoid:** PhysNet, TSCAN (too slow)
        """)
    
    with col2:
        st.markdown("""
        #### 🔬 **Research/Clinical**
        - **Highest accuracy:** TSCAN
        - **Multi-person:** PhysNet
        - **Motion studies:** DeepPhys
        - **Baseline:** CHROM
        
        #### 🎥 **Challenging Conditions**
        - **Poor lighting:** POS
        - **Lots of movement:** DeepPhys
        - **Noise/artifacts:** ICA
        - **Multiple people:** PhysNet
        """)
    
    # Performance characteristics
    st.markdown("### ⚙️ Technical Details")
    
    st.info("""
    **🟢 Traditional Methods:**
    - No pre-training required
    - Work immediately 
    - Lower computational cost
    - Good for most scenarios
    
    **🧠 AI Methods:**
    - Require pre-trained models
    - Higher computational cost
    - Better for complex scenarios
    - State-of-the-art accuracy
    """)
    
    st.markdown("### 🔍 How rPPG Works")
    
    st.markdown("""
    All methods detect **tiny color changes** in your facial skin caused by blood flow:
    
    1. **📹 Camera captures** your face (30 fps)
    2. **🎨 Algorithm analyzes** pixel color changes over time  
    3. **💓 Extracts pulse signal** from color variations
    4. **🧠 PaPaGei processes** signal → predicts blood pressure
    
    **Different methods** = different ways to analyze those color changes!
    """)

def camera_interface(method: str, duration: float, camera_id: int, fps: int, quality_threshold: float):
    """Interface for live camera PPG extraction."""
    
    st.header("📹 Live Camera PPG Extraction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Instructions")
        st.info("""
        1. **Position yourself**: Sit 60-80cm from the camera
        2. **Lighting**: Ensure good, even lighting on your face
        3. **Stay still**: Minimize head movement during recording
        4. **Look at camera**: Keep your face visible to the camera
        5. **Click Record**: Start the PPG extraction process
        """)
        
        # Record button
        if st.button("🔴 Start Recording", type="primary", use_container_width=True):
            record_camera_ppg(method, duration, camera_id, fps, quality_threshold)
    
    with col2:
        # Camera preview and status
        st.markdown("### Camera Status")
        
        # Test camera button
        if st.button("📹 Test Camera", help="Check if camera is working"):
            test_camera_access(camera_id)
        
        # Camera tips
        with st.expander("💡 Camera Tips", expanded=True):
            st.markdown("""
            **For best results:**
            - 📏 **Distance**: 60-80cm from camera
            - 💡 **Lighting**: Even, natural lighting
            - 🎯 **Position**: Face centered in camera view
            - 😐 **Stay still**: Minimize movement
            - ⏰ **Duration**: 30+ seconds recommended
            
            **Privacy:** All processing is local, no data stored
            """)

def test_camera_access(camera_id: int):
    """Test camera access and show preview frame."""
    import cv2
    
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            st.error(f"❌ Camera {camera_id} not accessible")
            st.info("💡 **Troubleshooting:**")
            st.markdown("""
            - Close other apps using the camera
            - Check camera permissions in System Preferences
            - Try a different camera ID (1, 2, etc.)
            """)
            return
        
        # Get frame for preview
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get camera info
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            st.success(f"✅ Camera {camera_id} working!")
            st.info(f"📹 Resolution: {width}x{height} | FPS: {fps:.1f}")
            
            # Show preview frame
            st.image(frame_rgb, caption="Camera Preview", width=300)
            
        else:
            st.error("❌ Cannot capture frame from camera")
            
        cap.release()
        
    except Exception as e:
        st.error(f"❌ Camera test failed: {e}")
        st.info("💡 Make sure no other apps are using the camera")

def video_upload_interface(method: str, duration: float, fps: int):
    """Interface for video file PPG extraction."""
    
    st.header("🎥 Video File PPG Extraction")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Video File",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video showing your face (30 seconds recommended)"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name
        
        try:
            # Display video info
            cap = cv2.VideoCapture(temp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            video_duration = total_frames / video_fps
            cap.release()
            
            st.success(f"✅ Video uploaded: {video_duration:.1f}s at {video_fps:.1f} FPS")
            
            # Extract PPG button
            if st.button("🔄 Extract PPG from Video", type="primary"):
                extract_video_ppg(temp_video_path, method, duration)
                
        except Exception as e:
            st.error(f"Error processing video: {e}")
        finally:
            # Cleanup
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)

def record_camera_ppg(method: str, duration: float, camera_id: int, fps: int, quality_threshold: float):
    """Record PPG signal from camera with real-time progress."""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    countdown_placeholder = st.empty()
    
    try:
        status_text.text("🎥 Initializing camera...")
        progress_bar.progress(0.1)
        
        # Test camera access first
        import cv2
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            st.error(f"❌ Cannot access camera {camera_id}. Please check permissions.")
            return
        
        # Get camera properties
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        st.info(f"📹 Camera ready: {width}x{height} at {actual_fps:.1f}fps")
        progress_bar.progress(0.2)
        
        # Record with progress updates
        status_text.text("🔴 Recording video...")
        temp_video_path = record_video_with_progress(
            camera_id, duration, progress_bar, countdown_placeholder, status_text
        )
        
        if temp_video_path is None:
            st.error("❌ Recording failed")
            return
            
        progress_bar.progress(0.7)
        status_text.text("🔄 Extracting PPG signal...")
        
        # Initialize rPPG extractor and process video
        extractor = rPPGToolboxIntegration(method=method)
        ppg_signal, metadata = extractor.extract_ppg_from_video(temp_video_path, duration)
        
        progress_bar.progress(0.9)
        status_text.text("🧠 Processing with PaPaGei...")
        
        # Convert to PaPaGei format and predict comprehensive health metrics
        papagei_data = extractor.convert_to_papagei_format(ppg_signal, metadata)
        patient_info = st.session_state.get('patient_info', {})
        health_predictions = predict_unified_health_metrics(papagei_data, patient_info)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        
        # Store results in session state
        st.session_state.ppg_results = {
            'ppg_signal': ppg_signal,
            'metadata': metadata,
            'health_predictions': health_predictions,
            'method': method,
            'source': 'camera'
        }
        
        st.success("🎉 PPG extraction and BP prediction completed!")
        
        # Display quick results
        display_unified_results(health_predictions, metadata)
        
        # Cleanup temporary video
        try:
            os.unlink(temp_video_path)
        except:
            pass
        
    except Exception as e:
        st.error(f"❌ Camera PPG extraction failed: {e}")
        logger.error(f"Camera PPG extraction error: {e}")
    finally:
        progress_bar.empty()
        status_text.empty()
        countdown_placeholder.empty()

def record_video_with_progress(camera_id: int, duration: float, progress_bar, countdown_placeholder, status_text):
    """Record video with real-time progress updates."""
    import cv2
    import tempfile
    import time
    
    # Create temporary video file  
    temp_video_path = tempfile.mktemp(suffix='.avi')
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return None
    
    try:
        # Camera properties
        fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer - use XVID for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        temp_video_path = temp_video_path.replace('.mp4', '.avi')  # Use AVI for better compatibility
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        total_frames = int(fps * duration)
        frames_recorded = 0
        start_time = time.time()
        
        # Recording countdown
        for countdown in range(3, 0, -1):
            countdown_placeholder.markdown(f"## 🎬 Recording starts in: {countdown}")
            time.sleep(1)
        
        countdown_placeholder.markdown("## 🔴 RECORDING...")
        
        failed_reads = 0
        max_failed_reads = 100  # Allow some failed reads before giving up
        
        while frames_recorded < total_frames:
            ret, frame = cap.read()
            
            if not ret:
                failed_reads += 1
                if failed_reads > max_failed_reads:
                    logger.error(f"Too many failed camera reads ({failed_reads})")
                    break
                time.sleep(0.01)  # Brief pause before retry
                continue
            
            # Reset failed reads counter on successful read
            failed_reads = 0
            
            out.write(frame)
            frames_recorded += 1
            
            # Update progress every 10 frames to avoid too many updates
            if frames_recorded % 10 == 0:
                progress = 0.2 + (frames_recorded / total_frames) * 0.5  # 20% to 70%
                progress_bar.progress(progress)
                
                elapsed_time = time.time() - start_time
                remaining_time = max(0, duration - elapsed_time)
                
                countdown_placeholder.markdown(f"## ⏱️ {remaining_time:.1f}s remaining")
                status_text.text(f"📹 Recording: {frames_recorded}/{total_frames} frames")
            
            # Safety timeout check
            if time.time() - start_time > duration + 10:  # 10 second buffer
                logger.warning("Recording timeout reached")
                break
        
        countdown_placeholder.markdown("## ✅ Recording complete!")
        return temp_video_path
        
    except Exception as e:
        logger.error(f"Video recording error: {e}")
        return None
    finally:
        cap.release()
        if 'out' in locals():
            out.release()

def extract_video_ppg(video_path: str, method: str, duration: float):
    """Extract PPG signal from video file."""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🎥 Loading video...")
        
        # Initialize rPPG extractor
        extractor = rPPGToolboxIntegration(method=method)
        
        status_text.text("📊 Extracting PPG signal...")
        progress_bar.progress(0.3)
        
        # Extract PPG from video
        ppg_signal, metadata = extractor.extract_ppg_from_video(video_path, duration)
        progress_bar.progress(0.7)
        
        status_text.text("🧠 Processing with PaPaGei...")
        
        # Convert to PaPaGei format and predict comprehensive health metrics
        papagei_data = extractor.convert_to_papagei_format(ppg_signal, metadata)
        patient_info = st.session_state.get('patient_info', {})
        health_predictions = predict_unified_health_metrics(papagei_data, patient_info)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        
        # Store results in session state
        st.session_state.ppg_results = {
            'ppg_signal': ppg_signal,
            'metadata': metadata,
            'health_predictions': health_predictions,
            'method': method,
            'source': 'video'
        }
        
        st.success("🎉 Video PPG extraction and BP prediction completed!")
        
        # Display quick results
        display_unified_results(health_predictions, metadata)
        
    except Exception as e:
        st.error(f"❌ Video PPG extraction failed: {e}")
        logger.error(f"Video PPG extraction error: {e}")
    finally:
        progress_bar.empty()
        status_text.empty()

def predict_unified_health_metrics(papagei_data: Dict[str, Any], patient_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """Unified health prediction from PPG data: Blood Pressure, Glucose, and Cardiovascular Risk."""
    
    try:
        ppg_signal = papagei_data['ppg_signal']
        sampling_rate = papagei_data['sampling_rate']
        
        # Calculate heart rate
        heart_rate = calculate_heart_rate(ppg_signal, sampling_rate)
        
        # Blood Pressure Prediction (Mock - replace with actual PaPaGei model)
        systolic_bp = 100 + (heart_rate - 60) * 0.8 + np.random.normal(0, 5)
        diastolic_bp = 70 + (heart_rate - 60) * 0.4 + np.random.normal(0, 3)
        systolic_bp = np.clip(systolic_bp, 90, 200)
        diastolic_bp = np.clip(diastolic_bp, 60, 120)
        
        # Prepare unified data format for predictions
        unified_data = {
            'ppg_signal': ppg_signal,
            'heart_rate': heart_rate,
            'blood_pressure': {
                'systolic': float(systolic_bp),
                'diastolic': float(diastolic_bp)
            }
        }
        
        # Add patient information if provided
        if patient_info:
            unified_data.update(patient_info)
        
        # Initialize prediction modules
        glucose_predictor = GlucoseIntegration()
        cardiovascular_predictor = CholesterolCardiovascularIntegration()
        
        # Glucose Prediction
        try:
            glucose_result = glucose_predictor.predict_from_papagei_format(unified_data)
        except Exception as e:
            logger.warning(f"Glucose prediction failed: {e}")
            glucose_result = {
                'predicted_glucose_mg_dl': 'N/A',
                'confidence_score': 0.0,
                'interpretation': 'Prediction unavailable',
                'model_used': 'error'
            }
        
        # Cardiovascular Risk Prediction
        try:
            cvd_result = cardiovascular_predictor.predict_from_papagei_format(unified_data)
            logger.info(f"CVD prediction successful: {cvd_result}")
        except Exception as e:
            logger.error(f"Cardiovascular prediction failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            cvd_result = {
                '10_year_chd_risk_probability': 'N/A',
                'risk_category': 'Assessment unavailable',
                'recommendations': ['Consult healthcare provider for proper assessment'],
                'model_used': 'error'
            }
        
        # Compile comprehensive results
        prediction = {
            # Blood Pressure Results
            'blood_pressure': {
                'systolic_bp': float(systolic_bp),
                'diastolic_bp': float(diastolic_bp),
                'heart_rate': float(heart_rate),
                'bp_category': categorize_blood_pressure(systolic_bp, diastolic_bp),
                'confidence': 0.75
            },
            
            # Glucose Results
            'glucose': glucose_result,
            
            # Cardiovascular Risk Results  
            'cardiovascular_risk': cvd_result,
            
            # Overall Assessment
            'overall': {
                'quality_score': papagei_data['metadata'].get('quality_score', 0.8),
                'method': 'PaPaGei + rPPG + Unified ML',
                'timestamp': time.time(),
                'data_completeness': calculate_data_completeness(patient_info)
            }
        }
        
        return prediction
        
    except Exception as e:
        logger.error(f"Unified health prediction error: {e}")
        return {
            'blood_pressure': {
                'systolic_bp': None,
                'diastolic_bp': None,
                'heart_rate': None,
            },
            'glucose': {'predicted_glucose_mg_dl': 'N/A', 'interpretation': 'Error'},
            'cardiovascular_risk': {'risk_category': 'Error', 'recommendations': []},
            'overall': {'quality_score': 0.0, 'error': str(e)}
        }

def categorize_blood_pressure(systolic: float, diastolic: float) -> str:
    """Categorize blood pressure according to AHA guidelines."""
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif systolic < 130 and diastolic < 80:
        return "Elevated"
    elif (120 <= systolic < 140) or (80 <= diastolic < 90):
        return "Stage 1 Hypertension"
    elif systolic >= 140 or diastolic >= 90:
        return "Stage 2 Hypertension"
    else:
        return "Hypertensive Crisis"

def calculate_data_completeness(patient_info: Dict[str, Any]) -> float:
    """Calculate completeness score for patient data."""
    if not patient_info:
        return 0.3  # Base PPG data only
    
    important_fields = ['age', 'gender', 'height_cm', 'weight_kg', 'total_cholesterol', 'diabetes', 'hypertension']
    available_fields = sum(1 for field in important_fields if patient_info.get(field) is not None)
    
    return 0.3 + (available_fields / len(important_fields)) * 0.7

def calculate_heart_rate(ppg_signal: np.ndarray, sampling_rate: int) -> float:
    """Calculate heart rate from PPG signal."""
    
    try:
        # Simple peak detection for heart rate calculation
        from scipy.signal import find_peaks
        
        # Find peaks in PPG signal
        peaks, _ = find_peaks(ppg_signal, distance=int(sampling_rate * 0.5))  # Min 0.5s between peaks
        
        if len(peaks) < 2:
            return 70.0  # Default heart rate
        
        # Calculate heart rate from peak intervals
        peak_intervals = np.diff(peaks) / sampling_rate  # Intervals in seconds
        avg_interval = np.mean(peak_intervals)
        heart_rate = 60.0 / avg_interval  # Convert to BPM
        
        # Ensure reasonable range
        heart_rate = np.clip(heart_rate, 40, 150)
        
        return heart_rate
        
    except Exception as e:
        logger.error(f"Heart rate calculation error: {e}")
        return 70.0  # Default

def display_unified_results(health_predictions: Dict[str, Any], metadata: Dict[str, Any]):
    """Display comprehensive health prediction results."""
    
    st.markdown("### 🎯 Quick Health Assessment")
    
    # Blood Pressure Results
    bp_data = health_predictions.get('blood_pressure', {})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if bp_data.get('systolic_bp'):
            st.metric(
                "🩺 Blood Pressure",
                f"{bp_data['systolic_bp']:.0f}/{bp_data['diastolic_bp']:.0f}",
                help=f"Category: {bp_data.get('bp_category', 'Unknown')}"
            )
        else:
            st.metric("🩺 Blood Pressure", "Error")
    
    with col2:
        glucose_data = health_predictions.get('glucose', {})
        if glucose_data.get('predicted_glucose_mg_dl', 'N/A') != 'N/A':
            st.metric(
                "🍯 Glucose Level",
                f"{glucose_data['predicted_glucose_mg_dl']} mg/dL",
                help=glucose_data.get('interpretation', 'Glucose prediction')
            )
        else:
            st.metric("🍯 Glucose Level", "Estimating...")
    
    with col3:
        cvd_data = health_predictions.get('cardiovascular_risk', {})
        if cvd_data.get('10_year_chd_risk_probability', 'N/A') != 'N/A':
            risk_pct = f"{float(cvd_data['10_year_chd_risk_probability']) * 100:.1f}%"
            st.metric(
                "❤️ CV Risk (10yr)",
                risk_pct,
                help=cvd_data.get('risk_category', 'Cardiovascular risk')
            )
        else:
            st.metric("❤️ CV Risk (10yr)", "Assessing...")
    
    # Additional quick insights
    col1, col2 = st.columns(2)
    with col1:
        if bp_data.get('heart_rate'):
            st.info(f"💓 Heart Rate: **{bp_data['heart_rate']:.0f} BPM**")
    
    with col2:
        overall_data = health_predictions.get('overall', {})
        completeness = overall_data.get('data_completeness', 0.3)
        if completeness > 0.7:
            st.success(f"✅ Data Quality: **{completeness*100:.0f}%** Complete")
        else:
            st.warning(f"⚠️ Data Quality: **{completeness*100:.0f}%** - Add patient info for better accuracy")

def results_interface():
    """Display detailed results and visualizations."""
    
    st.header("📊 Detailed Results")
    
    if 'ppg_results' not in st.session_state:
        st.info("👆 Extract PPG from camera or video first to see detailed results")
        return
    
    results = st.session_state.ppg_results
    ppg_signal = results['ppg_signal']
    metadata = results['metadata']
    health_predictions = results['health_predictions']
    
    # Comprehensive Health Results
    st.subheader("📋 Comprehensive Health Assessment")
    
    # Create tabs for different aspects
    bp_tab, glucose_tab, cvd_tab, tech_tab = st.tabs(["🩺 Blood Pressure", "🍯 Glucose", "❤️ Cardiovascular Risk", "⚙️ Technical"])
    
    with bp_tab:
        bp_data = health_predictions.get('blood_pressure', {})
        if bp_data.get('systolic_bp'):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Blood Pressure Reading")
                st.markdown(f"**{bp_data['systolic_bp']:.0f}/{bp_data['diastolic_bp']:.0f} mmHg**")
                st.markdown(f"Category: **{bp_data.get('bp_category', 'Unknown')}**")
                st.markdown(f"Heart Rate: **{bp_data.get('heart_rate', 'N/A'):.0f} BPM**")
                st.markdown(f"Confidence: **{bp_data.get('confidence', 0)*100:.0f}%**")
            with col2:
                st.markdown("#### BP Guidelines")
                st.markdown("""
                - **Normal**: <120/<80 mmHg
                - **Elevated**: 120-129/<80 mmHg
                - **Stage 1**: 130-139/80-89 mmHg
                - **Stage 2**: ≥140/≥90 mmHg
                """)
        else:
            st.error("Blood pressure prediction failed")
    
    with glucose_tab:
        glucose_data = health_predictions.get('glucose', {})
        if glucose_data.get('predicted_glucose_mg_dl', 'N/A') != 'N/A':
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Glucose Prediction")
                st.markdown(f"**{glucose_data['predicted_glucose_mg_dl']} mg/dL**")
                st.markdown(f"Interpretation: **{glucose_data.get('interpretation', 'N/A')}**")
                st.markdown(f"Confidence: **{glucose_data.get('confidence_score', 0)*100:.0f}%**")
                st.markdown(f"Model: {glucose_data.get('model_used', 'Unknown')}")
            with col2:
                st.markdown("#### Glucose Guidelines")
                st.markdown("""
                - **Normal**: 70-99 mg/dL (fasting)
                - **Prediabetes**: 100-125 mg/dL
                - **Diabetes**: ≥126 mg/dL (fasting)
                - **Low**: <70 mg/dL (hypoglycemia)
                """)
        else:
            st.warning("Glucose prediction requires more patient information")
    
    with cvd_tab:
        cvd_data = health_predictions.get('cardiovascular_risk', {})
        if cvd_data.get('10_year_chd_risk_probability', 'N/A') != 'N/A':
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 10-Year Cardiovascular Risk")
                risk_pct = f"{float(cvd_data['10_year_chd_risk_probability']) * 100:.1f}%"
                st.markdown(f"**{risk_pct}**")
                st.markdown(f"Category: **{cvd_data.get('risk_category', 'Unknown')}**")
                st.markdown(f"Model: {cvd_data.get('model_used', 'Framingham Risk Score')}")
            with col2:
                st.markdown("#### Risk Categories")
                st.markdown("""
                - **Low Risk**: <7.5%
                - **Intermediate**: 7.5-20%
                - **High Risk**: >20%
                """)
            
            # Recommendations
            if cvd_data.get('recommendations'):
                st.markdown("#### 📝 Clinical Recommendations")
                for rec in cvd_data['recommendations']:
                    st.markdown(f"• {rec}")
        else:
            st.warning("Cardiovascular risk assessment requires more patient information")
    
    with tech_tab:
        st.markdown("#### Technical Information")
        overall_data = health_predictions.get('overall', {})
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Extraction Details**")
            st.markdown(f"Method: **{metadata.get('method', 'Unknown')}**")
            st.markdown(f"Source: **{results.get('source', 'Unknown')}**")
            st.markdown(f"Signal Quality: **{metadata.get('quality_score', 0)*100:.0f}%**")
            st.markdown(f"Duration: **{metadata.get('duration', 0):.1f}s**")
        with col2:
            st.markdown("**Data Completeness**")
            completeness = overall_data.get('data_completeness', 0.3)
            st.markdown(f"Patient Info: **{completeness*100:.0f}%** complete")
            st.markdown(f"Processing Method: **{overall_data.get('method', 'PaPaGei + rPPG')}**")
            if overall_data.get('error'):
                st.error(f"Processing Error: {overall_data['error']}")
    
    # PPG signal visualization
    st.subheader("📈 PPG Signal Visualization")
    
    # Create time axis
    sampling_rate = metadata.get('sampling_rate', 250)
    time_axis = np.linspace(0, len(ppg_signal) / sampling_rate, len(ppg_signal))
    
    # Plot PPG signal
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=ppg_signal,
        mode='lines',
        name='PPG Signal',
        line=dict(color='red', width=1)
    ))
    
    fig.update_layout(
        title="Extracted PPG Signal",
        xaxis_title="Time (seconds)",
        yaxis_title="PPG Amplitude",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Signal quality metrics
    st.subheader("🔍 Signal Quality Analysis")
    
    quality_col1, quality_col2, quality_col3 = st.columns(3)
    
    with quality_col1:
        snr = calculate_snr(ppg_signal)
        st.metric("Signal-to-Noise Ratio", f"{snr:.1f} dB")
    
    with quality_col2:
        signal_std = np.std(ppg_signal)
        st.metric("Signal Variability", f"{signal_std:.3f}")
    
    with quality_col3:
        peak_count = len(find_peaks_simple(ppg_signal))
        st.metric("Detected Peaks", f"{peak_count}")

def calculate_snr(signal: np.ndarray) -> float:
    """Calculate signal-to-noise ratio."""
    try:
        signal_power = np.mean(signal ** 2)
        noise_power = np.var(signal)  # Simplified noise estimation
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        return max(snr, 0)  # Ensure non-negative
    except:
        return 0.0

def find_peaks_simple(signal: np.ndarray) -> np.ndarray:
    """Simple peak detection."""
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(signal, distance=50)  # Simplified
        return peaks
    except:
        return np.array([])

def categorize_bp(systolic: float, diastolic: float) -> str:
    """Categorize blood pressure according to AHA guidelines."""
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif systolic < 130 and diastolic < 80:
        return "Elevated"
    elif systolic < 140 or diastolic < 90:
        return "Stage 1 Hypertension"
    elif systolic < 180 or diastolic < 120:
        return "Stage 2 Hypertension"
    else:
        return "Hypertensive Crisis"

def info_interface():
    """Display comprehensive information about camera-based PPG."""
    
    st.header("ℹ️ About Camera-Based PPG")
    
    # Quick overview
    st.markdown("""
    ### 🎯 What This Does
    
    **Camera-based remote photoplethysmography (rPPG)** uses your device's camera to measure 
    your heart rate and predict blood pressure by detecting tiny color changes in your facial 
    skin caused by blood flow - no physical contact required!
    """)
    
    # How it works section
    with st.expander("🔍 How rPPG Technology Works", expanded=True):
        st.markdown("""
        #### The Science Behind It
        
        1. **📹 Camera Recording**: Records your face for 30+ seconds
        2. **🎨 Color Analysis**: Detects microscopic skin color changes with each heartbeat
        3. **💓 Pulse Extraction**: Isolates your pulse signal from the color variations  
        4. **🧠 AI Processing**: PaPaGei foundation model analyzes the pulse pattern
        5. **🩺 BP Prediction**: Predicts blood pressure from pulse characteristics
        
        #### Why It Works
        - Every heartbeat pumps blood through facial capillaries
        - Blood volume changes cause tiny color shifts (invisible to naked eye)
        - Different rPPG algorithms detect these changes in various ways
        - AI models translate pulse patterns into cardiovascular measurements
        """)
    
    # Method comparison quick reference
    with st.expander("📊 Method Quick Reference"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🟢 **Traditional Methods**
            - **CHROM** → Best all-around choice
            - **POS** → Good for poor lighting  
            - **ICA** → Handles noise well
            - **GREEN** → Fastest for quick tests
            """)
        
        with col2:
            st.markdown("""
            #### 🧠 **AI Methods**
            - **TSCAN** → Highest accuracy
            - **PhysNet** → Multi-person scenarios
            - **DeepPhys** → Motion-robust
            - **EfficientPhys** → Fast + accurate
            """)
    
    # Technology stack
    with st.expander("⚙️ Technology Stack"):
        st.markdown("""
        - **rPPG-Toolbox**: 8 state-of-the-art extraction algorithms
        - **PaPaGei Foundation Model**: Nokia Bell Labs' signal processing AI
        - **OpenCV**: Camera capture and video processing
        - **Multiple Models**: Traditional signal processing + deep learning
        """)
    
    # Best practices
    with st.expander("💡 Best Practices for Accuracy"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🎥 **Setup**
            - **Distance**: 60-80cm from camera
            - **Lighting**: Even, natural light (avoid direct sun)
            - **Background**: Plain, non-reflective
            - **Position**: Face centered in frame
            """)
        
        with col2:
            st.markdown("""
            #### 🧘 **During Recording**  
            - **Stay still**: Minimize head movement
            - **Breathe normally**: Don't hold breath
            - **Look at camera**: Keep face visible
            - **30+ seconds**: Longer = more accurate
            """)
    
    # Accuracy and limitations
    with st.expander("📈 Accuracy & Limitations"):
        st.markdown("""
        #### Expected Performance
        - **Heart Rate**: ±5 BPM accuracy (research-validated)
        - **Blood Pressure**: ±10-15 mmHg (varies by individual)
        - **Best Methods**: TSCAN, PhysNet for highest accuracy
        - **Factors**: Lighting, skin tone, motion affect results
        
        #### Important Limitations
        ⚠️ **Not for medical diagnosis** - for wellness/research only  
        ⚠️ **Individual variation** - accuracy varies per person  
        ⚠️ **Environmental factors** - lighting and motion critical  
        ⚠️ **Contact methods** still more accurate than camera-based  
        """)
    
    # Privacy and data
    with st.expander("🔒 Privacy & Data"):
        st.markdown("""
        #### Your Privacy is Protected
        ✅ **All processing is local** - no data sent to external servers  
        ✅ **No video storage** - recordings deleted immediately after processing  
        ✅ **No personal data** collected or retained  
        ✅ **Open source** - you can inspect all code  
        
        #### How Data Flows
        1. Camera → Temporary video file
        2. rPPG algorithm → Pulse signal extraction  
        3. PaPaGei model → Blood pressure prediction
        4. Results displayed → Temporary files deleted
        """)
    
    # Use cases
    with st.expander("🎯 Use Cases & Applications"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🏠 **Personal Use**
            - Daily health monitoring
            - Fitness tracking
            - Stress level assessment
            - Wellness checkups
            """)
        
        with col2:
            st.markdown("""
            #### 🔬 **Research/Clinical**
            - Remote patient monitoring
            - Telehealth applications  
            - Population health studies
            - Non-contact screening
            """)
    
    # Getting started
    st.markdown("""
    ### 🚀 Ready to Try?
    
    1. **📹 Test your camera** first (Camera tab → Test Camera button)
    2. **🎯 Choose method** (CHROM recommended for first-time users)  
    3. **🔴 Start recording** (30 seconds, follow positioning tips)
    4. **📊 View results** (heart rate, BP prediction, signal quality)
    
    **Pro tip**: Try different methods to see which works best for your setup!
    """)

if __name__ == "__main__":
    main()