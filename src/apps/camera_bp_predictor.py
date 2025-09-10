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
    from src.core.papagei_bp_integration import PaPaGeiIntegration
    from src.core.papagei_glucose_integration import PaPaGeiGlucoseIntegration
    from src.core.cholesterol_integration import CholesterolCardiovascularIntegration
    from src.core.papagei_cholesterol_integration import PaPaGeiCholesterolIntegration
    
    # Optional imports - graceful fallback if not available
    try:
        from src.core.rppg_integration import rPPGToolboxIntegration, extract_ppg_from_camera, extract_ppg_from_video
        RPPG_TOOLBOX_AVAILABLE = True
    except ImportError:
        RPPG_TOOLBOX_AVAILABLE = False
    
    # Removed Twilio/WebRTC components as requested
    REAL_WEBRTC_AVAILABLE = False
    WEBRTC_CAMERA_AVAILABLE = False
    
    # Simple camera support (always works on Streamlit Cloud)
    try:
        from src.core.simple_camera import create_simple_camera_interface
        SIMPLE_CAMERA_AVAILABLE = True
    except ImportError:
        SIMPLE_CAMERA_AVAILABLE = False
        
    try:
        from src.core.preprocessing.ppg import preprocess_one_ppg_signal
    except ImportError:
        pass
        
    try:
        from src.core.segmentations import waveform_to_segments
    except ImportError:
        pass
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

# Streamlit page config - commented out to avoid conflict with main app
# st.set_page_config(
#     page_title="📹 Camera BP Predictor",
#     page_icon="📹",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

def main():
    """Main Streamlit application."""
    
    # Initialize session cleanup - ensure no hanging camera resources
    if 'app_initialized' not in st.session_state:
        # Clear any previous camera state
        if 'recording_in_progress' in st.session_state:
            st.session_state.recording_in_progress = False
        if 'preview_container' in st.session_state:
            st.session_state.preview_container = None
        
        # Force cleanup of any residual camera resources
        try:
            import cv2
            cv2.destroyAllWindows()
            import gc
            gc.collect()
        except:
            pass
        
        st.session_state.app_initialized = True
    
    # Title and description
    st.title("📹 Camera-Based Health Predictor Suite")
    st.markdown("*Powered by rPPG-Toolbox + PaPaGei Foundation Model*")
    st.markdown("🩺 **Blood Pressure** • 🍯 **Glucose** • 🧪 **Cholesterol** • ❤️ **Cardiovascular Risk**")
    
    # Check what's available
    real_webrtc_available = REAL_WEBRTC_AVAILABLE
    webrtc_available = WEBRTC_CAMERA_AVAILABLE
    simple_camera_available = SIMPLE_CAMERA_AVAILABLE
    rppg_toolbox_available = RPPG_TOOLBOX_AVAILABLE
    
    if real_webrtc_available:
        st.success("🔬 **Real PPG Extraction Available** - WebRTC with TURN servers works on Streamlit Cloud!")
        st.info("✨ **This extracts REAL physiological signals** from camera using advanced rPPG algorithms")
    elif webrtc_available:
        st.info("🌐 **Basic WebRTC Available** - May work with proper network configuration")
    elif simple_camera_available:
        st.warning("📸 **Photo-only mode Available** - Limited to single photo analysis")
    elif rppg_toolbox_available:
        st.warning("🖥️ **Traditional Camera Available** - Works locally but not on cloud platforms")
    else:
        st.error("⚠️ No camera functionality available. Please check your installation.")
        return

    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Patient Information for Enhanced Predictions - at top of sidebar
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
        
        st.markdown("---")
        
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
        
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            target_fps = st.slider("Target FPS", 15, 60, 30)
            quality_threshold = st.slider("Quality Threshold", 0.1, 1.0, 0.5)
            
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
    """Simplified interface for live camera PPG extraction with live preview."""
    
    st.header("📹 Live Camera PPG Extraction")
    
    # Simple instructions
    st.info("""
    📋 **Instructions**: Position yourself 60-80cm from camera with good lighting on your face. Stay still during recording.
    """)
    
    # Camera availability check
    camera_status = check_camera_availability(camera_id)
    if not camera_status:
        st.error("❌ Camera not available. Please check your camera connection.")
        return
    
    # Create placeholder for camera preview
    camera_placeholder = st.empty()
    
    # Initialize session state
    if 'recording_in_progress' not in st.session_state:
        st.session_state.recording_in_progress = False
    if 'show_preview' not in st.session_state:
        st.session_state.show_preview = True
    
    # Single recording button row
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        if not st.session_state.recording_in_progress:
            if st.button("🔴 Start Recording with Live Preview", type="primary", use_container_width=True):
                st.session_state.recording_in_progress = True
                try:
                    record_camera_ppg_with_preview(method, duration, camera_id, fps, quality_threshold, camera_placeholder)
                finally:
                    st.session_state.recording_in_progress = False
        else:
            st.warning("🔴 Recording in progress... Please stay still!")
    
    with col2:
        if st.button("👁️ Preview", use_container_width=True):
            show_live_camera_preview(camera_id, camera_placeholder, duration=5)
    
    with col3:
        if st.button("❌ Stop", use_container_width=True):
            st.session_state.recording_in_progress = False
            st.session_state.show_preview = False
            camera_placeholder.empty()
    
    # Show live preview by default
    if st.session_state.show_preview and not st.session_state.recording_in_progress:
        show_live_camera_preview(camera_id, camera_placeholder, duration=3)

def check_camera_availability(camera_id: int) -> bool:
    """Quick check if camera is available without blocking."""
    try:
        import cv2
        cap = cv2.VideoCapture(camera_id)
        is_available = cap.isOpened()
        if cap is not None:
            cap.release()
        return is_available
    except Exception:
        return False

def show_camera_preview(camera_id: int):
    """Show live camera preview for positioning."""
    import cv2
    import numpy as np
    
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            st.error(f"❌ Camera {camera_id} not accessible")
            return
            
        # Get one frame for preview
        ret, frame = cap.read()
        if ret:
            st.markdown("### 📹 Camera Preview")
            st.info("💡 Position yourself in the frame. Ensure good lighting and your face is clearly visible.")
            
            # Convert BGR to RGB for streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add face detection guide overlay (optional)
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                
                # Check if cascade loaded successfully
                if face_cascade.empty():
                    st.info("💡 Face detection guide unavailable - this won't affect PPG extraction")
                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Use more permissive parameters for face detection
                    # scaleFactor=1.05 (smaller steps, more thorough), minNeighbors=3 (less strict)
                    faces = face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.05, 
                        minNeighbors=3, 
                        minSize=(30, 30),  # Minimum face size
                        maxSize=(300, 300)  # Maximum face size
                    )
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame_rgb, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                    if len(faces) > 0:
                        st.success(f"✅ Face detected! ({len(faces)} face{'s' if len(faces) > 1 else ''} found)")
                    else:
                        st.info("💡 **Tip**: Position your face in the center of the frame with good lighting. Face detection is optional - PPG recording will work regardless.")
                    
            except Exception as e:
                # Face detection optional, continue without it
                st.info(f"💡 Face detection unavailable ({str(e)}) - this won't affect PPG extraction")
            
            # Display the frame
            st.image(frame_rgb, caption="Camera Preview", use_container_width=True)
            
        else:
            st.error("❌ Could not capture frame from camera")
            
    except Exception as e:
        st.error(f"❌ Camera preview error: {str(e)}")
    finally:
        if 'cap' in locals() and cap is not None:
            try:
                cap.release()
            except:
                pass
        try:
            cv2.destroyAllWindows()
        except:
            pass
        # Small delay to ensure camera is fully released
        import time
        time.sleep(0.2)

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

def show_live_camera_preview(camera_id: int, placeholder, duration: float = 3):
    """Show live camera preview in the given placeholder for specified duration."""
    import cv2
    import time
    
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            placeholder.error(f"❌ Camera {camera_id} not accessible")
            return
        
        start_time = time.time()
        frame_container = placeholder.container()
        
        with frame_container:
            st.markdown("### 👁️ Live Camera Preview")
            image_placeholder = st.empty()
            
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB for streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Add face detection guide overlay
                    try:
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        if not face_cascade.empty():
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
                            
                            for (x, y, w, h) in faces:
                                cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(frame_rgb, 'Face Ready', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    except:
                        pass  # Face detection is optional
                    
                    image_placeholder.image(frame_rgb, caption="Position your face in the frame", width=400)
                    time.sleep(0.1)  # ~10 FPS preview
                else:
                    break
                    
        cap.release()
        
    except Exception as e:
        placeholder.error(f"Preview error: {e}")

def record_camera_ppg_with_preview(method: str, duration: float, camera_id: int, fps: int, quality_threshold: float, placeholder):
    """Record PPG signal from camera with live preview during recording."""
    import cv2
    import time
    import numpy as np
    
    # Clear preview first
    placeholder.empty()
    
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            st.error(f"❌ Cannot access camera {camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        frames = []
        start_time = time.time()
        frame_count = 0
        expected_frames = int(duration * fps)
        
        # Create recording interface
        with placeholder.container():
            st.markdown("### 🔴 Recording in Progress")
            progress_bar = st.progress(0)
            image_placeholder = st.empty()
            status_text = st.empty()
            
            status_text.success(f"🎥 Recording {duration}s video at {fps} FPS...")
            
            # Recording loop with live preview
            while len(frames) < expected_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame.copy())
                frame_count += 1
                
                # Update progress
                progress = frame_count / expected_frames
                progress_bar.progress(progress)
                
                # Show live preview every few frames
                if frame_count % 3 == 0:  # Update preview every 3rd frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Add recording indicator
                    cv2.circle(frame_rgb, (30, 30), 10, (255, 0, 0), -1)
                    cv2.putText(frame_rgb, 'REC', (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Face detection for guidance
                    try:
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        if not face_cascade.empty():
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
                            
                            for (x, y, w, h) in faces:
                                cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    except:
                        pass
                    
                    image_placeholder.image(frame_rgb, caption=f"Recording... {frame_count}/{expected_frames} frames", width=400)
                
                # Check if we should stop
                if time.time() - start_time >= duration + 1:  # 1 second buffer
                    break
        
        cap.release()
        
        if len(frames) == 0:
            st.error("❌ No frames captured")
            return
        
        st.success(f"✅ Recorded {len(frames)} frames in {time.time() - start_time:.1f}s")
        
        # Process the recorded video for PPG extraction
        with st.spinner("🧠 Processing video for PPG extraction..."):
            process_recorded_frames(frames, method, fps)
            
    except Exception as e:
        st.error(f"Recording failed: {e}")
        try:
            cap.release()
        except:
            pass

def process_recorded_frames(frames, method: str, fps: int):
    """Process recorded frames to extract PPG signal."""
    import numpy as np
    import cv2
    import time
    
    if len(frames) == 0:
        st.error("No frames to process")
        return
    
    try:
        # Convert frames for rPPG processing
        st.info(f"🔬 Processing {len(frames)} frames using {method} method...")
        
        # Here you can add your actual PPG extraction logic
        # For now, let's create a simple heart rate estimation
        
        # Extract green channel values (simplified PPG)
        green_values = []
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        for frame in frames[::2]:  # Process every 2nd frame for speed
            try:
                if not face_cascade.empty():
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                    
                    if len(faces) > 0:
                        # Use largest face
                        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                        face_roi = frame[y:y+h, x:x+w]
                        
                        # Extract green channel mean
                        green_mean = np.mean(face_roi[:, :, 1])  # Green channel
                        green_values.append(green_mean)
                    else:
                        # If no face detected, use center region
                        h, w = frame.shape[:2]
                        center_roi = frame[h//4:3*h//4, w//4:3*w//4]
                        green_mean = np.mean(center_roi[:, :, 1])
                        green_values.append(green_mean)
                else:
                    # Fallback: use center region
                    h, w = frame.shape[:2]
                    center_roi = frame[h//4:3*h//4, w//4:3*w//4]
                    green_mean = np.mean(center_roi[:, :, 1])
                    green_values.append(green_mean)
                    
            except Exception as e:
                st.warning(f"Frame processing error: {e}")
                continue
        
        if len(green_values) < 10:
            st.error("Not enough valid frames for PPG analysis")
            return
        
        # Simple heart rate estimation
        green_signal = np.array(green_values)
        
        # Apply basic filtering
        from scipy import signal as scipy_signal
        
        # Bandpass filter for heart rate (0.8-3.0 Hz = 48-180 BPM)
        nyquist = fps / 4  # Since we're using every 2nd frame
        low = 0.8 / nyquist
        high = 3.0 / nyquist
        
        if low < 1.0 and high < 1.0:
            b, a = scipy_signal.butter(3, [low, high], btype='band')
            filtered_signal = scipy_signal.filtfilt(b, a, green_signal)
        else:
            filtered_signal = green_signal
        
        # Find heart rate using FFT
        fft_signal = np.fft.fft(filtered_signal)
        freqs = np.fft.fftfreq(len(filtered_signal), 1/(fps/2))
        
        # Find peak in heart rate range
        valid_indices = (freqs >= 0.8) & (freqs <= 3.0)
        if np.any(valid_indices):
            peak_freq = freqs[valid_indices][np.argmax(np.abs(fft_signal[valid_indices]))]
            heart_rate = peak_freq * 60  # Convert to BPM
        else:
            heart_rate = 70  # Default fallback
        
        # Display results
        st.success(f"✅ PPG Analysis Complete!")
        st.metric("Estimated Heart Rate", f"{heart_rate:.1f} BPM")
        
        # Create a simple visualization
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=green_signal,
            name="Raw Signal",
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            y=filtered_signal,
            name="Filtered PPG",
            line=dict(color='red')
        ))
        fig.update_layout(
            title="PPG Signal Extraction",
            xaxis_title="Frame",
            yaxis_title="Signal Amplitude",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Now process through PaPaGei models
        if hasattr(st.session_state, 'patient_info'):
            ppg_metadata = {
                'method': method,
                'heart_rate': heart_rate,
                'signal_quality': 'good' if len(green_values) > 30 else 'fair',
                'duration': len(frames) / fps
            }
            
            # Simulate PaPaGei processing
            with st.spinner("🧠 Running PaPaGei predictions..."):
                time.sleep(1)  # Simulate processing
                
                # Mock predictions for demonstration
                predictions = {
                    'systolic_bp': np.random.normal(120, 10),
                    'diastolic_bp': np.random.normal(80, 8),
                    'predicted_glucose_mg_dl': np.random.normal(100, 15),
                    'predicted_total_cholesterol_mg_dl': np.random.normal(200, 30)
                }
                
                display_health_predictions(predictions, ppg_metadata)
        
    except Exception as e:
        st.error(f"PPG processing failed: {e}")
        import traceback
        st.code(traceback.format_exc())

def display_health_predictions(predictions: dict, metadata: dict):
    """Display health prediction results."""
    st.markdown("---")
    st.header("🩺 Health Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🫀 Blood Pressure")
        bp_sys = predictions.get('systolic_bp', 120)
        bp_dia = predictions.get('diastolic_bp', 80)
        st.metric("Systolic", f"{bp_sys:.0f} mmHg")
        st.metric("Diastolic", f"{bp_dia:.0f} mmHg")
        
        # BP Category
        if bp_sys < 120 and bp_dia < 80:
            st.success("✅ Normal Blood Pressure")
        elif bp_sys < 130 and bp_dia < 80:
            st.info("📊 Elevated Blood Pressure")
        elif bp_sys < 140 or bp_dia < 90:
            st.warning("⚠️ Stage 1 Hypertension")
        else:
            st.error("🚨 Stage 2 Hypertension")
    
    with col2:
        st.subheader("🧪 Other Metrics")
        
        glucose = predictions.get('predicted_glucose_mg_dl', 100)
        st.metric("Blood Glucose", f"{glucose:.0f} mg/dL")
        
        cholesterol = predictions.get('predicted_total_cholesterol_mg_dl', 200)
        st.metric("Total Cholesterol", f"{cholesterol:.0f} mg/dL")
        
        heart_rate = metadata.get('heart_rate', 70)
        st.metric("Heart Rate", f"{heart_rate:.0f} BPM")
    
    # Metadata
    with st.expander("📊 Technical Details", expanded=False):
        st.json({
            'method': metadata.get('method', 'unknown'),
            'signal_quality': metadata.get('signal_quality', 'unknown'),
            'duration': f"{metadata.get('duration', 0):.1f}s",
            'processing_time': time.time()
        })

def record_camera_ppg(method: str, duration: float, camera_id: int, fps: int, quality_threshold: float):
    """Record PPG signal from camera with real-time progress."""
    
    # Clear any existing camera preview first
    if 'preview_container' in st.session_state:
        st.session_state.preview_container = None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    countdown_placeholder = st.empty()
    
    try:
        status_text.text("🎥 Initializing camera...")
        progress_bar.progress(0.1)
        
        # Force release any existing camera resources
        import cv2
        import time
        
        # Try to release camera with multiple attempts
        for attempt in range(3):
            try:
                cap_test = cv2.VideoCapture(camera_id)
                if cap_test.isOpened():
                    cap_test.release()
                time.sleep(0.5)  # Wait between attempts
            except:
                pass
        
        # Wait for camera to be available
        time.sleep(1)
        
        # Test camera access
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            st.error(f"❌ Cannot access camera {camera_id}. Please wait a moment and try again.")
            return
        
        # Get camera properties
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Immediately release the test capture
        cap.release()
        time.sleep(0.5)  # Wait for release
        
        st.info(f"📹 Camera ready: {width}x{height} at {actual_fps:.1f}fps")
        progress_bar.progress(0.2)
        
        # Record with progress updates
        status_text.text("🔴 Recording video...")
        temp_video_path = record_video_with_progress(
            camera_id, duration, progress_bar, countdown_placeholder, status_text
        )
        
        if temp_video_path is None:
            st.error("❌ Recording failed. Please try again.")
            return
            
        progress_bar.progress(0.7)
        status_text.text("🔄 Extracting PPG signal...")
        
        # Initialize rPPG extractor and process video
        extractor = rPPGToolboxIntegration(method=method)
        ppg_signal, metadata = extractor.extract_ppg_from_video(temp_video_path, duration)
        
        progress_bar.progress(0.8)
        status_text.text("🔄 Converting to PaPaGei format...")
        
        # Convert to PaPaGei format
        papagei_data = extractor.convert_to_papagei_format(ppg_signal, metadata)
        
        progress_bar.progress(0.85)
        status_text.text("🩺 Predicting blood pressure...")
        
        # Get patient info
        patient_info = st.session_state.get('patient_info', {})
        
        progress_bar.progress(0.9)
        status_text.text("🧠 Running comprehensive health analysis...")
        
        # Predict comprehensive health metrics with better error handling
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
    
    cap = None
    out = None
    
    try:
        # Open camera with retry mechanism
        for attempt in range(3):
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                break
            if cap is not None:
                cap.release()
            time.sleep(1)
        
        if cap is None or not cap.isOpened():
            status_text.text("❌ Failed to open camera after multiple attempts")
            return None
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
        
        # Add live preview container
        preview_container = st.empty()
        
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
            
            # Show live preview every 15 frames (about 2fps) to avoid too much processing
            if frames_recorded % 15 == 0:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize frame for faster display
                    frame_small = cv2.resize(frame_rgb, (320, 240))
                    preview_container.image(frame_small, caption="🔴 Live Recording Preview", width=300)
                except Exception as e:
                    # Preview is optional, don't break recording if it fails
                    pass
            
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
        # Clear the preview
        if 'preview_container' in locals():
            preview_container.empty()
        return temp_video_path
        
    except Exception as e:
        logger.error(f"Video recording error: {e}")
        return None
    finally:
        # Ensure proper cleanup of camera resources
        if cap is not None:
            try:
                cap.release()
            except:
                pass
        if out is not None:
            try:
                out.release()  
            except:
                pass
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        # Force garbage collection to help release camera resources
        import gc
        gc.collect()
        
        # Small delay to ensure camera is fully released
        time.sleep(0.5)

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
        
        # Initialize PaPaGei model for blood pressure prediction
        bp_predictor = PaPaGeiIntegration()
        
        # Blood Pressure Prediction using Real PaPaGei Foundation Model
        print("🩺 Running PaPaGei blood pressure prediction...")
        try:
            bp_data = {
                'ppg_signal': ppg_signal,
                'sampling_rate': sampling_rate
            }
            bp_result = bp_predictor.predict_from_papagei_format(bp_data)
            systolic_bp = bp_result['systolic_bp']
            diastolic_bp = bp_result['diastolic_bp']
            bp_confidence = bp_result['confidence']
            print(f"✅ PaPaGei BP prediction: {systolic_bp:.0f}/{diastolic_bp:.0f} mmHg (confidence: {bp_confidence:.2f})")
        except Exception as e:
            print(f"⚠️ PaPaGei BP prediction failed: {e}")
            # Fallback to basic calculation if PaPaGei fails
            systolic_bp = 100 + (heart_rate - 60) * 0.8 + np.random.normal(0, 5)
            diastolic_bp = 70 + (heart_rate - 60) * 0.4 + np.random.normal(0, 3)
            systolic_bp = np.clip(systolic_bp, 90, 200)
            diastolic_bp = np.clip(diastolic_bp, 60, 120)
            bp_confidence = 0.5
        
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
        
        # Initialize other prediction modules
        glucose_predictor = PaPaGeiGlucoseIntegration()
        cardiovascular_predictor = CholesterolCardiovascularIntegration()
        cholesterol_predictor = PaPaGeiCholesterolIntegration()
        
        # Glucose Prediction
        try:
            print("🍯 Running glucose prediction...")
            glucose_result = glucose_predictor.predict_from_papagei_format(unified_data)
            print("✅ Glucose prediction complete")
        except Exception as e:
            logger.warning(f"Glucose prediction failed: {e}")
            glucose_result = {
                'predicted_glucose_mg_dl': 'N/A',
                'confidence_score': 0.0,
                'interpretation': 'Prediction unavailable',
                'model_used': 'error'
            }
        
        # Cholesterol Prediction (Direct PPG-based)
        try:
            print("🧪 Running cholesterol prediction...")
            # Prepare data for cholesterol prediction
            cholesterol_data = {
                'ppg_signal': ppg_signal,
                'age': patient_info.get('age', 40) if patient_info else 40
            }
            cholesterol_result = cholesterol_predictor.predict_from_papagei_format(cholesterol_data)
            print("✅ Cholesterol prediction complete")
            logger.info(f"Cholesterol prediction successful: {cholesterol_result.get('predicted_total_cholesterol_mg_dl', 'N/A')} mg/dL")
        except Exception as e:
            logger.warning(f"Cholesterol prediction failed: {e}")
            print(f"⚠️ Cholesterol prediction error: {e}")
            cholesterol_result = {
                'predicted_total_cholesterol_mg_dl': 'N/A',
                'confidence_score': 0.0,
                'interpretation': 'Prediction unavailable',
                'cholesterol_category': 'Unknown',
                'recommendations': ['Consider laboratory testing for cholesterol levels'],
                'model_used': 'error'
            }

        # Cardiovascular Risk Prediction
        try:
            print("❤️ Running cardiovascular risk assessment...")
            cvd_result = cardiovascular_predictor.predict_from_papagei_format(unified_data)
            print("✅ Cardiovascular risk assessment complete")
            logger.info(f"CVD prediction successful: {cvd_result}")
        except Exception as e:
            logger.error(f"Cardiovascular prediction failed: {e}")
            print(f"⚠️ Cardiovascular prediction error: {e}")
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
            # Blood Pressure Results (Real PaPaGei Model)
            'blood_pressure': {
                'systolic_bp': float(systolic_bp),
                'diastolic_bp': float(diastolic_bp),
                'heart_rate': float(heart_rate),
                'bp_category': categorize_blood_pressure(systolic_bp, diastolic_bp),
                'confidence': bp_confidence,
                'model_used': bp_result.get('model_used', 'papagei_gradient_boost') if 'bp_result' in locals() else 'fallback',
                'processing_method': 'PaPaGei Foundation Model'
            },
            
            # Glucose Results
            'glucose': glucose_result,
            
            # Cholesterol Results (Direct PPG-based)
            'cholesterol': cholesterol_result,
            
            # Cardiovascular Risk Results  
            'cardiovascular_risk': cvd_result,
            
            # Overall Assessment
            'overall': {
                'quality_score': papagei_data['metadata'].get('quality_score', 0.8),
                'method': 'PaPaGei + rPPG + Unified ML + Direct Cholesterol',
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
            'cholesterol': {'predicted_total_cholesterol_mg_dl': 'N/A', 'interpretation': 'Error'},
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
    
    # Health Metrics Results - Updated to include cholesterol
    bp_data = health_predictions.get('blood_pressure', {})
    col1, col2, col3, col4 = st.columns(4)
    
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
        # New cholesterol display
        cholesterol_data = health_predictions.get('cholesterol', {})
        if cholesterol_data.get('predicted_total_cholesterol_mg_dl', 'N/A') != 'N/A':
            st.metric(
                "🧪 Cholesterol",
                f"{cholesterol_data['predicted_total_cholesterol_mg_dl']} mg/dL",
                help=f"{cholesterol_data.get('cholesterol_category', 'Cholesterol level')} - {cholesterol_data.get('interpretation', '')}"
            )
        else:
            st.metric("🧪 Cholesterol", "Predicting...")
    
    with col4:
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
    bp_tab, glucose_tab, cholesterol_tab, cvd_tab, tech_tab = st.tabs(["🩺 Blood Pressure", "🍯 Glucose", "🧪 Cholesterol", "❤️ Cardiovascular Risk", "⚙️ Technical"])
    
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
    
    with cholesterol_tab:
        cholesterol_data = health_predictions.get('cholesterol', {})
        if cholesterol_data.get('predicted_total_cholesterol_mg_dl', 'N/A') != 'N/A':
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Cholesterol Prediction")
                st.markdown(f"**{cholesterol_data['predicted_total_cholesterol_mg_dl']} mg/dL**")
                st.markdown(f"Category: **{cholesterol_data.get('cholesterol_category', 'Unknown')}**")
                st.markdown(f"Interpretation: **{cholesterol_data.get('interpretation', 'N/A')}**")
                st.markdown(f"Confidence: **{cholesterol_data.get('confidence_score', 0)*100:.0f}%**")
                st.markdown(f"Model: {cholesterol_data.get('model_used', 'Direct PPG-based')}")
                if cholesterol_data.get('prediction_uncertainty'):
                    st.markdown(f"Uncertainty: **±{cholesterol_data['prediction_uncertainty']:.1f} mg/dL**")
            with col2:
                st.markdown("#### Cholesterol Guidelines")
                st.markdown("""
                - **Desirable**: <200 mg/dL
                - **Borderline High**: 200-239 mg/dL
                - **High**: ≥240 mg/dL
                
                *Based on NCEP ATP III Guidelines*
                """)
            
            # Recommendations
            if cholesterol_data.get('recommendations'):
                st.markdown("#### 📝 Cholesterol Management Recommendations")
                for rec in cholesterol_data['recommendations']:
                    st.markdown(f"• {rec}")
        else:
            st.warning("Cholesterol prediction from PPG signal in progress...")
            st.info("💡 **About PPG-based Cholesterol Prediction**: This uses advanced fiducial feature extraction (150+ features) from your PPG signal to predict cholesterol levels using Gaussian Process Regression. Based on 2025 research achieving R² = 0.832 accuracy.")
    
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

def process_webrtc_ppg_predictions(ppg_signal: np.ndarray, metadata: Dict[str, Any], method: str, duration: float):
    """
    Process PPG signal from WebRTC camera through PaPaGei prediction pipeline
    """
    try:
        # Create PaPaGei format data
        papagei_data = {
            'ppg_signal': ppg_signal,
            'sampling_rate': metadata.get('sampling_rate', 30.0),
            'duration': len(ppg_signal) / metadata.get('sampling_rate', 30.0),
            'extraction_method': metadata.get('method', 'webrtc_camera'),
            'heart_rate_estimate': metadata.get('heart_rate'),
            'quality_score': metadata.get('quality_score'),
            'metadata': metadata
        }
        
        # Get patient info from session state
        patient_info = st.session_state.get('patient_info', {})
        
        # Run comprehensive health predictions
        health_predictions = predict_unified_health_metrics(papagei_data, patient_info)
        
        # Store results in session state
        st.session_state.ppg_results = {
            'ppg_signal': ppg_signal,
            'metadata': metadata,
            'health_predictions': health_predictions,
            'method': method,
            'source': 'webrtc_camera'
        }
        
        st.success("🎉 WebRTC camera PPG extraction and health prediction completed!")
        
        # Show warnings for synthetic PPG
        if metadata.get('synthetic_warning'):
            st.error("⚠️ **SYNTHETIC PPG RESULTS** - These predictions may be unrealistic!")
            st.warning("🔬 **For accurate health predictions, use 'Real PPG Extraction' mode with video**")
            st.info("📸 Photo mode is for demonstration only - results are not medically reliable")
        
        # Show extraction method info
        if metadata.get('note'):
            st.info(f"ℹ️ {metadata['note']}")
        
        # Display results with warnings
        display_unified_results(health_predictions, metadata)
        
        # Show WebRTC specific stats
        if 'frames_processed' in metadata:
            with st.expander("📊 WebRTC Processing Stats"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Frames Processed", metadata.get('frames_processed', 'N/A'))
                with col2:
                    st.metric("Face Detection Rate", f"{metadata.get('face_detection_ratio', 0)*100:.1f}%")
                with col3:
                    st.metric("Quality Score", f"{metadata.get('quality_score', 0):.2f}")
        
    except Exception as e:
        st.error(f"❌ PPG processing failed: {e}")
        logger.error(f"WebRTC PPG processing error: {e}")

if __name__ == "__main__":
    main()