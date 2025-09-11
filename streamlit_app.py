# Main Streamlit App for Deployment
# This is the entry point for Streamlit Cloud deployment

import streamlit as st
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Set page config
st.set_page_config(
    page_title="Blood Pressure Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🩺 Comprehensive Health Prediction Suite")
    st.markdown("*Powered by PaPaGei Foundation Model + Advanced ML*")
    st.markdown("📹 **Camera Health Assessment** • 🩺 **Blood Pressure** • 🍯 **Glucose** • 🧪 **Cholesterol** • ❤️ **Cardiovascular Risk**")
    
    # App selection
    app_choice = st.selectbox(
        "Choose Application:",
        [
            "📹 Camera Health Predictor (NEW!)",
            "Realistic BP Predictor (Recommended)",
            "Extended BP Predictor", 
            "PPG Signal Processor"
        ]
    )
    
    if app_choice == "📹 Camera Health Predictor (NEW!)":
        st.markdown("---")
        st.info("📹 **Camera-Based Health Suite**: Extract PPG from camera + predict BP, glucose, cholesterol, cardiovascular risk")
        try:
            from src.apps.camera_bp_predictor import main as camera_main
            camera_main()
        except ImportError as e:
            st.error("🚫 **Camera import failed**")
            st.warning("🔧 **Debugging camera import issue**")
            
            # Show the actual error for debugging
            error_msg = str(e)
            st.code(f"Import error: {error_msg}")
            
            # More detailed diagnostics
            with st.expander("🔧 Detailed Import Diagnostics"):
                st.write("**Testing individual imports:**")
                
                # Test basic imports
                try:
                    import cv2
                    st.success("✅ OpenCV (cv2) - Available")
                except ImportError as cv2_err:
                    st.error(f"❌ OpenCV (cv2) - Failed: {cv2_err}")
                
                try:
                    import numpy as np
                    st.success("✅ NumPy - Available")
                except ImportError as np_err:
                    st.error(f"❌ NumPy - Failed: {np_err}")
                    
                try:
                    from streamlit_webrtc import webrtc_streamer
                    st.success("✅ Streamlit-WebRTC - Available")
                except ImportError as webrtc_err:
                    st.error(f"❌ Streamlit-WebRTC - Failed: {webrtc_err}")
                
                # Test our core imports
                try:
                    from src.core.railway_webrtc_camera import create_webrtc_ppg_interface
                    st.success("✅ WebRTC Camera Interface - Available") 
                except ImportError as core_err:
                    st.error(f"❌ WebRTC Camera Interface - Failed: {core_err}")
            
            st.info("""
            **Alternative Options:**
            - Use **Realistic BP Predictor** for immediate access (no camera required)
            - Use **Extended BP Predictor** for research features
            """)
            
            if "No module named" in error_msg:
                missing_module = error_msg.split("No module named ")[1].strip("'\"")
                st.warning(f"🔧 **Missing dependency**: `{missing_module}`")
                st.info("This dependency should be in requirements.txt. Railway may need time to install it.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh Page"):
                    st.rerun()
            with col2:
                if st.button("🎯 Use Alternative Predictor"):
                    st.session_state.app_choice = "Realistic BP Predictor (Recommended)"
                    st.rerun()
            
    elif app_choice == "Realistic BP Predictor (Recommended)":
        st.markdown("---")
        st.info("🎯 **Most Accurate**: Uses only clinically available features for real-world deployment")
        # Import and run the realistic BP predictor
        try:
            from src.apps.realistic_bp_predictor import main as realistic_main
            realistic_main()
        except ImportError as e:
            st.error("🚫 **Dependency Issue**: Some packages are still building on Streamlit Cloud")
            st.warning("⏳ **Please wait 2-3 minutes** and refresh the page - Streamlit Cloud is installing dependencies")
            st.info("""
            **If the issue persists**:
            - This is likely a Python 3.13 + pandas compilation issue
            - The app will work once all packages are installed
            - Try refreshing in a few minutes
            """)
            st.code(f"Technical details: {e}", language="python")
            
    elif app_choice == "Extended BP Predictor":
        st.markdown("---")
        st.info("🔬 **Research Mode**: Uses extended features for research and testing")
        try:
            from src.apps.bp_predictor import main as bp_main
            bp_main()
        except ImportError as e:
            st.error(f"Import error: {e}")
            
    elif app_choice == "PPG Signal Processor":
        st.markdown("---") 
        st.info("📊 **Signal Analysis**: Focus on PPG signal processing and visualization")
        try:
            from src.apps.streamlit_app_robust import main as robust_main
            robust_main()
        except ImportError as e:
            st.error(f"Import error: {e}")

if __name__ == "__main__":
    main()