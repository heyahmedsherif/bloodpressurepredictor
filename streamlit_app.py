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
            st.error(f"Camera Health Predictor not available: {e}")
            st.error("Please install rPPG-Toolbox dependencies: cd external/rppg-toolbox && bash setup.sh conda")
            
    elif app_choice == "Realistic BP Predictor (Recommended)":
        st.markdown("---")
        st.info("🎯 **Most Accurate**: Uses only clinically available features for real-world deployment")
        # Import and run the realistic BP predictor
        try:
            from src.apps.realistic_bp_predictor import main as realistic_main
            realistic_main()
        except ImportError as e:
            st.error(f"Import error: {e}")
            st.error("Please ensure all dependencies are installed")
            
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