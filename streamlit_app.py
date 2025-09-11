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
    
    # App selection (Camera disabled for emergency build)
    app_choice = st.selectbox(
        "Choose Application:",
        [
            "Realistic BP Predictor (Recommended)",
            "Extended BP Predictor", 
            "PPG Signal Processor"
        ]
    )
    
    if app_choice == "Realistic BP Predictor (Recommended)":
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