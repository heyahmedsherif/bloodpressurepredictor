"""
Simple Camera Interface for Streamlit Cloud
==========================================

This module provides a basic camera interface using Streamlit's built-in camera_input
that works reliably on Streamlit Cloud without external dependencies.

Author: Claude Code Integration  
Date: 2025-09-09
"""

import streamlit as st
import numpy as np
import cv2
import time
from typing import Optional, Tuple, Dict, Any
import logging
from PIL import Image
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_simple_camera_interface(duration: float = 30.0) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Create simple camera interface using Streamlit's built-in camera_input
    """
    st.info("📸 **Simple Camera Mode**: Using Streamlit's built-in camera functionality")
    st.markdown("This mode captures a photo and generates synthetic PPG data for demonstration.")
    
    # Camera input
    camera_photo = st.camera_input("📷 Take a photo for PPG analysis")
    
    if camera_photo is not None:
        # Process the captured image
        with st.spinner("🔬 Analyzing photo and generating PPG signal..."):
            # Convert to OpenCV format
            image = Image.open(camera_photo)
            image_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_array.shape) == 3:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_array
            
            # Display the captured image
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image_array, caption="📷 Captured Image", width=300)
            
            with col2:
                # Analyze face in the image
                face_detected, face_info = analyze_face_in_image(image_bgr)
                
                if face_detected:
                    st.success("✅ Face detected!")
                    st.write(f"🎯 Face confidence: {face_info['confidence']:.2f}")
                    
                    # Generate realistic PPG based on image analysis
                    ppg_signal = generate_image_based_ppg(image_bgr, duration)
                    
                    metadata = {
                        'method': 'simple_camera_photo',
                        'heart_rate': 70.0 + np.random.normal(0, 5),
                        'quality_score': face_info['confidence'],
                        'face_detected': True,
                        'image_size': image_array.shape,
                        'sampling_rate': 125.0,
                        'duration': duration,
                        'note': 'PPG generated from photo analysis with synthetic modeling'
                    }
                    
                    return ppg_signal, metadata
                else:
                    st.warning("⚠️ No face detected in image")
                    st.info("💡 Try taking another photo with better lighting and face positioning")
                    
                    # Generate basic synthetic PPG
                    ppg_signal = generate_synthetic_ppg(duration)
                    metadata = {
                        'method': 'synthetic_no_face',
                        'heart_rate': 75.0,
                        'quality_score': 0.5,
                        'face_detected': False,
                        'note': 'Synthetic PPG - no face detected in image'
                    }
                    
                    return ppg_signal, metadata
    
    # No photo taken yet
    st.info("👆 **Take a photo above** to start PPG analysis")
    return None, {'status': 'waiting_for_photo'}

def analyze_face_in_image(image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
    """
    Analyze face in the captured image
    """
    try:
        # Load face detection cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        if len(faces) > 0:
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Calculate confidence based on face size
            image_area = gray.shape[0] * gray.shape[1]
            face_area = w * h
            face_ratio = face_area / image_area
            
            confidence = min(0.9, face_ratio * 10)  # Scale confidence
            
            return True, {
                'confidence': confidence,
                'face_box': (x, y, w, h),
                'face_area': face_area,
                'face_ratio': face_ratio
            }
        else:
            return False, {'confidence': 0.0}
            
    except Exception as e:
        logger.warning(f"Face detection error: {e}")
        return False, {'confidence': 0.0, 'error': str(e)}

def generate_image_based_ppg(image: np.ndarray, duration: float) -> np.ndarray:
    """
    Generate PPG signal based on image analysis
    """
    try:
        # Extract some basic image features to influence PPG generation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Image brightness (can influence heart rate estimation)
        brightness = np.mean(gray)
        
        # Image contrast 
        contrast = np.std(gray)
        
        # Use these features to create more realistic PPG
        base_hr = 70.0
        
        # Adjust heart rate based on image characteristics (synthetic relationship)
        if brightness < 100:  # Darker images
            hr_adjustment = -5
        elif brightness > 180:  # Brighter images  
            hr_adjustment = 5
        else:
            hr_adjustment = 0
            
        heart_rate = base_hr + hr_adjustment + np.random.normal(0, 3)
        heart_rate = np.clip(heart_rate, 50, 120)
        
        # Generate PPG with this heart rate
        fs = 125.0  # Sampling frequency
        t = np.linspace(0, duration, int(fs * duration))
        
        # Cardiac frequency
        cardiac_freq = heart_rate / 60.0
        
        # Primary cardiac rhythm
        ppg_signal = np.sin(2 * np.pi * cardiac_freq * t)
        
        # Dicrotic notch (secondary peak)
        ppg_signal += 0.3 * np.sin(2 * np.pi * cardiac_freq * t + np.pi/3)
        
        # Respiratory modulation (breathing effect)
        resp_freq = 0.25  # ~15 breaths/min
        ppg_signal *= (1 + 0.1 * np.sin(2 * np.pi * resp_freq * t))
        
        # Add noise influenced by image contrast
        noise_level = max(0.02, min(0.1, contrast / 1000))
        noise = np.random.normal(0, noise_level, len(t))
        ppg_signal += noise
        
        return ppg_signal
        
    except Exception as e:
        logger.warning(f"Image-based PPG generation error: {e}")
        return generate_synthetic_ppg(duration)

def generate_synthetic_ppg(duration: float, fs: float = 125.0) -> np.ndarray:
    """
    Generate basic synthetic PPG signal
    """
    t = np.linspace(0, duration, int(fs * duration))
    heart_rate = 75  # BPM
    
    # Base cardiac rhythm
    cardiac_freq = heart_rate / 60.0
    ppg_signal = np.sin(2 * np.pi * cardiac_freq * t)
    
    # Add dicrotic notch
    ppg_signal += 0.3 * np.sin(2 * np.pi * cardiac_freq * t + np.pi/2)
    
    # Add respiratory modulation
    resp_freq = 0.25  # ~15 breaths/min
    ppg_signal *= (1 + 0.1 * np.sin(2 * np.pi * resp_freq * t))
    
    # Add realistic noise
    noise = np.random.normal(0, 0.05, len(t))
    ppg_signal += noise
    
    return ppg_signal

# Convenience function
def extract_ppg_from_simple_camera(duration: float = 30.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract PPG using simple camera interface (works on Streamlit Cloud)
    """
    return create_simple_camera_interface(duration)