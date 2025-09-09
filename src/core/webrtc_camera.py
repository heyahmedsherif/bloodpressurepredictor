"""
WebRTC Camera Integration for Streamlit Cloud
===========================================

This module provides real camera access for Streamlit Cloud deployments using streamlit-webrtc.
Unlike traditional OpenCV VideoCapture, this works in cloud environments by using WebRTC.

Author: Claude Code Integration
Date: 2025-09-09
"""

import streamlit as st
import numpy as np
import cv2
import time
import queue
import threading
from typing import Optional, Tuple, Dict, Any, List
import logging
from dataclasses import dataclass

# WebRTC imports
try:
    from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError as e:
    logging.warning(f"streamlit-webrtc not available: {e}")
    WEBRTC_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PPGFrameData:
    """Container for PPG-relevant frame data"""
    frame: np.ndarray
    timestamp: float
    roi_detected: bool
    face_confidence: float

class PPGVideoProcessor(VideoTransformerBase):
    """
    Real-time video processor for PPG extraction using WebRTC
    """
    
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=1000)
        self.roi_frames = []
        self.face_cascade = None
        self.recording = False
        self.start_time = None
        self.target_duration = 30.0
        
        # Load face detection
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            logger.warning(f"Could not load face cascade: {e}")
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process incoming video frames"""
        # Convert to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        if self.recording and self.start_time:
            elapsed = time.time() - self.start_time
            
            # Stop recording after target duration
            if elapsed >= self.target_duration:
                self.recording = False
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Detect face ROI for PPG
            roi_detected, confidence = self._detect_face_roi(img)
            
            # Store frame data
            frame_data = PPGFrameData(
                frame=img.copy(),
                timestamp=time.time(),
                roi_detected=roi_detected,
                face_confidence=confidence
            )
            
            # Add to queue (non-blocking)
            try:
                self.frame_queue.put_nowait(frame_data)
            except queue.Full:
                pass  # Skip frame if queue is full
            
            # Visual feedback
            if roi_detected:
                img = self._draw_roi_indicators(img)
            else:
                img = self._draw_no_face_warning(img)
            
            # Show recording progress
            progress = elapsed / self.target_duration
            img = self._draw_progress_bar(img, progress)
        
        else:
            # Preview mode - just show face detection
            roi_detected, confidence = self._detect_face_roi(img)
            if roi_detected:
                img = self._draw_roi_indicators(img)
            else:
                img = self._draw_preview_instructions(img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def start_recording(self, duration: float = 30.0):
        """Start PPG recording session"""
        self.recording = True
        self.start_time = time.time()
        self.target_duration = duration
        self.roi_frames = []
        
        # Clear existing frames
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info(f"Started PPG recording for {duration} seconds")
    
    def stop_recording(self):
        """Stop PPG recording session"""
        self.recording = False
        logger.info("Stopped PPG recording")
    
    def get_recorded_frames(self) -> List[PPGFrameData]:
        """Get all recorded frames"""
        frames = []
        while not self.frame_queue.empty():
            try:
                frames.append(self.frame_queue.get_nowait())
            except queue.Empty:
                break
        return frames
    
    def _detect_face_roi(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Detect face region of interest for PPG"""
        if self.face_cascade is None:
            return False, 0.0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=4,
            minSize=(100, 100)
        )
        
        if len(faces) > 0:
            # Use the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            return True, 0.8  # Dummy confidence
        
        return False, 0.0
    
    def _draw_roi_indicators(self, img: np.ndarray) -> np.ndarray:
        """Draw ROI indicators on frame"""
        if self.face_cascade is None:
            return img
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw PPG ROI (forehead region)
            roi_x = x + w//4
            roi_y = y + h//6
            roi_w = w//2
            roi_h = h//4
            cv2.rectangle(img, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 255), 2)
            
            # Add labels
            cv2.putText(img, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, "PPG ROI", (roi_x, roi_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return img
    
    def _draw_no_face_warning(self, img: np.ndarray) -> np.ndarray:
        """Draw no face detection warning"""
        h, w = img.shape[:2]
        cv2.putText(img, "⚠️ NO FACE DETECTED", (w//2-150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(img, "Position face in center", (w//2-120, h//2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img
    
    def _draw_preview_instructions(self, img: np.ndarray) -> np.ndarray:
        """Draw preview mode instructions"""
        h, w = img.shape[:2]
        cv2.putText(img, "📹 CAMERA PREVIEW", (w//2-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, "Position your face in view", (w//2-140, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img
    
    def _draw_progress_bar(self, img: np.ndarray, progress: float) -> np.ndarray:
        """Draw recording progress bar"""
        h, w = img.shape[:2]
        bar_width = 300
        bar_height = 20
        bar_x = (w - bar_width) // 2
        bar_y = h - 50
        
        # Background
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Progress
        progress_width = int(bar_width * min(progress, 1.0))
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
        
        # Text
        cv2.putText(img, f"🔴 RECORDING: {progress*100:.0f}%", (bar_x, bar_y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img

def extract_ppg_from_webrtc_frames(frames: List[PPGFrameData]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract PPG signal from WebRTC recorded frames
    """
    if not frames:
        logger.warning("No frames available for PPG extraction")
        return _generate_synthetic_ppg(30.0), {'method': 'synthetic_no_frames'}
    
    # Filter frames with detected faces
    face_frames = [f for f in frames if f.roi_detected]
    
    if len(face_frames) < 10:
        logger.warning(f"Insufficient face frames: {len(face_frames)}")
        return _generate_synthetic_ppg(30.0), {'method': 'synthetic_insufficient_frames'}
    
    logger.info(f"Processing {len(face_frames)} frames with face detection")
    
    # Extract ROI signals (simplified approach)
    ppg_values = []
    
    for frame_data in face_frames:
        # Extract forehead ROI
        roi_signal = _extract_roi_signal(frame_data.frame)
        ppg_values.append(roi_signal)
    
    # Convert to numpy array
    ppg_signal = np.array(ppg_values)
    
    # Basic preprocessing
    ppg_signal = _preprocess_ppg_signal(ppg_signal)
    
    # Calculate metadata
    duration = len(ppg_signal) / 30.0  # Assuming 30 FPS
    heart_rate = _estimate_heart_rate(ppg_signal, 30.0)
    
    metadata = {
        'method': 'webrtc_camera',
        'frames_processed': len(face_frames),
        'total_frames': len(frames),
        'face_detection_ratio': len(face_frames) / len(frames),
        'heart_rate': heart_rate,
        'duration': duration,
        'sampling_rate': 30.0,
        'quality_score': min(len(face_frames) / len(frames), 1.0)
    }
    
    return ppg_signal, metadata

def _extract_roi_signal(frame: np.ndarray) -> float:
    """Extract PPG signal from frame ROI"""
    # Detect face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(100, 100))
    
    if len(faces) == 0:
        return 0.0
    
    # Use largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    
    # Extract forehead ROI
    roi_x = x + w//4
    roi_y = y + h//6
    roi_w = w//2
    roi_h = h//4
    
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    if roi.size == 0:
        return 0.0
    
    # Calculate mean green channel (PPG signal)
    green_channel = roi[:, :, 1]  # Green channel for PPG
    return np.mean(green_channel)

def _preprocess_ppg_signal(signal: np.ndarray) -> np.ndarray:
    """Basic PPG signal preprocessing"""
    if len(signal) == 0:
        return signal
    
    # Remove DC component
    signal = signal - np.mean(signal)
    
    # Simple detrending
    if len(signal) > 1:
        trend = np.linspace(signal[0], signal[-1], len(signal))
        signal = signal - trend
    
    return signal

def _estimate_heart_rate(signal: np.ndarray, sampling_rate: float) -> float:
    """Estimate heart rate from PPG signal using FFT"""
    if len(signal) < 10:
        return 75.0  # Default
    
    # Simple FFT-based HR estimation
    fft = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)
    
    # Focus on physiological range (0.5-4 Hz = 30-240 BPM)
    valid_idx = (freqs >= 0.5) & (freqs <= 4.0)
    if not np.any(valid_idx):
        return 75.0
    
    peak_freq = freqs[valid_idx][np.argmax(fft[valid_idx])]
    heart_rate = peak_freq * 60  # Convert to BPM
    
    # Clamp to reasonable range
    return np.clip(heart_rate, 50, 120)

def _generate_synthetic_ppg(duration: float, fs: float = 30.0) -> np.ndarray:
    """Generate synthetic PPG signal for fallback"""
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

def create_webrtc_camera_interface(duration: float = 30.0) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Create WebRTC camera interface for PPG recording
    """
    if not WEBRTC_AVAILABLE:
        st.error("🚫 WebRTC not available. Please install streamlit-webrtc")
        return None, {'error': 'webrtc_unavailable'}
    
    # WebRTC configuration for cloud deployment
    rtc_configuration = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    })
    
    st.info("📹 **Real Camera Access Enabled!** This uses WebRTC technology that works on Streamlit Cloud.")
    
    # Initialize session state
    if 'webrtc_processor' not in st.session_state:
        st.session_state.webrtc_processor = None
    
    # Camera controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🎬 Start Recording", type="primary"):
            st.session_state.recording_requested = True
    
    with col2:
        if st.button("⏹️ Stop Recording"):
            st.session_state.recording_requested = False
            if st.session_state.webrtc_processor:
                st.session_state.webrtc_processor.stop_recording()
    
    with col3:
        show_preview = st.checkbox("📺 Show Preview", value=True)
    
    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="ppg-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=PPGVideoProcessor,
        media_stream_constraints={
            "video": {"width": 640, "height": 480, "frameRate": 30},
            "audio": False
        },
        async_processing=True
    )
    
    # Handle recording
    if webrtc_ctx.video_processor and hasattr(st.session_state, 'recording_requested'):
        processor = webrtc_ctx.video_processor
        st.session_state.webrtc_processor = processor
        
        if st.session_state.recording_requested:
            processor.start_recording(duration)
            st.session_state.recording_requested = False
            
            # Show recording progress
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Wait for recording to complete
            start_time = time.time()
            while processor.recording:
                elapsed = time.time() - start_time
                progress = elapsed / duration
                
                progress_placeholder.progress(min(progress, 1.0))
                status_placeholder.info(f"🔴 Recording... {elapsed:.1f}s / {duration:.1f}s")
                
                if elapsed >= duration:
                    break
                
                time.sleep(0.1)
            
            # Get recorded frames
            frames = processor.get_recorded_frames()
            
            if frames:
                status_placeholder.success(f"✅ Recording complete! Processed {len(frames)} frames")
                
                # Extract PPG
                with st.spinner("🔬 Extracting PPG signal..."):
                    ppg_signal, metadata = extract_ppg_from_webrtc_frames(frames)
                
                return ppg_signal, metadata
            else:
                status_placeholder.error("❌ No frames recorded. Please ensure camera access is allowed.")
    
    return None, {'status': 'preview_mode'}

# Convenience function for integration
def extract_ppg_from_webrtc_camera(duration: float = 30.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract PPG signal from WebRTC camera (works on Streamlit Cloud)
    """
    return create_webrtc_camera_interface(duration)