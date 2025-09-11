"""
Real-time WebRTC Camera for Railway Deployment
==============================================

This module provides real-time camera access through WebRTC that works on Railway
and other web deployments. Uses streamlit-webrtc for browser-based camera access.
"""

import streamlit as st
import numpy as np
import cv2
import time
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import threading
import queue

# WebRTC imports
try:
    from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode
    import av
    WEBRTC_AVAILABLE = True
except ImportError as e:
    logging.warning(f"streamlit-webrtc not available: {e}")
    WEBRTC_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PPGResult:
    """Results from PPG extraction."""
    heart_rate: float
    ppg_signal: np.ndarray
    confidence: float
    frames_processed: int
    duration: float

class PPGVideoProcessor(VideoTransformerBase):
    """Video processor for real-time PPG extraction."""
    
    def __init__(self):
        self.frames = []
        self.ppg_values = []
        self.face_cascade = None
        self.recording = False
        self.max_frames = 900  # 30 seconds at 30 FPS
        
        # Initialize face cascade
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            logger.warning("Face cascade not available")
    
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process each video frame for PPG extraction."""
        
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        if self.recording and len(self.frames) < self.max_frames:
            # Store frame for PPG processing
            self.frames.append(img.copy())
            
            # Extract PPG signal from face region
            ppg_value = self.extract_ppg_from_frame(img)
            if ppg_value is not None:
                self.ppg_values.append(ppg_value)
        
        # Add visual feedback
        img = self.add_visual_feedback(img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def extract_ppg_from_frame(self, frame: np.ndarray) -> Optional[float]:
        """Extract PPG signal from single frame."""
        
        if self.face_cascade is None:
            # Fallback: use center region
            h, w = frame.shape[:2]
            roi = frame[h//4:3*h//4, w//4:3*w//4]
            return np.mean(roi[:, :, 1])  # Green channel
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) > 0:
            # Use largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_roi = frame[y:y+h, x:x+w]
            
            # Extract green channel mean (simple PPG)
            return np.mean(face_roi[:, :, 1])
        
        return None
    
    def add_visual_feedback(self, frame: np.ndarray) -> np.ndarray:
        """Add visual feedback to the frame."""
        
        # Add recording indicator
        if self.recording:
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, 'REC', (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f'Frames: {len(self.frames)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add face detection rectangles
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                color = (0, 255, 0) if self.recording else (255, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add instructions based on state
        if self.recording:
            cv2.putText(frame, 'Stay very still - Recording PPG signal...', (10, frame.shape[0]-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f'Progress: {len(self.frames)}/{self.max_frames} frames', (10, frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Position your face in the detection box', (10, frame.shape[0]-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, 'Click Start when ready', (10, frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def start_recording(self):
        """Start PPG recording."""
        self.recording = True
        self.frames = []
        self.ppg_values = []
        logger.info("Started PPG recording")
    
    def stop_recording(self) -> PPGResult:
        """Stop recording and process PPG signal."""
        self.recording = False
        
        if len(self.ppg_values) < 10:
            logger.warning("Not enough PPG data collected")
            return PPGResult(
                heart_rate=70.0,
                ppg_signal=np.array([]),
                confidence=0.0,
                frames_processed=len(self.frames),
                duration=0.0
            )
        
        # Process PPG signal
        ppg_signal = np.array(self.ppg_values)
        heart_rate = self.calculate_heart_rate(ppg_signal)
        
        return PPGResult(
            heart_rate=heart_rate,
            ppg_signal=ppg_signal,
            confidence=min(1.0, len(self.ppg_values) / 100),
            frames_processed=len(self.frames),
            duration=len(self.frames) / 30.0  # Assume 30 FPS
        )
    
    def calculate_heart_rate(self, ppg_signal: np.ndarray, fps: float = 30.0) -> float:
        """Calculate heart rate from PPG signal using FFT."""
        
        try:
            from scipy import signal as scipy_signal
            
            # Apply bandpass filter for heart rate (0.8-3.0 Hz = 48-180 BPM)
            nyquist = fps / 2
            low = 0.8 / nyquist
            high = 3.0 / nyquist
            
            if low < 1.0 and high < 1.0:
                b, a = scipy_signal.butter(3, [low, high], btype='band')
                filtered_signal = scipy_signal.filtfilt(b, a, ppg_signal)
            else:
                filtered_signal = ppg_signal
            
            # Find heart rate using FFT
            fft_signal = np.fft.fft(filtered_signal)
            freqs = np.fft.fftfreq(len(filtered_signal), 1/fps)
            
            # Find peak in heart rate range
            valid_indices = (freqs >= 0.8) & (freqs <= 3.0) & (freqs > 0)
            if np.any(valid_indices):
                peak_freq = freqs[valid_indices][np.argmax(np.abs(fft_signal[valid_indices]))]
                heart_rate = peak_freq * 60  # Convert to BPM
                return np.clip(heart_rate, 50, 180)
            else:
                return 70.0  # Default fallback
                
        except ImportError:
            # Fallback without scipy
            logger.warning("Scipy not available, using simple HR calculation")
            
            # Simple peak counting method
            peaks = []
            for i in range(1, len(ppg_signal)-1):
                if ppg_signal[i] > ppg_signal[i-1] and ppg_signal[i] > ppg_signal[i+1]:
                    peaks.append(i)
            
            if len(peaks) > 1:
                avg_interval = np.mean(np.diff(peaks)) / fps  # seconds
                heart_rate = 60 / avg_interval
                return np.clip(heart_rate, 50, 180)
            
            return 70.0

def create_webrtc_ppg_interface(duration: float = 30.0) -> Tuple[Optional[PPGResult], Dict[str, Any]]:
    """Create WebRTC interface for real-time PPG extraction."""
    
    if not WEBRTC_AVAILABLE:
        st.error("❌ WebRTC not available. Please check requirements.txt")
        return None, {}
    
    st.markdown("### 📹 Real-time Camera PPG Extraction")
    st.info("🎥 **Instructions**: Position your face in the green box and stay still during recording.")
    
    # Create processor instance
    if 'ppg_processor' not in st.session_state:
        st.session_state.ppg_processor = PPGVideoProcessor()
    
    processor = st.session_state.ppg_processor
    
    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="ppg-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: processor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Single recording control
    if not processor.recording:
        # Before recording - show single start button
        if st.button("🔴 Start Live Recording & PPG Analysis", type="primary", use_container_width=True, help="Click to start recording with live video and face detection"):
            processor.start_recording()
            st.rerun()
    else:
        # During recording - show progress and stop option
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.warning(f"🔴 **Recording Live PPG...** {len(processor.frames)} frames captured")
            progress = min(1.0, len(processor.frames) / processor.max_frames)
            st.progress(progress)
            
            # Auto-stop when reaching max frames
            if len(processor.frames) >= processor.max_frames:
                result = processor.stop_recording()
                st.session_state.ppg_result = result
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop & Analyze", use_container_width=True):
                result = processor.stop_recording()
                st.session_state.ppg_result = result
                st.rerun()
    
    # Show results if available
    if 'ppg_result' in st.session_state:
        result = st.session_state.ppg_result
        
        st.success("✅ PPG Recording Complete!")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Heart Rate", f"{result.heart_rate:.1f} BPM")
        with col2:
            st.metric("Frames Processed", result.frames_processed)
        with col3:
            st.metric("Duration", f"{result.duration:.1f}s")
        
        # Plot PPG signal
        if len(result.ppg_signal) > 0:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=result.ppg_signal,
                name="PPG Signal",
                line=dict(color='red')
            ))
            fig.update_layout(
                title="Extracted PPG Signal",
                xaxis_title="Sample",
                yaxis_title="Amplitude",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Return result for further processing
        metadata = {
            'method': 'WebRTC Real-time',
            'heart_rate': result.heart_rate,
            'confidence': result.confidence,
            'frames_processed': result.frames_processed,
            'duration': result.duration
        }
        
        # Add new recording button
        st.markdown("---")
        if st.button("🔄 Record New PPG Session", use_container_width=True):
            # Reset processor state
            processor.recording = False
            processor.frames = []
            processor.ppg_values = []
            if 'ppg_result' in st.session_state:
                del st.session_state.ppg_result
            st.rerun()
        
        return result, metadata
    
    return None, {}

# Simplified interface function for compatibility
def create_real_webrtc_ppg_interface(duration: float = 30.0) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Compatibility wrapper for existing code."""
    
    result, metadata = create_webrtc_ppg_interface(duration)
    
    if result is not None:
        return result.ppg_signal, metadata
    
    return None, {}