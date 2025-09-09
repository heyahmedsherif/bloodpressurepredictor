"""
Real WebRTC Camera with PPG Extraction for Streamlit Cloud
=========================================================

This module provides REAL PPG extraction from camera using WebRTC with proper TURN server support.
Works on Streamlit Cloud by using Twilio for network traversal.

Key Features:
- Real-time video processing with actual rPPG extraction
- Twilio TURN server integration for cloud compatibility
- Face detection and ROI extraction
- Signal processing for heart rate estimation
- Green channel analysis for PPG signal extraction

Author: Claude Code Integration  
Date: 2025-09-09
"""

import streamlit as st
import numpy as np
import cv2
import time
import queue
import threading
import os
from typing import Optional, Tuple, Dict, Any, List
import logging
from dataclasses import dataclass
from scipy import signal
from scipy.stats import pearsonr

# WebRTC imports
try:
    from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError as e:
    logging.warning(f"streamlit-webrtc not available: {e}")
    WEBRTC_AVAILABLE = False

# Twilio import for TURN servers
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    logging.warning("Twilio not available - using public STUN servers only")
    TWILIO_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PPGFrameData:
    """Container for PPG-relevant frame data"""
    frame: np.ndarray
    timestamp: float
    roi_detected: bool
    face_box: Tuple[int, int, int, int]  # x, y, w, h
    green_mean: float
    red_mean: float
    blue_mean: float

class RealPPGProcessor(VideoTransformerBase):
    """
    Real-time PPG extraction from video using actual rPPG algorithms
    """
    
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=2000)  # Store ~60 seconds at 30fps
        self.recording = False
        self.start_time = None
        self.target_duration = 30.0
        self.frame_count = 0
        
        # PPG extraction state
        self.ppg_values = []
        self.timestamps = []
        
        # Face detection
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            logger.warning(f"Could not load face cascade: {e}")
            self.face_cascade = None
    
    def recv(self, frame):
        """Process incoming video frames for real PPG extraction"""
        if not WEBRTC_AVAILABLE:
            return frame
        
        # Convert to numpy array
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()
        
        if self.recording and self.start_time:
            elapsed = current_time - self.start_time
            
            # Stop recording after target duration
            if elapsed >= self.target_duration:
                self.recording = False
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Extract PPG data from this frame
            roi_detected, face_box, rgb_means = self._extract_ppg_from_frame(img)
            
            if roi_detected:
                # Store PPG data
                frame_data = PPGFrameData(
                    frame=img.copy(),
                    timestamp=current_time,
                    roi_detected=True,
                    face_box=face_box,
                    green_mean=rgb_means[1],  # Green channel is most important for PPG
                    red_mean=rgb_means[0],
                    blue_mean=rgb_means[2]
                )
                
                # Add to processing queue
                try:
                    self.frame_queue.put_nowait(frame_data)
                except queue.Full:
                    # Remove oldest frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame_data)
                    except queue.Empty:
                        pass
            
            # Visual feedback
            img = self._draw_recording_feedback(img, elapsed, roi_detected, face_box if roi_detected else None)
            self.frame_count += 1
        
        else:
            # Preview mode - show face detection
            roi_detected, face_box, _ = self._extract_ppg_from_frame(img)
            img = self._draw_preview_feedback(img, roi_detected, face_box if roi_detected else None)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _extract_ppg_from_frame(self, img: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]], Optional[Tuple[float, float, float]]]:
        """Extract PPG signal component from a single frame"""
        if self.face_cascade is None:
            return False, None, None
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(80, 80),
            maxSize=(400, 400)
        )
        
        if len(faces) == 0:
            return False, None, None
        
        # Use the largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Extract forehead/cheek ROI for PPG (more reliable than full face)
        roi_x = x + int(w * 0.3)
        roi_y = y + int(h * 0.1) 
        roi_w = int(w * 0.4)
        roi_h = int(h * 0.25)
        
        # Ensure ROI is within image bounds
        roi_x = max(0, roi_x)
        roi_y = max(0, roi_y)
        roi_w = min(roi_w, img.shape[1] - roi_x)
        roi_h = min(roi_h, img.shape[0] - roi_y)
        
        if roi_w <= 0 or roi_h <= 0:
            return False, None, None
        
        # Extract ROI
        roi = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        if roi.size == 0:
            return False, None, None
        
        # Calculate mean RGB values (this is the raw PPG signal)
        blue_mean = float(np.mean(roi[:, :, 0]))   # B
        green_mean = float(np.mean(roi[:, :, 1]))  # G  
        red_mean = float(np.mean(roi[:, :, 2]))    # R
        
        return True, (x, y, w, h), (red_mean, green_mean, blue_mean)
    
    def _draw_recording_feedback(self, img: np.ndarray, elapsed: float, roi_detected: bool, face_box: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        """Draw recording progress and ROI indicators"""
        h, w = img.shape[:2]
        
        if face_box:
            x, y, fw, fh = face_box
            # Draw face box
            cv2.rectangle(img, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            
            # Draw PPG ROI
            roi_x = x + int(fw * 0.3)
            roi_y = y + int(fh * 0.1)
            roi_w = int(fw * 0.4) 
            roi_h = int(fh * 0.25)
            cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 255), 2)
            
            # Labels
            cv2.putText(img, "FACE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, "PPG ROI", (roi_x, roi_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(img, "NO FACE - POSITION YOURSELF", (w//2-200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # Recording progress
        progress = elapsed / self.target_duration
        bar_width = 300
        bar_height = 20
        bar_x = (w - bar_width) // 2
        bar_y = h - 60
        
        # Progress bar background
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Progress bar fill
        fill_width = int(bar_width * min(progress, 1.0))
        color = (0, 255, 0) if roi_detected else (0, 0, 255)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        # Text
        status = "RECORDING PPG" if roi_detected else "NO FACE DETECTED"
        cv2.putText(img, f"🔴 {status}: {elapsed:.1f}s/{self.target_duration:.1f}s", 
                   (bar_x, bar_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img
    
    def _draw_preview_feedback(self, img: np.ndarray, roi_detected: bool, face_box: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        """Draw preview mode feedback"""
        h, w = img.shape[:2]
        
        cv2.putText(img, "📹 CAMERA PREVIEW - READY FOR PPG", (w//2-200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if face_box:
            x, y, fw, fh = face_box
            cv2.rectangle(img, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.putText(img, "✅ FACE DETECTED - READY", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show PPG ROI preview
            roi_x = x + int(fw * 0.3)
            roi_y = y + int(fh * 0.1)
            roi_w = int(fw * 0.4)
            roi_h = int(fh * 0.25)
            cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 2)
            cv2.putText(img, "PPG ROI", (roi_x, roi_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            cv2.putText(img, "⚠️ NO FACE - POSITION YOURSELF", (w//2-180, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        return img
    
    def start_recording(self, duration: float = 30.0):
        """Start PPG recording"""
        self.recording = True
        self.start_time = time.time()
        self.target_duration = duration
        self.frame_count = 0
        
        # Clear previous data
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info(f"Started real PPG recording for {duration}s")
    
    def stop_recording(self):
        """Stop PPG recording"""
        self.recording = False
        logger.info(f"Stopped PPG recording after {self.frame_count} frames")
    
    def get_recorded_ppg_data(self) -> List[PPGFrameData]:
        """Get all recorded PPG frame data"""
        frames = []
        while not self.frame_queue.empty():
            try:
                frames.append(self.frame_queue.get_nowait())
            except queue.Empty:
                break
        return frames

def extract_real_ppg_signal(ppg_frames: List[PPGFrameData]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract real PPG signal from recorded frames using advanced rPPG algorithms
    """
    if not ppg_frames:
        logger.warning("No PPG frames available")
        return np.array([]), {'error': 'no_frames'}
    
    # Extract timestamps and RGB signals
    timestamps = np.array([f.timestamp for f in ppg_frames])
    green_signal = np.array([f.green_mean for f in ppg_frames])
    red_signal = np.array([f.red_mean for f in ppg_frames])
    blue_signal = np.array([f.blue_mean for f in ppg_frames])
    
    logger.info(f"Processing {len(ppg_frames)} frames for real PPG extraction")
    
    # Normalize timestamps to start from 0
    timestamps = timestamps - timestamps[0]
    
    # Calculate sampling rate
    if len(timestamps) > 1:
        sampling_rate = 1.0 / np.mean(np.diff(timestamps))
    else:
        sampling_rate = 30.0  # Default
    
    # Apply CHROM algorithm (Chrominance-based rPPG)
    ppg_signal = apply_chrom_algorithm(red_signal, green_signal, blue_signal)
    
    # Signal processing
    ppg_signal = preprocess_ppg_signal(ppg_signal, sampling_rate)
    
    # Extract heart rate
    heart_rate = estimate_heart_rate_fft(ppg_signal, sampling_rate)
    
    # Calculate quality metrics
    quality_score = calculate_signal_quality(ppg_signal, sampling_rate)
    
    metadata = {
        'method': 'real_webrtc_chrom',
        'frames_processed': len(ppg_frames),
        'sampling_rate': sampling_rate,
        'duration': timestamps[-1] if len(timestamps) > 0 else 0,
        'heart_rate': heart_rate,
        'quality_score': quality_score,
        'algorithm': 'CHROM',
        'note': 'Real PPG extracted from camera using CHROM algorithm'
    }
    
    return ppg_signal, metadata

def apply_chrom_algorithm(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """
    Apply CHROM (Chrominance) algorithm for rPPG extraction
    
    Based on: "Algorithmic Principles of Remote PPG" by Wim Verkruysse et al.
    """
    # Normalize signals
    red_norm = red / np.mean(red)
    green_norm = green / np.mean(green) 
    blue_norm = blue / np.mean(blue)
    
    # CHROM algorithm: X = 3*R - 2*G, Y = 1.5*R + G - 1.5*B
    X = 3 * red_norm - 2 * green_norm
    Y = 1.5 * red_norm + green_norm - 1.5 * blue_norm
    
    # Calculate chrominance signal
    alpha = np.std(X) / np.std(Y) if np.std(Y) > 0 else 1
    chrom_signal = X - alpha * Y
    
    return chrom_signal

def preprocess_ppg_signal(ppg_signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    """
    Preprocess PPG signal with filtering and detrending
    """
    if len(ppg_signal) < 10:
        return ppg_signal
    
    # Detrend
    ppg_signal = signal.detrend(ppg_signal)
    
    # Bandpass filter for heart rate range (0.7-4 Hz = 42-240 BPM)
    nyquist = sampling_rate / 2
    if nyquist > 2:  # Ensure we can filter properly
        low_freq = 0.7 / nyquist
        high_freq = min(4.0 / nyquist, 0.99)
        
        try:
            b, a = signal.butter(4, [low_freq, high_freq], btype='band')
            ppg_signal = signal.filtfilt(b, a, ppg_signal)
        except Exception as e:
            logger.warning(f"Filtering failed: {e}")
    
    return ppg_signal

def estimate_heart_rate_fft(ppg_signal: np.ndarray, sampling_rate: float) -> float:
    """
    Estimate heart rate using FFT peak detection
    """
    if len(ppg_signal) < 50:
        return 75.0  # Default
    
    # Compute FFT
    fft_signal = np.abs(np.fft.fft(ppg_signal))
    freqs = np.fft.fftfreq(len(ppg_signal), 1/sampling_rate)
    
    # Focus on physiological range (0.7-4 Hz)
    valid_indices = (freqs >= 0.7) & (freqs <= 4.0)
    if not np.any(valid_indices):
        return 75.0
    
    # Find peak frequency
    valid_fft = fft_signal[valid_indices]
    valid_freqs = freqs[valid_indices]
    
    peak_idx = np.argmax(valid_fft)
    peak_freq = valid_freqs[peak_idx]
    
    # Convert to BPM
    heart_rate = peak_freq * 60
    
    # Sanity check
    return float(np.clip(heart_rate, 50, 150))

def calculate_signal_quality(ppg_signal: np.ndarray, sampling_rate: float) -> float:
    """
    Calculate PPG signal quality score
    """
    if len(ppg_signal) < 10:
        return 0.0
    
    # Signal-to-noise ratio estimate
    signal_power = np.var(ppg_signal)
    if signal_power == 0:
        return 0.0
    
    # Estimate noise as high-frequency components
    if len(ppg_signal) > 20:
        try:
            # High-pass filter to get noise
            nyquist = sampling_rate / 2
            high_freq = min(10.0 / nyquist, 0.99)
            b, a = signal.butter(2, high_freq, btype='high')
            noise = signal.filtfilt(b, a, ppg_signal)
            noise_power = np.var(noise)
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                quality = np.clip(snr / 30.0, 0.0, 1.0)  # Normalize to 0-1
            else:
                quality = 1.0
        except:
            quality = 0.5  # Default if filtering fails
    else:
        quality = 0.3  # Low quality for short signals
    
    return float(quality)

def get_rtc_configuration() -> RTCConfiguration:
    """
    Get WebRTC configuration with proper TURN servers for Streamlit Cloud
    """
    # Try to get Twilio TURN servers if available
    if TWILIO_AVAILABLE:
        try:
            # Check for Twilio credentials in secrets
            account_sid = st.secrets.get("TWILIO_ACCOUNT_SID")
            auth_token = st.secrets.get("TWILIO_AUTH_TOKEN")
            
            if account_sid and auth_token:
                client = Client(account_sid, auth_token)
                token = client.tokens.create()
                
                return RTCConfiguration({
                    "iceServers": token.ice_servers
                })
        except Exception as e:
            logger.warning(f"Failed to get Twilio TURN servers: {e}")
    
    # Fallback to public STUN/TURN servers
    return RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            # Public TURN servers (may have limitations)
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            }
        ]
    })

def create_real_webrtc_ppg_interface(duration: float = 30.0) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Create real WebRTC PPG extraction interface that works on Streamlit Cloud
    """
    if not WEBRTC_AVAILABLE:
        st.error("🚫 **WebRTC not available**")
        st.info("Please ensure streamlit-webrtc and aiortc are installed")
        return None, {'error': 'webrtc_unavailable'}
    
    st.success("🌐 **Real PPG Extraction Active** - Using WebRTC with TURN servers!")
    st.info("📹 This uses your camera to extract **real PPG signals** from facial blood flow changes")
    
    # Initialize session state
    if 'webrtc_processor' not in st.session_state:
        st.session_state.webrtc_processor = None
    
    # Get WebRTC configuration
    rtc_config = get_rtc_configuration()
    
    # Camera controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔴 Start PPG Recording", type="primary"):
            st.session_state.recording_requested = True
    
    with col2:
        if st.button("⏹️ Stop Recording"):
            st.session_state.recording_requested = False
            if st.session_state.webrtc_processor:
                st.session_state.webrtc_processor.stop_recording()
    
    with col3:
        show_instructions = st.checkbox("💡 Show Instructions", value=True)
    
    if show_instructions:
        with st.expander("📋 How Real PPG Extraction Works", expanded=True):
            st.markdown("""
            **🔬 Real rPPG Science**:
            1. **Camera captures** your face at 30fps 
            2. **Face detection** finds optimal skin regions
            3. **Color analysis** extracts tiny RGB changes from blood flow
            4. **CHROM algorithm** processes signals to isolate heart rhythm
            5. **Signal processing** filters noise and estimates heart rate
            6. **PPG signal** is sent to PaPaGei models for health predictions
            
            **💡 Tips for Best Results**:
            - Sit 60-80cm from camera
            - Ensure good, even lighting 
            - Keep face stable and visible
            - Avoid movement during recording
            """)
    
    # WebRTC streamer with real PPG processor
    webrtc_ctx = webrtc_streamer(
        key="real-ppg-extractor",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=RealPPGProcessor,
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
        
        if st.session_state.recording_requested and not processor.recording:
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
                status_placeholder.info(f"🔴 **Extracting Real PPG**: {elapsed:.1f}s / {duration:.1f}s")
                
                if elapsed >= duration + 2:  # Add buffer time
                    break
                
                time.sleep(0.5)
            
            # Get recorded data
            ppg_frames = processor.get_recorded_ppg_data()
            
            if ppg_frames:
                status_placeholder.success(f"✅ **Real PPG Extracted!** Processed {len(ppg_frames)} frames with face detection")
                
                # Extract PPG signal using real rPPG algorithms
                with st.spinner("🧬 Processing PPG signal with CHROM algorithm..."):
                    ppg_signal, metadata = extract_real_ppg_signal(ppg_frames)
                
                # Show extraction results
                if len(ppg_signal) > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Heart Rate", f"{metadata.get('heart_rate', 0):.0f} BPM")
                    with col2:
                        st.metric("Signal Quality", f"{metadata.get('quality_score', 0):.2f}")
                    with col3:
                        st.metric("Frames Used", f"{metadata.get('frames_processed', 0)}")
                    
                    return ppg_signal, metadata
                else:
                    status_placeholder.error("❌ Failed to extract valid PPG signal")
            else:
                status_placeholder.error("❌ No frames with face detection recorded. Please ensure your face is visible.")
    
    return None, {'status': 'preview_mode'}

# Main interface function
def extract_ppg_from_real_webrtc_camera(duration: float = 30.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract real PPG signal from WebRTC camera (works on Streamlit Cloud with TURN servers)
    """
    return create_real_webrtc_ppg_interface(duration)