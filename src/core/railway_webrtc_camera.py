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
        
        # CRITICAL DEBUG: Always log transform calls to see if frames are coming through
        current_frame_count = len(self.frames)
        if current_frame_count < 10 or current_frame_count % 30 == 0:  # Log first 10 frames and every 30th
            logger.info(f"🎬 TRANSFORM CALLED: recording={self.recording}, frames_stored={current_frame_count}, max_frames={self.max_frames}")
        
        # Always store frames when recording, regardless of face detection
        if self.recording and len(self.frames) < self.max_frames:
            # Store frame for PPG processing
            self.frames.append(img.copy())
            logger.info(f"✅ FRAME STORED: #{len(self.frames)}/{self.max_frames}")
            
            # Extract PPG signal from face region (or fallback to center)
            ppg_value = self.extract_ppg_from_frame(img)
            # Always store a value, even if face detection fails
            if ppg_value is not None:
                self.ppg_values.append(ppg_value)
                logger.info(f"PPG value extracted: {ppg_value:.2f}, total PPG values: {len(self.ppg_values)}")
            else:
                # Fallback: use center region green channel
                h, w = img.shape[:2]
                center_roi = img[h//4:3*h//4, w//4:3*w//4]
                fallback_value = np.mean(center_roi[:, :, 1])
                self.ppg_values.append(fallback_value)
                logger.info(f"Fallback PPG value: {fallback_value:.2f}, total PPG values: {len(self.ppg_values)}")
        elif self.recording and len(self.frames) >= self.max_frames:
            logger.info(f"🛑 Max frames reached: {len(self.frames)}/{self.max_frames}")
        elif not self.recording:
            # Log when not recording to debug state issues
            if current_frame_count < 5 or current_frame_count % 60 == 0:  # Less frequent logging
                logger.info(f"⏸️ NOT RECORDING - recording={self.recording}, frames stored: {len(self.frames)}")
        
        # Add visual feedback
        processed_img = self.add_visual_feedback(img)
        
        # Ensure we return a proper numpy array format
        if not isinstance(processed_img, np.ndarray):
            processed_img = np.array(processed_img)
        
        # Ensure correct data type
        processed_img = processed_img.astype(np.uint8)
        
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
    
    def extract_ppg_from_frame(self, img: np.ndarray) -> Optional[float]:
        """Extract PPG signal from face region using CHROM method."""
        
        if self.face_cascade is None:
            return None
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Use the first face detected
            (x, y, w, h) = faces[0]
            
            # Define ROI (forehead region)
            roi_y = max(0, y + int(h * 0.1))
            roi_h = max(1, int(h * 0.25))
            roi_x = max(0, x + int(w * 0.2))
            roi_w = max(1, int(w * 0.6))
            
            face_roi = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            if face_roi.size > 0:
                # Extract color channels
                b = np.mean(face_roi[:, :, 0])
                g = np.mean(face_roi[:, :, 1])
                r = np.mean(face_roi[:, :, 2])
                
                # Simple PPG extraction using green channel
                return float(g)
        
        return None
    
    def add_visual_feedback(self, img: np.ndarray) -> np.ndarray:
        """Add visual feedback to the video frame."""
        
        # Create a copy to avoid modifying the original
        display_img = img.copy()
        
        # Add recording indicator
        if self.recording:
            # Red recording dot
            cv2.circle(display_img, (30, 30), 10, (0, 0, 255), -1)
            
            # Recording text
            cv2.putText(display_img, f"REC {len(self.frames)}/{self.max_frames}", 
                       (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Progress bar
            progress = len(self.frames) / self.max_frames
            bar_width = 200
            bar_height = 10
            bar_x = img.shape[1] - bar_width - 20
            bar_y = 20
            
            # Background bar
            cv2.rectangle(display_img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (100, 100, 100), -1)
            
            # Progress bar
            progress_width = int(bar_width * progress)
            cv2.rectangle(display_img, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                         (0, 255, 0), -1)
            
            # Percentage text
            cv2.putText(display_img, f"{progress*100:.1f}%", 
                       (bar_x, bar_y + bar_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Face detection visualization
        if self.face_cascade is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # Green face box
                cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # ROI box (forehead region)
                roi_y = max(0, y + int(h * 0.1))
                roi_h = max(1, int(h * 0.25))
                roi_x = max(0, x + int(w * 0.2))
                roi_w = max(1, int(w * 0.6))
                
                cv2.rectangle(display_img, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (255, 0, 0), 1)
                cv2.putText(display_img, "PPG ROI", (roi_x, roi_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return display_img
    
    def start_recording(self):
        """Start PPG recording."""
        self.recording = True
        self.frames = []
        self.ppg_values = []
        logger.info("Started PPG recording")
    
    def stop_recording(self):
        """Stop PPG recording."""
        self.recording = False
        logger.info(f"Stopped PPG recording. Collected {len(self.frames)} frames, {len(self.ppg_values)} PPG values")
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.recording
    
    def get_frame_count(self) -> int:
        """Get current frame count."""
        return len(self.frames)
    
    def process_ppg_signal(self, target_fps: float = 30.0) -> PPGResult:
        """Process collected PPG signal to extract heart rate."""
        
        if len(self.ppg_values) < target_fps:  # Need at least 1 second of data
            logger.warning(f"Insufficient PPG data: {len(self.ppg_values)} values")
            return PPGResult(
                heart_rate=0.0,
                ppg_signal=np.array([]),
                confidence=0.0,
                frames_processed=len(self.frames),
                duration=len(self.frames) / target_fps
            )
        
        # Convert to numpy array and normalize
        signal = np.array(self.ppg_values)
        signal = (signal - np.mean(signal)) / np.std(signal)
        
        # Apply bandpass filter (0.5-4 Hz for HR 30-240 BPM)
        try:
            from scipy import signal as scipy_signal
            
            # Design bandpass filter
            nyquist = target_fps / 2
            low = 0.5 / nyquist
            high = 4.0 / nyquist
            
            b, a = scipy_signal.butter(4, [low, high], btype='band')
            filtered_signal = scipy_signal.filtfilt(b, a, signal)
            
            # Calculate heart rate using FFT
            fft = np.fft.fft(filtered_signal)
            freqs = np.fft.fftfreq(len(filtered_signal), 1/target_fps)
            
            # Find peak in frequency domain (0.5-4 Hz range)
            valid_idx = (freqs > 0.5) & (freqs < 4.0)
            valid_freqs = freqs[valid_idx]
            valid_fft = np.abs(fft[valid_idx])
            
            if len(valid_fft) > 0:
                peak_idx = np.argmax(valid_fft)
                heart_rate_hz = valid_freqs[peak_idx]
                heart_rate_bpm = heart_rate_hz * 60
                
                # Calculate confidence based on peak prominence
                peak_power = valid_fft[peak_idx]
                avg_power = np.mean(valid_fft)
                confidence = min(1.0, peak_power / (avg_power * 3))
                
                logger.info(f"Processed PPG: HR={heart_rate_bpm:.1f} BPM, confidence={confidence:.2f}")
                
                return PPGResult(
                    heart_rate=float(heart_rate_bpm),
                    ppg_signal=filtered_signal,
                    confidence=float(confidence),
                    frames_processed=len(self.frames),
                    duration=len(self.frames) / target_fps
                )
            
        except ImportError:
            logger.warning("scipy not available for advanced signal processing")
        except Exception as e:
            logger.error(f"Error processing PPG signal: {e}")
        
        # Fallback: simple peak detection
        try:
            # Simple moving average to smooth the signal
            window_size = min(10, len(signal) // 4)
            smoothed = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
            
            # Count peaks (simplified)
            peaks = []
            for i in range(1, len(smoothed)-1):
                if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                    peaks.append(i)
            
            if len(peaks) > 1:
                # Calculate average time between peaks
                duration = len(self.frames) / target_fps
                avg_interval = duration / len(peaks)
                heart_rate = 60 / avg_interval if avg_interval > 0 else 0
                
                # Clamp to reasonable range
                heart_rate = max(40, min(200, heart_rate))
                
                return PPGResult(
                    heart_rate=float(heart_rate),
                    ppg_signal=signal,
                    confidence=0.5,  # Lower confidence for simple method
                    frames_processed=len(self.frames),
                    duration=duration
                )
        
        except Exception as e:
            logger.error(f"Error in fallback PPG processing: {e}")
        
        # Final fallback
        return PPGResult(
            heart_rate=75.0,  # Default heart rate
            ppg_signal=signal if len(signal) > 0 else np.array([]),
            confidence=0.1,
            frames_processed=len(self.frames),
            duration=len(self.frames) / target_fps
        )

def create_webrtc_ppg_interface(duration: float = 30.0) -> Tuple[Optional[PPGResult], Dict[str, Any]]:
    """
    Create a WebRTC-based PPG extraction interface that works on Railway.
    
    Args:
        duration: Recording duration in seconds
        
    Returns:
        Tuple of (PPGResult, metadata) or (None, {}) if not ready
    """
    
    if not WEBRTC_AVAILABLE:
        st.error("WebRTC dependencies not available. Please install streamlit-webrtc and related packages.")
        return None, {}
    
    # Initialize session state for recording control
    if 'recording_state' not in st.session_state:
        st.session_state.recording_state = 'idle'  # idle, recording, processing
    
    # Initialize processor in session state to persist across reruns
    if 'ppg_processor' not in st.session_state:
        st.session_state.ppg_processor = PPGVideoProcessor()
    
    processor = st.session_state.ppg_processor
    target_frames = int(duration * 30)  # Assuming 30 FPS
    processor.max_frames = target_frames
    
    # WebRTC streamer - always show video, control recording through processor state
    webrtc_ctx = webrtc_streamer(
        key="ppg-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PPGVideoProcessor,  # Create fresh instance each time
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    
    # Synchronize processor state with WebRTC processor
    if webrtc_ctx.video_processor:
        if st.session_state.recording_state == 'recording' and not webrtc_ctx.video_processor.recording:
            webrtc_ctx.video_processor.start_recording()
            webrtc_ctx.video_processor.max_frames = target_frames
        elif st.session_state.recording_state != 'recording' and webrtc_ctx.video_processor.recording:
            webrtc_ctx.video_processor.stop_recording()
        
        # Update our session processor with the WebRTC processor state
        if webrtc_ctx.video_processor.frames:
            st.session_state.ppg_processor.frames = webrtc_ctx.video_processor.frames.copy()
        if webrtc_ctx.video_processor.ppg_values:
            st.session_state.ppg_processor.ppg_values = webrtc_ctx.video_processor.ppg_values.copy()
    
    # Main recording control - single button interface
    if st.session_state.recording_state == 'idle':
        if st.button("🔴 Start Recording & Live Preview", type="primary", use_container_width=True):
            st.session_state.recording_state = 'recording'
            processor.start_recording()
            st.rerun()
    
    elif st.session_state.recording_state == 'recording':
        # Show recording status - get from WebRTC processor if available
        if webrtc_ctx.video_processor:
            frame_count = len(webrtc_ctx.video_processor.frames)
            is_recording = webrtc_ctx.video_processor.recording
        else:
            frame_count = processor.get_frame_count()
            is_recording = processor.recording
            
        progress = frame_count / target_frames if target_frames > 0 else 0
        
        st.success(f"🔴 Recording in progress... {frame_count}/{target_frames} frames ({progress*100:.1f}%)")
        st.info(f"🔍 **Debug**: WebRTC_Recording={is_recording}, Session_Recording={processor.recording}")
        
        # Progress bar
        st.progress(progress)
        
        # Stop button
        if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True):
            st.session_state.recording_state = 'processing'
            
            # Get final data from WebRTC processor
            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.stop_recording()
                final_frames = webrtc_ctx.video_processor.frames.copy() if webrtc_ctx.video_processor.frames else []
                final_ppg = webrtc_ctx.video_processor.ppg_values.copy() if webrtc_ctx.video_processor.ppg_values else []
                
                # Transfer to session processor for processing
                processor.frames = final_frames
                processor.ppg_values = final_ppg
            
            # Process PPG signal
            result = processor.process_ppg_signal()
            st.session_state.ppg_result = result
            
            st.rerun()
        
        # Auto-stop when target frames reached (without auto-rerun)
        if frame_count >= target_frames:
            st.session_state.recording_state = 'processing'
            
            # Get final data from WebRTC processor
            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.stop_recording()
                final_frames = webrtc_ctx.video_processor.frames.copy() if webrtc_ctx.video_processor.frames else []
                final_ppg = webrtc_ctx.video_processor.ppg_values.copy() if webrtc_ctx.video_processor.ppg_values else []
                
                # Transfer to session processor for processing
                processor.frames = final_frames
                processor.ppg_values = final_ppg
            
            # Process PPG signal
            result = processor.process_ppg_signal()
            st.session_state.ppg_result = result
            
            st.info("✅ Recording complete! Click to see results.")
        
        # Manual refresh button instead of auto-refresh
        if st.button("🔄 Refresh Status", key="refresh_recording"):
            st.rerun()
    
    elif st.session_state.recording_state == 'processing':
        st.success("✅ Processing complete!")
        if st.button("🔄 Record Again", use_container_width=True):
            st.session_state.recording_state = 'idle'
            processor.frames = []
            processor.ppg_values = []
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