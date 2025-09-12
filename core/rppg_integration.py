"""
rPPG Integration Module
=======================
Integrates webcam-pulse-detector's heart rate detection into Flask app
"""

import sys
import os
import numpy as np
import cv2
import time
from typing import Dict, List, Optional, Tuple
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
import logging

# Add webcam-pulse-detector to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external', 'webcam-pulse-detector'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external', 'webcam-pulse-detector', 'lib'))

logger = logging.getLogger(__name__)

class SimplifiedRPPGProcessor:
    """Simplified rPPG processor based on webcam-pulse-detector"""
    
    def __init__(self):
        self.buffer_size = 250
        self.data_buffer = []
        self.times = []
        self.fps = 30  # Default FPS
        self.bpm = 0
        self.face_cascade = None
        
        # Try to load face cascade
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
            'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt.xml'
        ]
        
        for path in cascade_paths:
            if os.path.exists(path):
                self.face_cascade = cv2.CascadeClassifier(path)
                if not self.face_cascade.empty():
                    logger.info(f"Face cascade loaded from {path}")
                    break
        
        if self.face_cascade is None or self.face_cascade.empty():
            logger.warning("Could not load face cascade - face detection disabled")
    
    def detect_face(self, frame):
        """Detect face in frame and return forehead region"""
        if self.face_cascade is None or self.face_cascade.empty():
            # Return center region if no face detection
            h, w = frame.shape[:2]
            return frame[h//4:h//2, w//3:2*w//3]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Get the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Extract forehead region (top 40% of face, middle 60% width)
            forehead_x = int(x + w * 0.2)
            forehead_y = int(y + h * 0.05)
            forehead_w = int(w * 0.6)
            forehead_h = int(h * 0.4)
            
            return frame[forehead_y:forehead_y+forehead_h, 
                        forehead_x:forehead_x+forehead_w]
        
        # Return center region if no face found
        h, w = frame.shape[:2]
        return frame[h//4:h//2, w//3:2*w//3]
    
    def extract_color_signal(self, frames):
        """Extract PPG signal from frames using green channel"""
        signal = []
        frame_hashes = []  # To check if frames are different
        
        for i, frame in enumerate(frames):
            try:
                # Calculate frame hash to check for duplicates
                frame_hash = hash(frame.tobytes())
                frame_hashes.append(frame_hash)
                
                # Detect face and get forehead region
                forehead = self.detect_face(frame)
                
                if forehead.size == 0:
                    # Use whole frame if detection failed
                    forehead = frame
                    logger.warning(f"Face detection failed for frame {i}")
                
                # Extract green channel mean (best for PPG)
                if len(forehead.shape) == 3:
                    green_mean = np.mean(forehead[:, :, 1])
                    # Also log min/max for debugging
                    green_min = np.min(forehead[:, :, 1])
                    green_max = np.max(forehead[:, :, 1])
                else:
                    green_mean = np.mean(forehead)
                    green_min = np.min(forehead)
                    green_max = np.max(forehead)
                
                signal.append(green_mean)
                
                # Debug logging for first 10 frames and last 10 frames
                if i < 10 or i >= len(frames) - 10:
                    logger.info(f"Frame {i}: green_mean={green_mean:.2f}, min={green_min:.2f}, max={green_max:.2f}")
                
            except Exception as e:
                logger.warning(f"Error processing frame {i}: {e}")
                signal.append(signal[-1] if signal else 128)
        
        # Check for duplicate frames
        unique_frames = len(set(frame_hashes))
        logger.info(f"Processed {len(frames)} frames, {unique_frames} unique frames")
        if unique_frames < len(frames) * 0.9:
            logger.warning(f"Many duplicate frames detected! Only {unique_frames}/{len(frames)} unique")
        
        # Log signal statistics
        signal_array = np.array(signal)
        logger.info(f"Signal stats - Mean: {np.mean(signal_array):.2f}, Std: {np.std(signal_array):.4f}, "
                   f"Min: {np.min(signal_array):.2f}, Max: {np.max(signal_array):.2f}")
        
        return signal_array
    
    def calculate_heart_rate_fft(self, signal, fps=30):
        """Calculate heart rate using FFT"""
        try:
            # Detrend signal
            signal = scipy_signal.detrend(signal)
            
            # Apply bandpass filter (0.75-4 Hz for 45-240 BPM)
            nyquist = fps / 2
            low = 0.75 / nyquist
            high = min(4.0 / nyquist, 0.99)
            
            if low < high:
                b, a = scipy_signal.butter(2, [low, high], btype='band')
                signal = scipy_signal.filtfilt(b, a, signal)
            
            # Compute FFT
            fft = np.fft.rfft(signal)
            freqs = np.fft.rfftfreq(len(signal), 1/fps)
            
            # Find peak in physiological range
            valid_range = (freqs >= 0.75) & (freqs <= 4.0)
            if np.any(valid_range):
                fft_magnitude = np.abs(fft[valid_range])
                valid_freqs = freqs[valid_range]
                peak_idx = np.argmax(fft_magnitude)
                peak_freq = valid_freqs[peak_idx]
                heart_rate = peak_freq * 60
                
                # Sanity check
                if 45 <= heart_rate <= 180:
                    return heart_rate
            
            return 75.0  # Default
            
        except Exception as e:
            logger.error(f"FFT calculation error: {e}")
            return 75.0
    
    def calculate_heart_rate_peaks(self, signal, fps=30):
        """Calculate heart rate using peak detection"""
        try:
            # Find peaks
            min_distance = int(fps * 0.4)  # Minimum 0.4 seconds between beats
            peaks, _ = find_peaks(signal, distance=min_distance)
            
            if len(peaks) > 1:
                # Calculate average interval
                intervals = np.diff(peaks) / fps  # Convert to seconds
                avg_interval = np.mean(intervals)
                heart_rate = 60 / avg_interval
                
                if 45 <= heart_rate <= 180:
                    return heart_rate
            
            return 75.0  # Default
            
        except Exception as e:
            logger.error(f"Peak detection error: {e}")
            return 75.0
    
    def process_frames(self, frames, fps=30):
        """Process frames and extract heart rate"""
        if len(frames) < fps * 2:  # Need at least 2 seconds
            return {
                'success': False,
                'error': 'Not enough frames for analysis',
                'heart_rate': 0,
                'confidence': 0
            }
        
        try:
            # Extract color signal
            signal = self.extract_color_signal(frames)
            
            # Normalize signal
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
            
            # Calculate heart rate using both methods
            hr_fft = self.calculate_heart_rate_fft(signal, fps)
            hr_peaks = self.calculate_heart_rate_peaks(signal, fps)
            
            # Average the two methods
            heart_rate = (hr_fft + hr_peaks) / 2
            
            # Calculate confidence based on agreement
            confidence = 1.0 - abs(hr_fft - hr_peaks) / 100.0
            confidence = max(0.1, min(1.0, confidence))
            
            return {
                'success': True,
                'heart_rate': heart_rate,
                'heart_rate_fft': hr_fft,
                'heart_rate_peaks': hr_peaks,
                'confidence': confidence,
                'signal': signal.tolist()[:500],  # Limit for display
                'signal_length': len(signal),
                'fps': fps
            }
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return {
                'success': False,
                'error': str(e),
                'heart_rate': 75,
                'confidence': 0
            }


def process_video_frames(frames, fps=15):
    """Main entry point for processing video frames"""
    processor = SimplifiedRPPGProcessor()
    return processor.process_frames(frames, fps)


def test_with_synthetic_data():
    """Test the processor with synthetic data"""
    # Generate synthetic frames with a heartbeat pattern
    fps = 30
    duration = 10  # seconds
    heart_rate = 72  # BPM
    
    num_frames = fps * duration
    t = np.linspace(0, duration, num_frames)
    
    # Create synthetic PPG signal
    heart_freq = heart_rate / 60  # Hz
    signal = 128 + 10 * np.sin(2 * np.pi * heart_freq * t)
    
    # Create fake frames (just arrays with the signal value)
    frames = []
    for val in signal:
        # Create a fake 3-channel image with uint8 values
        frame = np.ones((100, 100, 3), dtype=np.uint8) * int(val)
        frames.append(frame)
    
    # Process
    processor = SimplifiedRPPGProcessor()
    result = processor.process_frames(frames, fps)
    
    print(f"Test Results:")
    print(f"  Expected HR: {heart_rate} BPM")
    print(f"  Detected HR: {result.get('heart_rate', 0):.1f} BPM")
    print(f"  Confidence: {result.get('confidence', 0):.2f}")
    print(f"  Success: {result.get('success', False)}")
    
    return result


if __name__ == "__main__":
    # Run test
    test_with_synthetic_data()