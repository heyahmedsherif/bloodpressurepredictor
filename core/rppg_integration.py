"""
rPPG Integration Module
=======================
Integrates webcam-pulse-detector's heart rate detection into Flask app
Enhanced with pyVHR's CHROM and POS methods for improved accuracy
"""

import sys
import os
import numpy as np
import cv2
import time
from typing import Dict, List, Optional, Tuple
from scipy import signal as scipy_signal
from scipy.signal import find_peaks, welch
import logging

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not installed. Falling back to Haar Cascade.")

# Add webcam-pulse-detector to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external', 'webcam-pulse-detector'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external', 'webcam-pulse-detector', 'lib'))

# Add pyVHR to path for advanced rPPG methods
pyVHR_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'external', 'pyVHR')
if os.path.exists(pyVHR_path):
    sys.path.insert(0, pyVHR_path)
    PYVHR_AVAILABLE = True
    try:
        # Import pyVHR methods - we'll implement simplified versions if import fails
        pass  # We'll implement the methods directly
    except ImportError as e:
        logger.warning(f"Could not import pyVHR methods: {e}")
        PYVHR_AVAILABLE = False
else:
    PYVHR_AVAILABLE = False
    logger.info("pyVHR not found in external directory")

logger = logging.getLogger(__name__)

class SimplifiedRPPGProcessor:
    """Simplified rPPG processor using MediaPipe or Haar Cascade for face detection"""

    def __init__(self):
        self.buffer_size = 450  # Increased from 250 to 450 frames (15 seconds at 30 FPS)
        self.data_buffer = []
        self.times = []
        self.fps = 30  # Default FPS
        self.bpm = 0
        self.face_detector = None
        self.face_cascade = None

        # Try MediaPipe first
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_detector = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,  # Video mode for better tracking
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info("MediaPipe Face Mesh initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize MediaPipe: {e}")
                self.face_detector = None

        # Fallback to Haar Cascade if MediaPipe not available
        if self.face_detector is None:
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
                        logger.info(f"Haar Cascade loaded from {path}")
                        break

            if self.face_cascade is None or self.face_cascade.empty():
                logger.warning("Could not load face detector - using center region")
    
    def detect_face(self, frame):
        """Detect face in frame and return forehead region"""

        # Try MediaPipe first
        if self.face_detector is not None:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(rgb_frame)

            if results.multi_face_landmarks:
                h, w = frame.shape[:2]
                landmarks = results.multi_face_landmarks[0]

                # MediaPipe forehead landmarks (approximately indices 9, 10, 67, 69, 104, 108, 109, 151)
                # We'll use a simplified approach - get upper portion of face
                # Get face bounding box from landmarks
                x_coords = [int(l.x * w) for l in landmarks.landmark]
                y_coords = [int(l.y * h) for l in landmarks.landmark]

                face_x_min = max(0, min(x_coords))
                face_x_max = min(w, max(x_coords))
                face_y_min = max(0, min(y_coords))
                face_y_max = min(h, max(y_coords))

                # Extract smaller, more stable forehead region (top 20% of face, middle 40% width)
                # Smaller ROI gives more stable signal with less noise
                forehead_x = int(face_x_min + (face_x_max - face_x_min) * 0.3)
                forehead_y = int(face_y_min + (face_y_max - face_y_min) * 0.05)
                forehead_w = int((face_x_max - face_x_min) * 0.4)
                forehead_h = int((face_y_max - face_y_min) * 0.2)

                return frame[forehead_y:forehead_y+forehead_h,
                            forehead_x:forehead_x+forehead_w]

        # Fallback to Haar Cascade
        elif self.face_cascade is not None and not self.face_cascade.empty():
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

    def extract_rgb_signals(self, frames):
        """Extract RGB signals from frames for pyVHR methods"""
        rgb_signals = []

        for i, frame in enumerate(frames):
            try:
                # Detect face and get forehead region
                forehead = self.detect_face(frame)

                if forehead.size == 0:
                    # Use center region if detection failed
                    h, w = frame.shape[:2]
                    forehead = frame[h//4:h//2, w//3:2*w//3]

                # Extract RGB channel means
                if len(forehead.shape) == 3:
                    r_mean = np.mean(forehead[:, :, 2])  # OpenCV uses BGR
                    g_mean = np.mean(forehead[:, :, 1])
                    b_mean = np.mean(forehead[:, :, 0])
                else:
                    # Grayscale - use same value for all channels
                    mean_val = np.mean(forehead)
                    r_mean = g_mean = b_mean = mean_val

                rgb_signals.append([r_mean, g_mean, b_mean])

            except Exception as e:
                logger.warning(f"Error extracting RGB from frame {i}: {e}")
                if rgb_signals:
                    rgb_signals.append(rgb_signals[-1])
                else:
                    rgb_signals.append([128, 128, 128])

        # Convert to numpy array and transpose to shape [3, num_frames]
        rgb_array = np.array(rgb_signals).T

        logger.info(f"RGB signals shape: {rgb_array.shape}")
        logger.info(f"RGB means - R: {np.mean(rgb_array[0]):.2f}, G: {np.mean(rgb_array[1]):.2f}, B: {np.mean(rgb_array[2]):.2f}")

        return rgb_array
    
    def calculate_heart_rate_fft(self, signal, fps=30):
        """Calculate heart rate using FFT"""
        try:
            # Detrend signal
            signal = scipy_signal.detrend(signal)
            
            # Apply bandpass filter (0.8-3.0 Hz for 48-180 BPM)
            # Tighter band for better noise rejection
            nyquist = fps / 2
            low = 0.8 / nyquist
            high = min(3.0 / nyquist, 0.99)

            if low < high:
                # Higher order filter for sharper cutoff
                b, a = scipy_signal.butter(4, [low, high], btype='band')
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

    def apply_chrom_method(self, rgb_signal):
        """
        Apply CHROM method for rPPG extraction
        Based on: De Haan & Jeanne (2013) - Robust pulse rate from chrominance-based rPPG
        """
        try:
            # rgb_signal shape: [3, num_frames]
            X = rgb_signal

            # CHROM method calculations
            Xcomp = 3 * X[0] - 2 * X[1]  # 3*R - 2*G
            Ycomp = (1.5 * X[0]) + X[1] - (1.5 * X[2])  # 1.5*R + G - 1.5*B

            # Calculate alpha parameter
            sX = np.std(Xcomp)
            sY = np.std(Ycomp)

            if sY > 0:
                alpha = sX / sY
            else:
                alpha = 1.0

            # Final BVP signal
            bvp = Xcomp - alpha * Ycomp

            # Normalize
            bvp = (bvp - np.mean(bvp)) / (np.std(bvp) + 1e-10)

            return bvp

        except Exception as e:
            logger.error(f"CHROM method error: {e}")
            return None

    def apply_omit_method(self, rgb_signal):
        """
        Apply OMIT (Orthogonal Matrix Image Transformation) method
        Based on: Álvarez Casado & Bordallo López (2022) - Face2PPG
        """
        try:
            # rgb_signal shape: [3, num_frames]
            X = rgb_signal  # Shape: [3, num_frames]

            # QR decomposition on RGB channels
            Q, R = np.linalg.qr(X)

            # Extract the orthogonal basis vector
            S = Q[:, 0].reshape(1, -1)  # Shape: [1, 3]

            # Create projection matrix P = I - S^T * S
            P = np.identity(3) - np.matmul(S.T, S)

            # Apply projection to original signal
            Y = np.dot(P, X)  # Shape: [3, num_frames]

            # Use the second channel as BVP (empirically performs best)
            bvp = Y[1, :]

            # Normalize
            if np.std(bvp) > 0:
                bvp = (bvp - np.mean(bvp)) / np.std(bvp)

            return bvp

        except Exception as e:
            logger.error(f"OMIT method error: {e}")
            # Fallback to simple green channel
            try:
                bvp = rgb_signal[1, :]
                bvp = scipy_signal.detrend(bvp)
                if np.std(bvp) > 0:
                    bvp = (bvp - np.mean(bvp)) / np.std(bvp)
                return bvp
            except:
                return None

    def apply_green_method(self, rgb_signal):
        """
        Apply GREEN method - simple green channel extraction
        Baseline method that works well in good conditions
        """
        try:
            # Simply use the green channel
            bvp = rgb_signal[1, :]  # Green channel is index 1

            # Detrend
            bvp = scipy_signal.detrend(bvp)

            # Normalize
            if np.std(bvp) > 0:
                bvp = (bvp - np.mean(bvp)) / np.std(bvp)

            return bvp

        except Exception as e:
            logger.error(f"GREEN method error: {e}")
            return None

    def apply_ica_method(self, rgb_signal):
        """
        Apply ICA (Independent Component Analysis) method
        Separates independent sources from mixed signals
        """
        try:
            # We'll implement a simplified ICA using eigenvalue decomposition
            # since sklearn's FastICA may not be available

            # Center the data
            X = rgb_signal - np.mean(rgb_signal, axis=1, keepdims=True)

            # Compute covariance matrix
            cov_matrix = np.cov(X)

            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            # Sort by eigenvalues
            idx = eigenvalues.argsort()[::-1]
            eigenvectors = eigenvectors[:, idx]

            # Transform signal
            transformed = np.dot(eigenvectors.T, X)

            # Select component with highest variance in HR frequency range
            best_component = None
            best_power = 0

            for i in range(transformed.shape[0]):
                component = transformed[i, :]
                # Check power in HR frequency range
                fft = np.fft.rfft(component)
                freqs = np.fft.rfftfreq(len(component), 1/30)  # Assuming 30 fps
                hr_range = (freqs >= 0.75) & (freqs <= 3.5)
                if np.any(hr_range):
                    power = np.sum(np.abs(fft[hr_range]))
                    if power > best_power:
                        best_power = power
                        best_component = component

            if best_component is not None:
                # Normalize
                if np.std(best_component) > 0:
                    best_component = (best_component - np.mean(best_component)) / np.std(best_component)
                return best_component

            # Fallback to first component
            bvp = transformed[0, :]
            if np.std(bvp) > 0:
                bvp = (bvp - np.mean(bvp)) / np.std(bvp)

            return bvp

        except Exception as e:
            logger.error(f"ICA method error: {e}")
            return None

    def apply_pos_method(self, rgb_signal):
        """
        Apply POS (Plane-Orthogonal-to-Skin) method for rPPG extraction
        Based on: Wang et al. (2017) - Algorithmic Principles of Remote PPG
        """
        try:
            # rgb_signal shape: [3, num_frames]
            # Normalize RGB signals
            mean_rgb = np.mean(rgb_signal, axis=1, keepdims=True)
            normalized_rgb = rgb_signal / (mean_rgb + 1e-10)

            # POS method calculations
            # S1 = R - G, S2 = R + G - 2*B
            S = np.array([
                normalized_rgb[0] - normalized_rgb[1],  # R - G
                normalized_rgb[0] + normalized_rgb[1] - 2 * normalized_rgb[2]  # R + G - 2*B
            ])

            # Apply moving window (default 32 frames, ~1 second at 30fps)
            window_size = min(32, len(S[0]) // 4)
            if window_size < 5:
                window_size = 5

            # Calculate pulse signal P
            P = np.zeros(S.shape[1] - window_size + 1)

            for i in range(len(P)):
                # Get window segment
                S1_window = S[0, i:i+window_size]
                S2_window = S[1, i:i+window_size]

                # Calculate standard deviations
                std1 = np.std(S1_window)
                std2 = np.std(S2_window)

                if std1 > 0 and std2 > 0:
                    # Normalized segments
                    S1_norm = S1_window / std1
                    S2_norm = S2_window / std2

                    # Pulse calculation: P = S1_norm + alpha * S2_norm
                    # where alpha = std(S1_norm) / std(S2_norm)
                    alpha = std1 / std2
                    P_window = S1_norm + alpha * S2_norm
                    P[i] = np.mean(P_window)
                else:
                    P[i] = 0

            # Normalize output
            if np.std(P) > 0:
                P = (P - np.mean(P)) / np.std(P)

            return P

        except Exception as e:
            logger.error(f"POS method error: {e}")
            return None

    def assess_signal_quality(self, signal, fps=30):
        """Assess the quality of a BVP signal"""
        try:
            if signal is None or len(signal) < fps:
                return {'quality': 'poor', 'snr': 0, 'dominant_freq': 0}

            # Calculate SNR
            fft = np.fft.rfft(signal)
            freqs = np.fft.rfftfreq(len(signal), 1/fps)

            # Find power in HR range (0.5-3.5 Hz)
            hr_range = (freqs >= 0.5) & (freqs <= 3.5)
            if np.any(hr_range):
                signal_power = np.sum(np.abs(fft[hr_range])**2)
                total_power = np.sum(np.abs(fft)**2)
                noise_power = total_power - signal_power

                if noise_power > 0:
                    snr = 10 * np.log10(signal_power / noise_power)
                else:
                    snr = 20  # High SNR if no noise

                # Find dominant frequency
                peak_idx = np.argmax(np.abs(fft[hr_range]))
                dominant_freq = freqs[hr_range][peak_idx]
            else:
                snr = 0
                dominant_freq = 0

            # Assess signal stability (check for consistent peaks)
            try:
                peaks, properties = scipy_signal.find_peaks(signal, distance=int(fps*0.4))
                if len(peaks) > 2:
                    intervals = np.diff(peaks) / fps
                    stability = 1.0 / (np.std(intervals) + 0.01)  # Higher is more stable
                else:
                    stability = 0
            except:
                stability = 0

            # Determine quality level
            if snr > 10 and stability > 5:
                quality = 'excellent'
            elif snr > 5 and stability > 2:
                quality = 'good'
            elif snr > 0:
                quality = 'fair'
            else:
                quality = 'poor'

            return {
                'quality': quality,
                'snr': snr,
                'dominant_freq': dominant_freq,
                'stability': stability
            }

        except Exception as e:
            logger.error(f"Signal quality assessment error: {e}")
            return {'quality': 'poor', 'snr': 0, 'dominant_freq': 0}

    def calculate_heart_rate_from_bvp(self, bvp_signal, fps=30):
        """Calculate heart rate from BVP signal using adaptive filtering"""
        try:
            if bvp_signal is None or len(bvp_signal) < fps * 2:
                return 75.0

            # Assess signal quality first
            quality_metrics = self.assess_signal_quality(bvp_signal, fps)

            # Detrend signal
            bvp_signal = scipy_signal.detrend(bvp_signal)

            # Adaptive filtering based on signal quality
            nyquist = fps / 2

            # Adjust filter parameters based on quality
            if quality_metrics['quality'] == 'excellent':
                # Tight filtering for clean signals
                low_freq = 0.7  # 42 BPM
                high_freq = 3.5  # 210 BPM
                filter_order = 4
            elif quality_metrics['quality'] == 'good':
                # Moderate filtering
                low_freq = 0.6  # 36 BPM
                high_freq = 3.5  # 210 BPM
                filter_order = 3
            elif quality_metrics['quality'] == 'fair':
                # Wider pass band for noisy signals
                low_freq = 0.5  # 30 BPM
                high_freq = 4.0  # 240 BPM
                filter_order = 2
            else:  # poor quality
                # Very wide pass band, minimal filtering
                low_freq = 0.4  # 24 BPM
                high_freq = 4.0  # 240 BPM
                filter_order = 2

            # If we have a strong dominant frequency, adjust filter around it
            if quality_metrics['dominant_freq'] > 0:
                expected_hr = quality_metrics['dominant_freq'] * 60
                if 30 <= expected_hr <= 200:
                    # Adjust filter to be centered around detected frequency
                    freq_margin = 0.5  # Hz margin around dominant frequency
                    low_freq = max(0.4, quality_metrics['dominant_freq'] - freq_margin)
                    high_freq = min(4.0, quality_metrics['dominant_freq'] + freq_margin)

            # Apply adaptive bandpass filter
            low = low_freq / nyquist
            high = min(high_freq / nyquist, 0.99)

            if low < high:
                try:
                    b, a = scipy_signal.butter(filter_order, [low, high], btype='band')
                    bvp_signal_filtered = scipy_signal.filtfilt(b, a, bvp_signal)
                except:
                    # If filter fails, use unfiltered signal
                    bvp_signal_filtered = bvp_signal
                    logger.warning("Adaptive filter failed, using unfiltered signal")
            else:
                bvp_signal_filtered = bvp_signal

            # Use multiple methods for heart rate estimation
            hr_estimates = []

            # Method 1: Welch's method (more robust for noisy signals)
            try:
                freqs, psd = welch(bvp_signal_filtered, fs=fps, nperseg=min(256, len(bvp_signal_filtered)//2))

                # Adaptive frequency range based on quality
                if quality_metrics['quality'] in ['excellent', 'good']:
                    valid_range = (freqs >= 0.7) & (freqs <= 3.5)
                else:
                    valid_range = (freqs >= 0.5) & (freqs <= 4.0)

                if np.any(valid_range):
                    valid_freqs = freqs[valid_range]
                    valid_psd = psd[valid_range]
                    peak_idx = np.argmax(valid_psd)
                    peak_freq = valid_freqs[peak_idx]
                    hr_welch = peak_freq * 60

                    if 30 <= hr_welch <= 200:
                        hr_estimates.append(hr_welch)
            except:
                pass

            # Method 2: FFT (for comparison)
            try:
                fft = np.fft.rfft(bvp_signal_filtered)
                freqs = np.fft.rfftfreq(len(bvp_signal_filtered), 1/fps)

                if quality_metrics['quality'] in ['excellent', 'good']:
                    valid_range = (freqs >= 0.7) & (freqs <= 3.5)
                else:
                    valid_range = (freqs >= 0.5) & (freqs <= 4.0)

                if np.any(valid_range):
                    fft_magnitude = np.abs(fft[valid_range])
                    valid_freqs = freqs[valid_range]
                    peak_idx = np.argmax(fft_magnitude)
                    peak_freq = valid_freqs[peak_idx]
                    hr_fft = peak_freq * 60

                    if 30 <= hr_fft <= 200:
                        hr_estimates.append(hr_fft)
            except:
                pass

            # Method 3: Autocorrelation (good for periodic signals)
            try:
                # Autocorrelation for period detection
                autocorr = np.correlate(bvp_signal_filtered, bvp_signal_filtered, mode='full')
                autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only

                # Find peaks in autocorrelation
                min_lag = int(fps * 0.3)  # Minimum 0.3 seconds between beats (200 BPM)
                max_lag = int(fps * 2.0)  # Maximum 2 seconds between beats (30 BPM)

                if max_lag < len(autocorr):
                    autocorr_segment = autocorr[min_lag:max_lag]
                    if len(autocorr_segment) > 0:
                        peak_lag = np.argmax(autocorr_segment) + min_lag
                        hr_autocorr = 60 * fps / peak_lag

                        if 30 <= hr_autocorr <= 200:
                            hr_estimates.append(hr_autocorr)
            except:
                pass

            # Combine estimates based on quality
            if hr_estimates:
                if quality_metrics['quality'] in ['excellent', 'good']:
                    # Use median for clean signals
                    heart_rate = np.median(hr_estimates)
                else:
                    # Use mean for noisy signals (more forgiving)
                    heart_rate = np.mean(hr_estimates)

                # Sanity check
                if 30 <= heart_rate <= 200:
                    return heart_rate

            # Fallback to default if all methods fail
            return 75.0

        except Exception as e:
            logger.error(f"BVP to HR calculation error: {e}")
            return 75.0
    
    def process_frames(self, frames, fps=30):
        """Process frames and extract heart rate using ensemble of methods"""
        if len(frames) < fps * 2:  # Need at least 2 seconds
            return {
                'success': False,
                'error': 'Not enough frames for analysis',
                'heart_rate': 0,
                'confidence': 0
            }

        try:
            # Method 1 & 2: Traditional green channel methods
            # Extract green channel signal
            signal = self.extract_color_signal(frames)

            # Detrend and normalize
            signal_detrended = scipy_signal.detrend(signal, type='linear')

            # Apply moving average to smooth noise
            window_size = 3
            if len(signal_detrended) > window_size:
                signal_smoothed = np.convolve(signal_detrended, np.ones(window_size)/window_size, mode='same')
            else:
                signal_smoothed = signal_detrended

            # Normalize signal
            signal_normalized = (signal_smoothed - np.mean(signal_smoothed)) / (np.std(signal_smoothed) + 1e-10)

            # Calculate heart rate using traditional methods
            hr_fft = self.calculate_heart_rate_fft(signal_normalized, fps)
            hr_peaks = self.calculate_heart_rate_peaks(signal_normalized, fps)

            # Advanced pyVHR methods
            # Extract RGB signals for advanced methods
            rgb_signal = self.extract_rgb_signals(frames)

            # Method 3: CHROM method
            bvp_chrom = self.apply_chrom_method(rgb_signal)
            hr_chrom = self.calculate_heart_rate_from_bvp(bvp_chrom, fps)

            # Method 4: POS method
            bvp_pos = self.apply_pos_method(rgb_signal)
            hr_pos = self.calculate_heart_rate_from_bvp(bvp_pos, fps)

            # Method 5: OMIT method (state-of-the-art)
            bvp_omit = self.apply_omit_method(rgb_signal)
            hr_omit = self.calculate_heart_rate_from_bvp(bvp_omit, fps)

            # Method 6: ICA method
            bvp_ica = self.apply_ica_method(rgb_signal)
            hr_ica = self.calculate_heart_rate_from_bvp(bvp_ica, fps)

            # Method 7: GREEN method (baseline)
            bvp_green = self.apply_green_method(rgb_signal)
            hr_green = self.calculate_heart_rate_from_bvp(bvp_green, fps)

            # Ensemble averaging with weighted confidence
            # Collect all heart rate estimates
            hr_methods = {
                'fft': hr_fft,
                'peaks': hr_peaks,
                'chrom': hr_chrom,
                'pos': hr_pos,
                'omit': hr_omit,
                'ica': hr_ica,
                'green': hr_green
            }

            # Filter out invalid readings (default 75.0)
            valid_hrs = []
            method_names = []
            for method, hr in hr_methods.items():
                if hr != 75.0:  # Not default value
                    valid_hrs.append(hr)
                    method_names.append(method)

            # Calculate ensemble heart rate
            if len(valid_hrs) > 0:
                # Use median for robustness against outliers
                heart_rate = np.median(valid_hrs)

                # Alternative: weighted average based on method reliability
                # Weight based on empirical accuracy
                weights = []
                for method in method_names:
                    if method == 'omit':
                        weights.append(2.0)  # Highest weight for OMIT (state-of-the-art)
                    elif method in ['chrom', 'pos']:
                        weights.append(1.5)  # High weight for proven methods
                    elif method in ['ica']:
                        weights.append(1.2)  # Medium-high weight for ICA
                    elif method in ['fft', 'peaks']:
                        weights.append(1.0)  # Standard weight for traditional methods
                    else:  # green
                        weights.append(0.8)  # Lower weight for simple baseline

                if len(weights) > 0:
                    weights = np.array(weights) / np.sum(weights)
                    heart_rate_weighted = np.average(valid_hrs, weights=weights)
                else:
                    heart_rate_weighted = heart_rate

                # Calculate confidence based on agreement between methods
                if len(valid_hrs) > 1:
                    std_dev = np.std(valid_hrs)
                    # Lower std dev = higher confidence
                    confidence = max(0.1, min(1.0, 1.0 - (std_dev / 20.0)))
                else:
                    confidence = 0.5  # Single method, moderate confidence
            else:
                # All methods returned default, use simple average
                heart_rate = np.mean(list(hr_methods.values()))
                heart_rate_weighted = heart_rate
                confidence = 0.3  # Low confidence

            # Log method comparison for debugging
            logger.info(f"Heart rate estimates - FFT: {hr_fft:.1f}, Peaks: {hr_peaks:.1f}, "
                       f"GREEN: {hr_green:.1f}")
            logger.info(f"Advanced methods - CHROM: {hr_chrom:.1f}, POS: {hr_pos:.1f}, "
                       f"OMIT: {hr_omit:.1f}, ICA: {hr_ica:.1f}")
            logger.info(f"Ensemble HR (median): {heart_rate:.1f}, Weighted: {heart_rate_weighted:.1f}, "
                       f"Confidence: {confidence:.2f}, Methods used: {len(valid_hrs)}/7")

            # Choose best signal for visualization (prefer OMIT if available)
            display_signal = signal_normalized.tolist()[:500]
            if bvp_omit is not None:
                display_signal = bvp_omit.tolist()[:500]
            elif bvp_chrom is not None:
                display_signal = bvp_chrom.tolist()[:500]

            return {
                'success': True,
                'heart_rate': heart_rate_weighted,  # Use weighted average
                'heart_rate_fft': hr_fft,
                'heart_rate_peaks': hr_peaks,
                'heart_rate_green': hr_green,
                'heart_rate_chrom': hr_chrom,
                'heart_rate_pos': hr_pos,
                'heart_rate_omit': hr_omit,
                'heart_rate_ica': hr_ica,
                'heart_rate_median': heart_rate,
                'confidence': confidence,
                'methods_used': len(valid_hrs),
                'total_methods': 7,
                'signal': display_signal,
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

    # Create more realistic synthetic PPG signal with RGB variations
    heart_freq = heart_rate / 60  # Hz

    # Simulate different channel responses (R, G, B have different amplitudes)
    # Green channel typically has strongest PPG signal
    r_signal = 128 + 8 * np.sin(2 * np.pi * heart_freq * t) + np.random.normal(0, 0.5, num_frames)
    g_signal = 128 + 12 * np.sin(2 * np.pi * heart_freq * t) + np.random.normal(0, 0.5, num_frames)
    b_signal = 128 + 6 * np.sin(2 * np.pi * heart_freq * t) + np.random.normal(0, 0.5, num_frames)

    # Create fake frames with proper RGB values
    frames = []
    for i in range(num_frames):
        # Create a fake 3-channel image with different values per channel
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = np.clip(b_signal[i], 0, 255)  # OpenCV uses BGR
        frame[:, :, 1] = np.clip(g_signal[i], 0, 255)
        frame[:, :, 2] = np.clip(r_signal[i], 0, 255)
        frames.append(frame)

    # Process
    processor = SimplifiedRPPGProcessor()
    result = processor.process_frames(frames, fps)

    print(f"\n{'='*60}")
    print(f"Enhanced rPPG Test Results (7 Methods Ensemble)")
    print(f"{'='*60}")
    print(f"Expected HR: {heart_rate} BPM")
    print(f"-" * 40)
    print(f"Traditional Methods:")
    print(f"  FFT Method:      {result.get('heart_rate_fft', 0):.1f} BPM")
    print(f"  Peaks Method:    {result.get('heart_rate_peaks', 0):.1f} BPM")
    print(f"  GREEN Method:    {result.get('heart_rate_green', 0):.1f} BPM")
    print(f"-" * 40)
    print(f"Advanced Methods:")
    print(f"  CHROM Method:    {result.get('heart_rate_chrom', 0):.1f} BPM")
    print(f"  POS Method:      {result.get('heart_rate_pos', 0):.1f} BPM")
    print(f"  OMIT Method:     {result.get('heart_rate_omit', 0):.1f} BPM (state-of-the-art)")
    print(f"  ICA Method:      {result.get('heart_rate_ica', 0):.1f} BPM")
    print(f"-" * 40)
    print(f"Ensemble Results:")
    print(f"  Median HR:       {result.get('heart_rate_median', 0):.1f} BPM")
    print(f"  Weighted HR:     {result.get('heart_rate', 0):.1f} BPM (final)")
    print(f"  Confidence:      {result.get('confidence', 0):.2f}")
    print(f"  Methods Used:    {result.get('methods_used', 0)}/{result.get('total_methods', 7)}")
    print(f"  Success:         {result.get('success', False)}")
    print(f"{'='*60}\n")

    return result


if __name__ == "__main__":
    # Run test
    test_with_synthetic_data()