"""
PPG Feature Extractor for ML Model Inputs
Extracts features from PPG signals for health prediction models.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)

class PPGFeatureExtractor:
    """Extract features from PPG signal for ML models."""
    
    def __init__(self, fps=30):
        self.fps = fps
        
    def extract_features(self, ppg_signal, heart_rate=None):
        """
        Extract PPG features for ML model inputs.
        
        Args:
            ppg_signal: Raw PPG signal array
            heart_rate: Pre-calculated heart rate (optional)
            
        Returns:
            Dictionary of features for ML models
        """
        features = {}
        
        try:
            # Ensure signal is numpy array
            if not isinstance(ppg_signal, np.ndarray):
                ppg_signal = np.array(ppg_signal)
            
            # Normalize signal
            ppg_norm = (ppg_signal - np.mean(ppg_signal)) / (np.std(ppg_signal) + 1e-10)
            
            # 1. PPG Amplitude (peak-to-peak)
            features['ppg_amplitude'] = np.ptp(ppg_norm)
            
            # 2. Heart Rate (if not provided, calculate it)
            if heart_rate is None:
                heart_rate = self._calculate_heart_rate(ppg_norm)
            features['heart_rate'] = heart_rate
            
            # 3. Pulse Width (average width of pulses)
            features['ppg_width'] = self._calculate_pulse_width(ppg_norm)
            
            # 4. PPG Variability (similar to HRV)
            features['ppg_variability'] = self._calculate_ppg_variability(ppg_norm)
            
            # 5. Pulse Transit Time approximation (from PPG morphology)
            features['pulse_transit_time'] = self._estimate_ptt(ppg_norm)
            
            # 6. Systolic and Diastolic peaks (from PPG waveform)
            systolic, diastolic = self._estimate_bp_features(ppg_norm, heart_rate)
            features['systolic'] = systolic
            features['diastolic'] = diastolic
            
            logger.info(f"Extracted PPG features: {features}")
            
        except Exception as e:
            logger.error(f"Error extracting PPG features: {e}")
            # Return default features if extraction fails
            features = {
                'ppg_amplitude': 1.0,
                'heart_rate': heart_rate or 75,
                'ppg_width': 0.35,
                'ppg_variability': 0.2,
                'pulse_transit_time': 0.25,
                'systolic': 120,
                'diastolic': 80
            }
        
        return features
    
    def _calculate_heart_rate(self, signal):
        """Calculate heart rate from PPG signal using FFT."""
        try:
            # Apply bandpass filter
            nyquist = self.fps / 2
            low = 0.75 / nyquist
            high = min(4.0 / nyquist, 0.99)
            
            if low < high:
                b, a = scipy_signal.butter(2, [low, high], btype='band')
                filtered = scipy_signal.filtfilt(b, a, signal)
            else:
                filtered = signal
            
            # FFT
            fft = np.fft.rfft(filtered)
            freqs = np.fft.rfftfreq(len(filtered), 1/self.fps)
            
            # Find peak in physiological range
            valid_range = (freqs >= 0.75) & (freqs <= 4.0)
            if np.any(valid_range):
                fft_magnitude = np.abs(fft[valid_range])
                valid_freqs = freqs[valid_range]
                peak_idx = np.argmax(fft_magnitude)
                peak_freq = valid_freqs[peak_idx]
                heart_rate = peak_freq * 60
                
                if 45 <= heart_rate <= 180:
                    return heart_rate
            
            return 75.0  # Default
            
        except Exception as e:
            logger.error(f"HR calculation error: {e}")
            return 75.0
    
    def _calculate_pulse_width(self, signal):
        """Calculate average pulse width."""
        try:
            # Find peaks
            peaks, properties = find_peaks(signal, distance=int(self.fps*0.4), 
                                         prominence=0.3)
            
            if len(peaks) > 1:
                # Calculate widths at half prominence
                widths = scipy_signal.peak_widths(signal, peaks, rel_height=0.5)
                avg_width = np.mean(widths[0]) / self.fps  # Convert to seconds
                return min(0.5, max(0.2, avg_width))  # Clamp to reasonable range
            
            return 0.35  # Default
            
        except Exception as e:
            logger.error(f"Pulse width calculation error: {e}")
            return 0.35
    
    def _calculate_ppg_variability(self, signal):
        """Calculate PPG variability (similar to HRV)."""
        try:
            # Find peaks
            peaks, _ = find_peaks(signal, distance=int(self.fps*0.4))
            
            if len(peaks) > 2:
                # Calculate intervals between peaks
                intervals = np.diff(peaks) / self.fps  # Convert to seconds
                # Calculate standard deviation of intervals
                variability = np.std(intervals)
                return min(0.5, max(0.1, variability))  # Clamp to reasonable range
            
            return 0.2  # Default
            
        except Exception as e:
            logger.error(f"PPG variability calculation error: {e}")
            return 0.2
    
    def _estimate_ptt(self, signal):
        """Estimate pulse transit time from PPG morphology."""
        try:
            # Find peaks and troughs
            peaks, _ = find_peaks(signal, distance=int(self.fps*0.4))
            troughs, _ = find_peaks(-signal, distance=int(self.fps*0.4))
            
            if len(peaks) > 0 and len(troughs) > 0:
                # Average time from trough to peak (simplified PTT proxy)
                ptt_samples = []
                for peak in peaks:
                    # Find nearest preceding trough
                    preceding_troughs = troughs[troughs < peak]
                    if len(preceding_troughs) > 0:
                        nearest_trough = preceding_troughs[-1]
                        ptt = (peak - nearest_trough) / self.fps
                        if 0.1 < ptt < 0.5:  # Reasonable range
                            ptt_samples.append(ptt)
                
                if ptt_samples:
                    return np.mean(ptt_samples)
            
            return 0.25  # Default
            
        except Exception as e:
            logger.error(f"PTT estimation error: {e}")
            return 0.25
    
    def _estimate_bp_features(self, signal, heart_rate):
        """
        Estimate blood pressure features from PPG morphology.
        Note: This is a simplified estimation for research purposes.
        """
        try:
            # Systolic estimation based on PPG amplitude and HR
            # Higher amplitude generally correlates with lower BP
            amplitude = np.ptp(signal)
            systolic = 120 + (heart_rate - 70) * 0.3 - amplitude * 5
            systolic = np.clip(systolic, 90, 180)
            
            # Diastolic estimation
            diastolic = 80 + (heart_rate - 70) * 0.2 - amplitude * 3
            diastolic = np.clip(diastolic, 60, 110)
            
            return systolic, diastolic
            
        except Exception as e:
            logger.error(f"BP feature estimation error: {e}")
            return 120, 80