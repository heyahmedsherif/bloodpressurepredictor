"""
PPG Processor - Signal processing utilities
"""
import numpy as np
from scipy import signal


class PPGProcessor:
    """PPG signal processing utilities."""
    
    @staticmethod
    def extract_heart_rate(ppg_signal, fps=30.0):
        """Extract heart rate from PPG signal using FFT."""
        if len(ppg_signal) < fps:
            return 75.0, 0.1  # Default values
        
        # Normalize signal
        signal_array = np.array(ppg_signal)
        normalized = (signal_array - np.mean(signal_array)) / np.std(signal_array)
        
        # Bandpass filter (0.5-4 Hz for 30-240 BPM)
        nyquist = fps / 2
        low = 0.5 / nyquist
        high = 4.0 / nyquist
        
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, normalized)
            
            # FFT analysis
            fft = np.fft.fft(filtered)
            freqs = np.fft.fftfreq(len(filtered), 1/fps)
            
            # Find peak in valid frequency range
            valid_idx = (freqs > 0.5) & (freqs < 4.0)
            if np.any(valid_idx):
                valid_freqs = freqs[valid_idx]
                valid_fft = np.abs(fft[valid_idx])
                
                peak_idx = np.argmax(valid_fft)
                heart_rate_hz = valid_freqs[peak_idx]
                heart_rate_bpm = heart_rate_hz * 60
                
                # Calculate confidence
                peak_power = valid_fft[peak_idx]
                avg_power = np.mean(valid_fft)
                confidence = min(1.0, peak_power / (avg_power * 3))
                
                return float(heart_rate_bpm), float(confidence)
        
        except Exception as e:
            print(f"PPG processing error: {e}")
        
        return 75.0, 0.1  # Fallback values