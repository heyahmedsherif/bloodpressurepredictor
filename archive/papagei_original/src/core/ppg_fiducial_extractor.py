"""
PPG Fiducial Feature Extractor for Cholesterol Prediction

This module implements comprehensive fiducial feature extraction from PPG signals
based on the 2025 research: "Non-invasive prediction of cholesterol levels from 
photoplethysmogram (PPG)-based features using machine learning techniques"

The extractor identifies key fiducial points and derives 150+ features that correlate
with cholesterol levels in blood.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PPGFiducialFeatureExtractor:
    """
    Comprehensive PPG fiducial point detection and feature extraction for cholesterol prediction.
    
    Based on research showing high cholesterol affects PPG morphology, particularly:
    - Systolic phase characteristics
    - Dicrotic notch properties  
    - Pulse wave velocity indicators
    - Signal amplitude and timing features
    """
    
    def __init__(self, sampling_rate: float = 250.0):
        """
        Initialize the fiducial feature extractor.
        
        Args:
            sampling_rate: PPG signal sampling rate in Hz
        """
        self.fs = sampling_rate
        self.feature_names = []
        self._initialize_feature_names()
    
    def preprocess_signal(self, ppg_signal: np.ndarray) -> np.ndarray:
        """
        Preprocess PPG signal using 4th-order Chebyshev type II filter.
        
        Research specifications: Cut-off frequencies 0.4-8 Hz to remove 
        high-frequency noise and low-frequency baseline wander.
        """
        # Design 4th-order Chebyshev Type II filter
        nyquist = self.fs / 2
        low_cutoff = 0.4 / nyquist
        high_cutoff = 8.0 / nyquist
        
        # Create bandpass filter
        sos = signal.cheby2(4, 20, [low_cutoff, high_cutoff], btype='band', output='sos')
        
        # Apply filter
        filtered_signal = signal.sosfilt(sos, ppg_signal)
        
        # Additional smoothing with Savitzky-Golay filter
        if len(filtered_signal) > 25:
            filtered_signal = savgol_filter(filtered_signal, window_length=25, polyorder=3)
        
        return filtered_signal
    
    def detect_fiducial_points(self, ppg_signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect key fiducial points in PPG signal.
        
        Fiducial points critical for cholesterol prediction:
        - Systolic peaks (S): Maximum amplitude points
        - Onset points (O): Beginning of each pulse
        - Dicrotic notches (N): Secondary peaks from aortic valve closure  
        - Diastolic peaks (D): Local maxima in diastolic phase
        - End points (E): End of pulse cycle
        """
        # Normalize signal
        normalized_signal = (ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)
        
        # 1. Detect systolic peaks (primary maxima)
        systolic_peaks, _ = find_peaks(
            normalized_signal, 
            height=0.3,  # Minimum height threshold
            distance=int(0.6 * self.fs),  # Minimum distance between peaks (60 BPM max)
            prominence=0.2
        )
        
        # 2. Detect onset points (pulse start)
        # First derivative to find steep upstrokes
        first_derivative = np.gradient(normalized_signal)
        onset_candidates, _ = find_peaks(first_derivative, height=0.1, distance=int(0.4 * self.fs))
        
        # Match onset points to systolic peaks
        onset_points = []
        for peak in systolic_peaks:
            # Find onset point before this peak
            candidates_before_peak = onset_candidates[onset_candidates < peak]
            if len(candidates_before_peak) > 0:
                onset_points.append(candidates_before_peak[-1])
        onset_points = np.array(onset_points)
        
        # 3. Detect dicrotic notches (secondary peaks)
        # Look for local minima after systolic peaks
        dicrotic_notches = []
        for i, peak in enumerate(systolic_peaks):
            # Search window after systolic peak
            search_start = peak + int(0.1 * self.fs)
            if i < len(systolic_peaks) - 1:
                search_end = min(peak + int(0.4 * self.fs), systolic_peaks[i + 1])
            else:
                search_end = min(peak + int(0.4 * self.fs), len(normalized_signal) - 1)
            
            if search_start < search_end:
                # Find local minimum (dicrotic notch)
                search_segment = normalized_signal[search_start:search_end]
                local_minima, _ = find_peaks(-search_segment, distance=int(0.05 * self.fs))
                if len(local_minima) > 0:
                    dicrotic_notches.append(search_start + local_minima[0])
        dicrotic_notches = np.array(dicrotic_notches)
        
        # 4. Detect diastolic peaks (after dicrotic notch)
        diastolic_peaks = []
        for i, notch in enumerate(dicrotic_notches):
            search_start = notch + int(0.05 * self.fs)
            if i < len(systolic_peaks) - 1:
                search_end = min(notch + int(0.3 * self.fs), systolic_peaks[i + 1])
            else:
                search_end = min(notch + int(0.3 * self.fs), len(normalized_signal) - 1)
            
            if search_start < search_end:
                search_segment = normalized_signal[search_start:search_end]
                local_maxima, _ = find_peaks(search_segment, height=0.1)
                if len(local_maxima) > 0:
                    diastolic_peaks.append(search_start + local_maxima[0])
        diastolic_peaks = np.array(diastolic_peaks)
        
        # 5. Detect end points (pulse cycle end)
        end_points = []
        for i in range(len(onset_points) - 1):
            end_points.append(onset_points[i + 1] - 1)
        if len(onset_points) > 0:
            # Last end point
            end_points.append(min(onset_points[-1] + int(1.0 * self.fs), len(normalized_signal) - 1))
        end_points = np.array(end_points)
        
        return {
            'systolic_peaks': systolic_peaks,
            'onset_points': onset_points,
            'dicrotic_notches': dicrotic_notches,
            'diastolic_peaks': diastolic_peaks,
            'end_points': end_points,
            'signal': normalized_signal
        }
    
    def extract_all_features(self, ppg_signal: np.ndarray, age: float) -> np.ndarray:
        """
        Extract all 150+ fiducial features from PPG signal plus age.
        
        Feature categories:
        1. Amplitude features (30)
        2. Temporal features (40) 
        3. Morphological features (35)
        4. Spectral features (25)
        5. Statistical features (20)
        6. Age feature (1)
        """
        # Preprocess signal
        filtered_signal = self.preprocess_signal(ppg_signal)
        
        # Detect fiducial points
        fiducial_points = self.detect_fiducial_points(filtered_signal)
        
        # Extract features by category
        features = []
        
        # 1. Amplitude features
        amplitude_features = self._extract_amplitude_features(fiducial_points)
        features.extend(amplitude_features)
        
        # 2. Temporal features
        temporal_features = self._extract_temporal_features(fiducial_points)
        features.extend(temporal_features)
        
        # 3. Morphological features
        morphological_features = self._extract_morphological_features(fiducial_points)
        features.extend(morphological_features)
        
        # 4. Spectral features
        spectral_features = self._extract_spectral_features(filtered_signal)
        features.extend(spectral_features)
        
        # 5. Statistical features  
        statistical_features = self._extract_statistical_features(filtered_signal)
        features.extend(statistical_features)
        
        # 6. Age feature
        features.append(age)
        
        return np.array(features)
    
    def _extract_amplitude_features(self, fiducial_points: Dict) -> List[float]:
        """Extract amplitude-based features (30 features)."""
        signal = fiducial_points['signal']
        systolic_peaks = fiducial_points['systolic_peaks']
        dicrotic_notches = fiducial_points['dicrotic_notches']
        diastolic_peaks = fiducial_points['diastolic_peaks']
        
        features = []
        
        if len(systolic_peaks) > 0:
            # Systolic peak amplitudes
            sys_amplitudes = signal[systolic_peaks]
            features.extend([
                np.mean(sys_amplitudes),           # Mean systolic amplitude
                np.std(sys_amplitudes),            # STD systolic amplitude  
                np.max(sys_amplitudes),            # Max systolic amplitude
                np.min(sys_amplitudes),            # Min systolic amplitude
                np.median(sys_amplitudes),         # Median systolic amplitude
            ])
            
            # Peak-to-peak amplitude variations
            if len(sys_amplitudes) > 1:
                amplitude_diffs = np.diff(sys_amplitudes)
                features.extend([
                    np.mean(amplitude_diffs),      # Mean amplitude variation
                    np.std(amplitude_diffs),       # STD amplitude variation
                    np.max(amplitude_diffs),       # Max amplitude variation
                    np.min(amplitude_diffs),       # Min amplitude variation
                ])
            else:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0] * 9)
        
        # Dicrotic notch amplitudes
        if len(dicrotic_notches) > 0:
            notch_amplitudes = signal[dicrotic_notches]
            features.extend([
                np.mean(notch_amplitudes),         # Mean notch amplitude
                np.std(notch_amplitudes),          # STD notch amplitude
                np.max(notch_amplitudes),          # Max notch amplitude
                np.min(notch_amplitudes),          # Min notch amplitude
            ])
            
            # Systolic-to-dicrotic amplitude ratio
            if len(systolic_peaks) > 0:
                sys_amplitudes = signal[systolic_peaks[:len(notch_amplitudes)]]
                amplitude_ratios = sys_amplitudes / (notch_amplitudes + 1e-8)
                features.extend([
                    np.mean(amplitude_ratios),     # Mean sys/dicrotic ratio
                    np.std(amplitude_ratios),      # STD sys/dicrotic ratio
                ])
            else:
                features.extend([0, 0])
        else:
            features.extend([0] * 6)
        
        # Diastolic peak amplitudes  
        if len(diastolic_peaks) > 0:
            dias_amplitudes = signal[diastolic_peaks]
            features.extend([
                np.mean(dias_amplitudes),          # Mean diastolic amplitude
                np.std(dias_amplitudes),           # STD diastolic amplitude
                np.max(dias_amplitudes),           # Max diastolic amplitude
                np.min(dias_amplitudes),           # Min diastolic amplitude
            ])
        else:
            features.extend([0] * 4)
        
        # Pulse amplitude (peak-to-valley)
        if len(systolic_peaks) > 1:
            pulse_amplitudes = []
            for i in range(len(systolic_peaks) - 1):
                start_idx = systolic_peaks[i]
                end_idx = systolic_peaks[i + 1]
                segment = signal[start_idx:end_idx]
                pulse_amp = np.max(segment) - np.min(segment)
                pulse_amplitudes.append(pulse_amp)
            
            if pulse_amplitudes:
                features.extend([
                    np.mean(pulse_amplitudes),     # Mean pulse amplitude
                    np.std(pulse_amplitudes),      # STD pulse amplitude
                    np.max(pulse_amplitudes),      # Max pulse amplitude
                    np.min(pulse_amplitudes),      # Min pulse amplitude
                    np.ptp(pulse_amplitudes),      # Range pulse amplitude
                ])
            else:
                features.extend([0] * 5)
        else:
            features.extend([0] * 5)
        
        # Ensure we have exactly 30 amplitude features
        while len(features) < 30:
            features.append(0.0)
        
        return features[:30]
    
    def _extract_temporal_features(self, fiducial_points: Dict) -> List[float]:
        """Extract time-domain features (40 features)."""
        systolic_peaks = fiducial_points['systolic_peaks']
        onset_points = fiducial_points['onset_points']
        dicrotic_notches = fiducial_points['dicrotic_notches']
        
        features = []
        
        # Heart rate and RR intervals
        if len(systolic_peaks) > 1:
            rr_intervals = np.diff(systolic_peaks) / self.fs  # Convert to seconds
            heart_rates = 60.0 / rr_intervals  # Convert to BPM
            
            features.extend([
                np.mean(heart_rates),              # Mean heart rate
                np.std(heart_rates),               # STD heart rate (HRV)
                np.max(heart_rates),               # Max heart rate
                np.min(heart_rates),               # Min heart rate
                np.median(heart_rates),            # Median heart rate
                np.ptp(heart_rates),               # Heart rate range
            ])
            
            # RR interval statistics
            features.extend([
                np.mean(rr_intervals),             # Mean RR interval
                np.std(rr_intervals),              # STD RR interval (RMSSD)
                np.max(rr_intervals),              # Max RR interval
                np.min(rr_intervals),              # Min RR interval
            ])
        else:
            features.extend([0] * 10)
        
        # Pulse width timing
        if len(onset_points) > 0 and len(systolic_peaks) > 0:
            pulse_widths = []
            min_length = min(len(onset_points), len(systolic_peaks))
            
            for i in range(min_length):
                if systolic_peaks[i] > onset_points[i]:
                    pulse_width = (systolic_peaks[i] - onset_points[i]) / self.fs
                    pulse_widths.append(pulse_width)
            
            if pulse_widths:
                features.extend([
                    np.mean(pulse_widths),         # Mean pulse width
                    np.std(pulse_widths),          # STD pulse width
                    np.max(pulse_widths),          # Max pulse width
                    np.min(pulse_widths),          # Min pulse width
                ])
            else:
                features.extend([0] * 4)
        else:
            features.extend([0] * 4)
        
        # Systolic rise time
        if len(onset_points) > 0 and len(systolic_peaks) > 0:
            rise_times = []
            min_length = min(len(onset_points), len(systolic_peaks))
            
            for i in range(min_length):
                if systolic_peaks[i] > onset_points[i]:
                    rise_time = (systolic_peaks[i] - onset_points[i]) / self.fs
                    rise_times.append(rise_time)
            
            if rise_times:
                features.extend([
                    np.mean(rise_times),           # Mean rise time
                    np.std(rise_times),            # STD rise time  
                    np.max(rise_times),            # Max rise time
                    np.min(rise_times),            # Min rise time
                ])
            else:
                features.extend([0] * 4)
        else:
            features.extend([0] * 4)
        
        # Dicrotic notch timing
        if len(systolic_peaks) > 0 and len(dicrotic_notches) > 0:
            notch_delays = []
            min_length = min(len(systolic_peaks), len(dicrotic_notches))
            
            for i in range(min_length):
                if dicrotic_notches[i] > systolic_peaks[i]:
                    delay = (dicrotic_notches[i] - systolic_peaks[i]) / self.fs
                    notch_delays.append(delay)
            
            if notch_delays:
                features.extend([
                    np.mean(notch_delays),         # Mean dicrotic delay
                    np.std(notch_delays),          # STD dicrotic delay
                    np.max(notch_delays),          # Max dicrotic delay
                    np.min(notch_delays),          # Min dicrotic delay
                ])
            else:
                features.extend([0] * 4)
        else:
            features.extend([0] * 4)
        
        # Pulse cycle timing features
        if len(systolic_peaks) > 2:
            cycle_times = np.diff(systolic_peaks) / self.fs
            features.extend([
                np.mean(cycle_times),              # Mean cycle time
                np.std(cycle_times),               # STD cycle time
                np.max(cycle_times),               # Max cycle time
                np.min(cycle_times),               # Min cycle time
                np.ptp(cycle_times),               # Range cycle time
                len(systolic_peaks) / (len(fiducial_points['signal']) / self.fs),  # Pulse rate
            ])
        else:
            features.extend([0] * 6)
        
        # Additional temporal variability measures
        if len(systolic_peaks) > 1:
            peak_intervals = np.diff(systolic_peaks)
            if len(peak_intervals) > 1:
                # Successive difference statistics
                successive_diffs = np.diff(peak_intervals)
                features.extend([
                    np.mean(successive_diffs),     # Mean successive difference
                    np.std(successive_diffs),      # STD successive difference
                    np.mean(np.abs(successive_diffs)), # Mean absolute successive difference
                    np.sqrt(np.mean(successive_diffs**2)), # RMSSD
                ])
            else:
                features.extend([0] * 4)
        else:
            features.extend([0] * 4)
        
        # Ensure exactly 40 temporal features
        while len(features) < 40:
            features.append(0.0)
        
        return features[:40]
    
    def _extract_morphological_features(self, fiducial_points: Dict) -> List[float]:
        """Extract pulse shape and morphology features (35 features)."""
        signal = fiducial_points['signal']
        systolic_peaks = fiducial_points['systolic_peaks']
        onset_points = fiducial_points['onset_points']
        
        features = []
        
        # Pulse shape ratios and slopes
        if len(onset_points) > 0 and len(systolic_peaks) > 0:
            shape_features = []
            min_length = min(len(onset_points), len(systolic_peaks))
            
            for i in range(min_length):
                if systolic_peaks[i] > onset_points[i]:
                    # Extract single pulse
                    start_idx = onset_points[i]
                    peak_idx = systolic_peaks[i]
                    
                    # Find end of pulse (next onset or fixed duration)
                    if i < len(onset_points) - 1:
                        end_idx = onset_points[i + 1]
                    else:
                        end_idx = min(start_idx + int(1.2 * self.fs), len(signal))
                    
                    pulse = signal[start_idx:end_idx]
                    peak_pos = peak_idx - start_idx
                    
                    if len(pulse) > 10 and peak_pos > 0 and peak_pos < len(pulse):
                        # Upstroke and downstroke slopes
                        upstroke = pulse[:peak_pos]
                        downstroke = pulse[peak_pos:]
                        
                        if len(upstroke) > 1:
                            upstroke_slope = np.mean(np.diff(upstroke))
                        else:
                            upstroke_slope = 0
                        
                        if len(downstroke) > 1:
                            downstroke_slope = np.mean(np.diff(downstroke))
                        else:
                            downstroke_slope = 0
                        
                        # Asymmetry ratio
                        asymmetry = len(upstroke) / len(pulse) if len(pulse) > 0 else 0
                        
                        # Pulse area and width ratios
                        pulse_area = np.trapz(pulse)
                        pulse_width = len(pulse) / self.fs
                        
                        # Skewness and kurtosis of pulse shape
                        if np.std(pulse) > 0:
                            pulse_skewness = np.mean(((pulse - np.mean(pulse)) / np.std(pulse)) ** 3)
                            pulse_kurtosis = np.mean(((pulse - np.mean(pulse)) / np.std(pulse)) ** 4)
                        else:
                            pulse_skewness = 0
                            pulse_kurtosis = 0
                        
                        shape_features.append([
                            upstroke_slope, downstroke_slope, asymmetry,
                            pulse_area, pulse_width, pulse_skewness, pulse_kurtosis
                        ])
            
            if shape_features:
                shape_features = np.array(shape_features)
                # Statistical measures across all pulses
                for feat_idx in range(shape_features.shape[1]):
                    feat_values = shape_features[:, feat_idx]
                    features.extend([
                        np.mean(feat_values),
                        np.std(feat_values),
                        np.max(feat_values),
                        np.min(feat_values),
                    ])
            else:
                features.extend([0] * 28)  # 7 features × 4 statistics
        else:
            features.extend([0] * 28)
        
        # Signal complexity measures
        # Approximate entropy (ApEn)
        try:
            apen = self._approximate_entropy(signal, m=2, r=0.2)
            features.append(apen)
        except:
            features.append(0.0)
        
        # Sample entropy (SampEn) 
        try:
            sampen = self._sample_entropy(signal, m=2, r=0.2)
            features.append(sampen)
        except:
            features.append(0.0)
        
        # Signal energy and power
        signal_energy = np.sum(signal ** 2)
        signal_power = np.mean(signal ** 2)
        features.extend([signal_energy, signal_power])
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.signbit(signal)))
        zero_crossing_rate = zero_crossings / len(signal)
        features.append(zero_crossing_rate)
        
        # Signal variability measures
        features.extend([
            np.var(signal),                    # Variance
            np.ptp(signal),                    # Peak-to-peak range
        ])
        
        # Ensure exactly 35 morphological features
        while len(features) < 35:
            features.append(0.0)
        
        return features[:35]
    
    def _extract_spectral_features(self, ppg_signal: np.ndarray) -> List[float]:
        """Extract frequency domain features (25 features)."""
        features = []
        
        # Compute power spectral density
        freqs, psd = signal.welch(ppg_signal, fs=self.fs, nperseg=min(256, len(ppg_signal)//4))
        
        # Frequency bands relevant to PPG
        vlf_band = (freqs >= 0.0033) & (freqs <= 0.04)    # Very low frequency
        lf_band = (freqs >= 0.04) & (freqs <= 0.15)       # Low frequency  
        hf_band = (freqs >= 0.15) & (freqs <= 0.4)        # High frequency
        cardiac_band = (freqs >= 0.8) & (freqs <= 3.0)    # Cardiac frequency
        
        # Band power features
        vlf_power = np.trapz(psd[vlf_band], freqs[vlf_band]) if np.any(vlf_band) else 0
        lf_power = np.trapz(psd[lf_band], freqs[lf_band]) if np.any(lf_band) else 0  
        hf_power = np.trapz(psd[hf_band], freqs[hf_band]) if np.any(hf_band) else 0
        cardiac_power = np.trapz(psd[cardiac_band], freqs[cardiac_band]) if np.any(cardiac_band) else 0
        total_power = np.trapz(psd, freqs)
        
        features.extend([
            vlf_power, lf_power, hf_power, cardiac_power, total_power
        ])
        
        # Normalized band powers
        if total_power > 0:
            features.extend([
                vlf_power / total_power,
                lf_power / total_power, 
                hf_power / total_power,
                cardiac_power / total_power,
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # LF/HF ratio (autonomic balance)
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
        features.append(lf_hf_ratio)
        
        # Spectral centroids and spread
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0
        spectral_rolloff = freqs[np.where(np.cumsum(psd) >= 0.95 * np.sum(psd))[0][0]] if len(freqs) > 0 else 0
        
        features.extend([spectral_centroid, spectral_spread, spectral_rolloff])
        
        # Peak frequency and power
        peak_freq_idx = np.argmax(psd)
        peak_frequency = freqs[peak_freq_idx]
        peak_power = psd[peak_freq_idx]
        
        features.extend([peak_frequency, peak_power])
        
        # Spectral entropy
        normalized_psd = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        spectral_entropy = -np.sum(normalized_psd * np.log2(normalized_psd + 1e-15))
        features.append(spectral_entropy)
        
        # Additional frequency domain measures
        features.extend([
            np.std(psd),                       # PSD standard deviation
            np.max(psd),                       # Maximum PSD value
            np.min(psd),                       # Minimum PSD value
            np.mean(psd),                      # Mean PSD value
            np.median(psd),                    # Median PSD value
        ])
        
        # Ensure exactly 25 spectral features
        while len(features) < 25:
            features.append(0.0)
        
        return features[:25]
    
    def _extract_statistical_features(self, ppg_signal: np.ndarray) -> List[float]:
        """Extract statistical features (20 features)."""
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(ppg_signal),               # Mean
            np.std(ppg_signal),                # Standard deviation
            np.var(ppg_signal),                # Variance
            np.median(ppg_signal),             # Median
            np.max(ppg_signal),                # Maximum
            np.min(ppg_signal),                # Minimum
            np.ptp(ppg_signal),                # Range (peak-to-peak)
        ])
        
        # Distribution shape
        if np.std(ppg_signal) > 0:
            skewness = np.mean(((ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)) ** 3)
            kurtosis = np.mean(((ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)) ** 4)
        else:
            skewness = 0
            kurtosis = 0
        
        features.extend([skewness, kurtosis])
        
        # Percentiles
        features.extend([
            np.percentile(ppg_signal, 10),     # 10th percentile
            np.percentile(ppg_signal, 25),     # 25th percentile  
            np.percentile(ppg_signal, 75),     # 75th percentile
            np.percentile(ppg_signal, 90),     # 90th percentile
        ])
        
        # Interquartile range
        iqr = np.percentile(ppg_signal, 75) - np.percentile(ppg_signal, 25)
        features.append(iqr)
        
        # Root mean square
        rms = np.sqrt(np.mean(ppg_signal ** 2))
        features.append(rms)
        
        # Mean absolute deviation
        mad = np.mean(np.abs(ppg_signal - np.mean(ppg_signal)))
        features.append(mad)
        
        # Signal length and sampling statistics
        features.extend([
            len(ppg_signal),                   # Signal length
            len(ppg_signal) / self.fs,         # Signal duration
        ])
        
        # Additional robust statistics
        features.extend([
            np.mean(np.abs(ppg_signal)),       # Mean absolute value
            np.sum(np.abs(ppg_signal)),        # Sum absolute value
        ])
        
        # Ensure exactly 20 statistical features
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    def _approximate_entropy(self, signal: np.ndarray, m: int, r: float) -> float:
        """Calculate approximate entropy of the signal."""
        N = len(signal)
        
        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([signal[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            
            for i in range(N - m + 1):
                template = patterns[i]
                matches = [1 for pattern in patterns if _maxdist(template, pattern) <= r]
                C[i] = len(matches) / float(N - m + 1.0)
            
            phi = np.mean([np.log(c) for c in C if c > 0])
            return phi
        
        try:
            return _phi(m) - _phi(m + 1)
        except:
            return 0.0
    
    def _sample_entropy(self, signal: np.ndarray, m: int, r: float) -> float:
        """Calculate sample entropy of the signal."""
        N = len(signal)
        
        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([signal[i:i + m] for i in range(N - m + 1)])
            matches = 0
            comparisons = 0
            
            for i in range(len(patterns)):
                for j in range(i + 1, len(patterns)):
                    comparisons += 1
                    if _maxdist(patterns[i], patterns[j]) <= r:
                        matches += 1
            
            return matches / float(comparisons) if comparisons > 0 else 0
        
        try:
            A = _phi(m)
            B = _phi(m + 1)
            return -np.log(B / A) if A > 0 and B > 0 else 0.0
        except:
            return 0.0
    
    def _initialize_feature_names(self):
        """Initialize feature names for interpretability."""
        # This would contain all 150+ feature names
        # Abbreviated for space - in practice would have descriptive names
        categories = [
            ['amp_' + str(i) for i in range(30)],      # Amplitude features
            ['temp_' + str(i) for i in range(40)],     # Temporal features  
            ['morph_' + str(i) for i in range(35)],    # Morphological features
            ['spec_' + str(i) for i in range(25)],     # Spectral features
            ['stat_' + str(i) for i in range(20)],     # Statistical features
            ['age']                                     # Age feature
        ]
        
        self.feature_names = []
        for category in categories:
            self.feature_names.extend(category)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return self.feature_names.copy()
    
    def get_feature_count(self) -> int:
        """Get total number of features extracted."""
        return len(self.feature_names)