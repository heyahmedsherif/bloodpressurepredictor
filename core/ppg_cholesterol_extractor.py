"""
Enhanced PPG Feature Extractor for Cholesterol (Total, LDL, HDL)
==================================================================
Extracts both basic and advanced PPG features for cholesterol prediction.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import find_peaks, peak_widths
import logging

logger = logging.getLogger(__name__)

class CholesterolFeatureExtractor:
    """Extract PPG features for both total and LDL/HDL cholesterol prediction"""

    def __init__(self):
        self.sampling_rate = 20  # Based on our camera capture rate

    def extract_all_features(self, ppg_signal, demographics=None):
        """
        Extract features for both original and new cholesterol models

        Args:
            ppg_signal: Array of PPG values
            demographics: Dict with 'age', 'gender', 'bmi' etc.

        Returns:
            Dict with basic and advanced features
        """
        features = {}

        # Ensure we have enough signal
        if len(ppg_signal) < self.sampling_rate * 2:
            logger.warning("PPG signal too short for feature extraction")
            return self._get_default_features()

        # Normalize signal
        ppg_signal = np.array(ppg_signal)
        ppg_signal = (ppg_signal - np.mean(ppg_signal)) / (np.std(ppg_signal) + 1e-10)

        # Apply bandpass filter
        filtered_signal = self._filter_signal(ppg_signal)

        # Find peaks (heartbeats)
        peaks, properties = find_peaks(filtered_signal,
                                     distance=self.sampling_rate*0.4,
                                     prominence=0.1)

        if len(peaks) > 2:
            # Basic features (for original model)
            features.update(self._extract_basic_features(filtered_signal, peaks, properties))

            # Advanced features (for LDL/HDL model)
            features.update(self._extract_advanced_features(filtered_signal, peaks, properties))
        else:
            logger.warning("Not enough peaks detected")
            return self._get_default_features()

        # Add demographics if provided
        if demographics:
            features['age'] = demographics.get('age', 45)
            features['sex'] = 1 if demographics.get('gender', 'M') == 'M' else 0
            features['bmi'] = demographics.get('bmi', 25)

        return features

    def _filter_signal(self, signal):
        """Apply bandpass filter to PPG signal"""
        nyquist = self.sampling_rate / 2
        low = 0.5 / nyquist
        high = min(4.0 / nyquist, 0.99)

        if low < high:
            b, a = scipy_signal.butter(4, [low, high], btype='band')
            filtered = scipy_signal.filtfilt(b, a, signal)
            return filtered
        return signal

    def _extract_basic_features(self, signal, peaks, properties):
        """Extract basic features for original cholesterol model"""
        features = {}

        # Heart rate
        duration = len(signal) / self.sampling_rate
        features['heart_rate'] = len(peaks) / duration * 60

        # Heart rate variability
        if len(peaks) > 1:
            intervals = np.diff(peaks) / self.sampling_rate
            features['hrv'] = np.std(intervals) * 1000  # Convert to ms
            features['ppg_variability'] = features['hrv'] / 1000  # For compatibility
        else:
            features['hrv'] = 50
            features['ppg_variability'] = 0.05

        # Amplitude features
        if 'prominences' in properties:
            features['ppg_amplitude'] = np.mean(properties['prominences'])
        else:
            features['ppg_amplitude'] = 0.5

        # Pulse width (basic)
        features['ppg_width'] = 0.3  # Default, will be updated if available

        return features

    def _extract_advanced_features(self, signal, peaks, properties):
        """Extract advanced features for LDL/HDL models"""
        features = {}

        # Amplitude statistics
        if 'prominences' in properties:
            features['mean_amplitude'] = np.mean(properties['prominences'])
            features['std_amplitude'] = np.std(properties['prominences'])
        else:
            features['mean_amplitude'] = 0.5
            features['std_amplitude'] = 0.1

        # Pulse width features
        try:
            widths, _, _, _ = peak_widths(signal, peaks, rel_height=0.5)
            features['mean_width'] = np.mean(widths) / self.sampling_rate
            features['std_width'] = np.std(widths) / self.sampling_rate
            features['ppg_width'] = features['mean_width']  # Update basic feature
        except:
            features['mean_width'] = 0.3
            features['std_width'] = 0.05

        # Augmentation Index (AI) - arterial stiffness indicator
        ai_values = []
        for i, peak_idx in enumerate(peaks[:-1]):
            try:
                # Look for dicrotic notch and peak
                segment = signal[peak_idx:peaks[i+1]]
                if len(segment) > 10:
                    # Find dicrotic notch (local minimum in latter part)
                    notch_search_start = len(segment) // 3
                    notch_idx = np.argmin(segment[notch_search_start:]) + notch_search_start

                    if notch_idx < len(segment) - 3:
                        # Find dicrotic peak after notch
                        dicrotic_peak = np.max(segment[notch_idx:])
                        systolic_peak = signal[peak_idx]

                        if systolic_peak > 0:
                            ai = dicrotic_peak / systolic_peak
                            ai_values.append(ai)
            except Exception as e:
                logger.debug(f"AI calculation error: {e}")
                continue

        if ai_values:
            features['mean_ai'] = np.mean(ai_values)
            features['std_ai'] = np.std(ai_values)
        else:
            # Default values based on typical ranges
            features['mean_ai'] = 0.4
            features['std_ai'] = 0.1

        # Stiffness Index (SI) - pulse wave velocity proxy
        rise_times = []
        for peak_idx in peaks:
            # Find pulse foot (start of upstroke)
            start_search = max(0, peak_idx - self.sampling_rate//2)
            segment = signal[start_search:peak_idx]

            if len(segment) > 0:
                foot_idx = np.argmin(segment)
                rise_time = (peak_idx - (start_search + foot_idx)) / self.sampling_rate
                if rise_time > 0:
                    rise_times.append(rise_time)

        if rise_times:
            # Stiffness index inversely related to rise time
            features['stiffness_index'] = 1.0 / np.mean(rise_times) * 10  # Scale factor
        else:
            features['stiffness_index'] = 10.0

        # Reflection Index - combined vascular indicator
        features['reflection_index'] = features['mean_ai'] * features['stiffness_index']

        return features

    def _get_default_features(self):
        """Return default features when extraction fails"""
        return {
            # Basic features (original model)
            'heart_rate': 75,
            'hrv': 50,
            'ppg_amplitude': 0.5,
            'ppg_width': 0.3,
            'ppg_variability': 0.05,

            # Advanced features (LDL/HDL models)
            'mean_amplitude': 0.5,
            'std_amplitude': 0.1,
            'mean_width': 0.3,
            'std_width': 0.05,
            'mean_ai': 0.4,
            'std_ai': 0.1,
            'stiffness_index': 10.0,
            'reflection_index': 4.0,

            # Demographics (defaults)
            'age': 45,
            'sex': 0,
            'bmi': 25
        }

    def calculate_risk_category(self, ldl, hdl):
        """
        Calculate cardiovascular risk based on LDL/HDL ratio

        Risk categories based on medical guidelines:
        - Optimal: < 2.0
        - Low: 2.0 - 2.5
        - Moderate: 2.5 - 3.5
        - High: > 3.5
        """
        if hdl <= 0:
            return "Unknown"

        ratio = ldl / hdl

        if ratio < 2.0:
            return "Optimal"
        elif ratio < 2.5:
            return "Low"
        elif ratio < 3.5:
            return "Moderate"
        else:
            return "High"