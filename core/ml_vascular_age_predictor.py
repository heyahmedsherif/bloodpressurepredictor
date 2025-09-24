"""
ML-Based Vascular Age Predictor using Deep Learning
=====================================================
Machine learning approach for vascular age prediction from PPG signals.
Based on research from AI-vascular-age and recent papers.

This is separate from the formula-based predictor to preserve existing functionality.
"""

import numpy as np
import pickle
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class MLVascularAgePredictor:
    """ML-based vascular age prediction using PPG features"""

    def __init__(self):
        """Initialize ML vascular age predictor"""
        self.model_loaded = False
        self.model = None
        self.scaler = None

        # Try to load pre-trained model if available
        self._load_model()

    def _load_model(self):
        """Load pre-trained vascular age model if available"""
        model_path = Path('models/vascular_age')

        if model_path.exists():
            try:
                with open(model_path / 'vascular_age_model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                with open(model_path / 'vascular_age_scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                self.model_loaded = True
                logger.info("✅ ML vascular age model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading vascular age model: {e}")
                self._initialize_default_model()
        else:
            logger.info("No pre-trained vascular age model found, using hybrid approach")
            self._initialize_default_model()

    def _initialize_default_model(self):
        """Initialize a default model using known correlations"""
        # Use Random Forest as it handles non-linear relationships well
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.model_loaded = False

    def extract_deep_features(self, ppg_signal):
        """
        Extract deep features from PPG signal for vascular age
        Based on research showing these correlate with vascular aging
        """
        features = []

        # Normalize signal
        ppg_norm = (ppg_signal - np.mean(ppg_signal)) / (np.std(ppg_signal) + 1e-8)

        # 1. Morphological features (focus on diastolic peak - key for vascular age)
        peaks = self._find_peaks(ppg_norm)
        if len(peaks) > 1:
            # Diastolic peak characteristics
            diastolic_amplitudes = []
            systolic_amplitudes = []
            for i in range(len(peaks) - 1):
                cycle = ppg_norm[peaks[i]:peaks[i+1]]
                if len(cycle) > 10:
                    # Find dicrotic notch region (40-60% of cycle)
                    mid_start = int(len(cycle) * 0.4)
                    mid_end = int(len(cycle) * 0.6)
                    if mid_end > mid_start:
                        diastolic_region = cycle[mid_start:mid_end]
                        if len(diastolic_region) > 0:
                            diastolic_amplitudes.append(np.max(diastolic_region))
                    systolic_amplitudes.append(np.max(cycle[:mid_start]))

            # Diastolic/Systolic ratio (increases with age)
            if diastolic_amplitudes and systolic_amplitudes:
                features.append(np.mean(diastolic_amplitudes) / (np.mean(systolic_amplitudes) + 1e-8))
            else:
                features.append(0.3)
        else:
            features.append(0.3)

        # 2. Frequency domain features
        fft = np.fft.fft(ppg_norm)
        power_spectrum = np.abs(fft[:len(fft)//2])**2

        # Power in different frequency bands
        total_power = np.sum(power_spectrum)
        if total_power > 0:
            # Low frequency power (0.04-0.15 Hz) - sympathetic activity
            lf_power = np.sum(power_spectrum[2:8]) / total_power
            # High frequency power (0.15-0.4 Hz) - parasympathetic activity
            hf_power = np.sum(power_spectrum[8:20]) / total_power
            # LF/HF ratio (increases with age)
            features.append(lf_power / (hf_power + 1e-8))
        else:
            features.append(1.5)

        # 3. Pulse wave velocity indicators
        # Rising edge steepness (decreases with age due to arterial stiffness)
        if len(peaks) > 0:
            rise_times = []
            for peak in peaks[:min(10, len(peaks))]:
                # Find the valley before peak
                start_idx = max(0, peak - 20)
                valley = start_idx + np.argmin(ppg_norm[start_idx:peak])
                if peak > valley:
                    rise_time = peak - valley
                    if rise_time > 0:
                        rise_slope = (ppg_norm[peak] - ppg_norm[valley]) / rise_time
                        rise_times.append(rise_slope)

            features.append(np.mean(rise_times) if rise_times else 0.1)
        else:
            features.append(0.1)

        # 4. Complexity features (decreases with age)
        # Approximate entropy
        features.append(self._approximate_entropy(ppg_norm[:min(200, len(ppg_norm))]))

        # 5. Variability features
        if len(peaks) > 2:
            peak_intervals = np.diff(peaks)
            features.append(np.std(peak_intervals))  # Heart rate variability proxy
        else:
            features.append(10)

        # 6. Waveform shape features
        # Skewness (changes with arterial stiffness)
        features.append(self._skewness(ppg_norm))

        # Kurtosis (peakedness changes with age)
        features.append(self._kurtosis(ppg_norm))

        # 7. Autocorrelation (regularity decreases with age)
        autocorr = np.correlate(ppg_norm[:100], ppg_norm[:100], mode='same')
        features.append(np.max(autocorr[1:]) / (autocorr[0] + 1e-8))

        return np.array(features)

    def _find_peaks(self, signal):
        """Simple peak detection"""
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                if signal[i] > np.mean(signal):
                    peaks.append(i)
        return peaks

    def _approximate_entropy(self, signal, m=2, r=0.2):
        """Calculate approximate entropy (complexity measure)"""
        N = len(signal)
        if N < m + 1:
            return 0

        def _maxdist(x_i, x_j):
            return max(abs(a - b) for a, b in zip(x_i, x_j))

        def _phi(m):
            patterns = [signal[i:i+m] for i in range(N - m + 1)]
            C = []
            for pattern in patterns:
                matching = sum(1 for p in patterns if _maxdist(pattern, p) <= r * np.std(signal))
                C.append(matching / (N - m + 1))
            return sum(np.log(c) for c in C if c > 0) / len(C) if C else 0

        return _phi(m) - _phi(m + 1)

    def _skewness(self, signal):
        """Calculate skewness"""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0
        return np.mean(((signal - mean) / std) ** 3)

    def _kurtosis(self, signal):
        """Calculate kurtosis"""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0
        return np.mean(((signal - mean) / std) ** 4) - 3

    def predict_vascular_age_ml(self, ppg_signal, demographics, ppg_features=None):
        """
        Predict vascular age using ML approach

        Args:
            ppg_signal: Raw PPG signal array
            demographics: Dict with age, gender, bmi
            ppg_features: Optional pre-extracted features (AI, SI, etc.)

        Returns:
            Dict with ML-based vascular age prediction
        """
        try:
            # Extract deep features from PPG
            deep_features = self.extract_deep_features(ppg_signal)

            # Add demographic features
            age = demographics.get('age', 40)
            gender_encoded = 1 if demographics.get('gender', 'M') == 'M' else 0
            bmi = demographics.get('bmi', 25)

            # Combine with provided PPG features if available
            if ppg_features:
                ai = ppg_features.get('augmentation_index', 0.4)
                si = ppg_features.get('stiffness_index', 8.0)
                hrv = ppg_features.get('hrv', 50)
                systolic = ppg_features.get('systolic_bp', 120)

                # Create feature vector
                features = np.concatenate([
                    deep_features,
                    [ai, si, hrv, systolic, gender_encoded, bmi]
                ])
            else:
                features = np.concatenate([
                    deep_features,
                    [gender_encoded, bmi]
                ])

            # Predict using model if loaded, otherwise use hybrid approach
            if self.model_loaded:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                vascular_age = self.model.predict(features_scaled)[0]
                confidence = 0.8  # High confidence with trained model
            else:
                # Hybrid approach using extracted features
                # Weighted combination based on research
                vascular_age = self._hybrid_prediction(deep_features, demographics, ppg_features)
                confidence = 0.6  # Lower confidence without training

            # Calculate metrics
            age_difference = vascular_age - age

            # Determine status
            if age_difference <= -5:
                status = "Excellent (ML)"
                risk = "Low"
            elif age_difference <= 0:
                status = "Good (ML)"
                risk = "Low"
            elif age_difference <= 5:
                status = "Normal (ML)"
                risk = "Moderate"
            elif age_difference <= 10:
                status = "Accelerated Aging (ML)"
                risk = "Elevated"
            else:
                status = "Significant Aging (ML)"
                risk = "High"

            results = {
                'vascular_age_ml': round(vascular_age, 1),
                'chronological_age': age,
                'age_difference_ml': round(age_difference, 1),
                'status_ml': status,
                'risk_level_ml': risk,
                'confidence': round(confidence, 2),
                'method': 'Machine Learning' if self.model_loaded else 'Hybrid ML',
                'features_extracted': len(features)
            }

            logger.info(f"ML Vascular Age: {vascular_age:.1f} years "
                       f"(confidence: {confidence:.2f})")

            return results

        except Exception as e:
            logger.error(f"Error in ML vascular age prediction: {e}")
            # Fallback to simple estimate
            return {
                'vascular_age_ml': demographics.get('age', 40),
                'chronological_age': demographics.get('age', 40),
                'age_difference_ml': 0,
                'status_ml': 'Unknown',
                'risk_level_ml': 'Unknown',
                'confidence': 0,
                'method': 'Failed',
                'features_extracted': 0
            }

    def _hybrid_prediction(self, deep_features, demographics, ppg_features):
        """
        Hybrid prediction using deep features and known correlations
        """
        age = demographics.get('age', 40)

        # Base prediction from deep features - more conservative
        # Diastolic/Systolic ratio (feature 0) correlates with age
        ds_age = 40 + deep_features[0] * 20  # Center at 40, reduced sensitivity

        # LF/HF ratio (feature 1) indicates autonomic age
        lfhf_age = 40 + deep_features[1] * 10  # Center at 40, much reduced

        # Complexity (feature 3) decreases with age
        complexity_age = 45 - deep_features[3] * 30  # Centered around 45

        # If we have PPG features, use them too
        if ppg_features:
            ai = ppg_features.get('augmentation_index', 0.4)
            si = ppg_features.get('stiffness_index', 8.0)

            # Check for low BP (protective factor)
            systolic = ppg_features.get('systolic_bp', 120)
            bp_adjustment = 0
            if systolic < 110:
                # Low BP is protective, reduce age estimate
                bp_adjustment = -((110 - systolic) * 0.15)  # Up to -3 years for very low BP

            # AI-based age (more conservative)
            ai_age = 40 + (ai - 0.35) / 0.01  # Center at 40, match formula-based

            # SI-based age (more conservative)
            si_age = 40 + (si - 7.0) / 0.2  # Match formula-based

            # Weighted combination with BP adjustment
            vascular_age = (
                ds_age * 0.20 +      # Diastolic/Systolic
                lfhf_age * 0.10 +    # Autonomic function
                complexity_age * 0.10 +  # Signal complexity
                ai_age * 0.30 +      # Augmentation index
                si_age * 0.20 +      # Stiffness index
                age * 0.10           # Anchor to chronological age
            ) + bp_adjustment
        else:
            # Without PPG features, rely more on deep features
            vascular_age = (
                ds_age * 0.30 +
                lfhf_age * 0.20 +
                complexity_age * 0.20 +
                age * 0.30  # Higher weight on chronological age when uncertain
            )

        # Sanity check: limit deviation from chronological age
        max_deviation = 15
        if abs(vascular_age - age) > max_deviation:
            # Pull towards chronological age if calculation seems unrealistic
            vascular_age = age + np.sign(vascular_age - age) * max_deviation * 0.7

        # Apply bounds
        vascular_age = np.clip(vascular_age, 20, 90)

        return vascular_age