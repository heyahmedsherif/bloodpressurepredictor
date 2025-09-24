"""
Vascular Age Optimization and Confidence Scoring
=================================================
Provides additional optimization and confidence metrics for vascular age predictions.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class VascularAgeOptimizer:
    """Optimizes vascular age predictions with confidence scoring"""

    def __init__(self):
        """Initialize optimizer with quality thresholds"""
        self.quality_thresholds = {
            'excellent': {'min_frames': 150, 'max_noise': 0.1, 'min_peaks': 10},
            'good': {'min_frames': 100, 'max_noise': 0.2, 'min_peaks': 7},
            'fair': {'min_frames': 75, 'max_noise': 0.3, 'min_peaks': 5},
            'poor': {'min_frames': 50, 'max_noise': 0.5, 'min_peaks': 3}
        }

    def calculate_confidence(self, ppg_signal, features, demographics):
        """
        Calculate confidence level for vascular age prediction

        Args:
            ppg_signal: Raw PPG signal
            features: Extracted features
            demographics: Patient demographics

        Returns:
            Dict with confidence metrics
        """
        confidence_factors = []

        # 1. Signal quality (0-100)
        signal_quality = self._assess_signal_quality(ppg_signal)
        confidence_factors.append(('Signal Quality', signal_quality, 0.3))

        # 2. Feature consistency (0-100)
        feature_consistency = self._assess_feature_consistency(features)
        confidence_factors.append(('Feature Consistency', feature_consistency, 0.25))

        # 3. Demographic alignment (0-100)
        demographic_alignment = self._assess_demographic_alignment(features, demographics)
        confidence_factors.append(('Demographic Match', demographic_alignment, 0.2))

        # 4. Physiological plausibility (0-100)
        plausibility = self._assess_physiological_plausibility(features)
        confidence_factors.append(('Plausibility', plausibility, 0.25))

        # Calculate weighted confidence
        total_confidence = sum(score * weight for _, score, weight in confidence_factors)

        # Determine confidence level
        if total_confidence >= 80:
            level = 'High'
            color = 'success'
        elif total_confidence >= 60:
            level = 'Moderate'
            color = 'warning'
        else:
            level = 'Low'
            color = 'danger'

        return {
            'score': round(total_confidence, 1),
            'level': level,
            'color': color,
            'factors': {name: round(score, 1) for name, score, _ in confidence_factors},
            'recommendations': self._get_recommendations(confidence_factors)
        }

    def _assess_signal_quality(self, ppg_signal):
        """Assess PPG signal quality"""
        if len(ppg_signal) < 100:
            return 20

        # Check signal-to-noise ratio
        signal_std = np.std(ppg_signal)
        if signal_std == 0:
            return 10

        # Normalize signal
        normalized = (ppg_signal - np.mean(ppg_signal)) / signal_std

        # Check for clipping
        clipping_ratio = np.sum(np.abs(normalized) > 3) / len(normalized)
        if clipping_ratio > 0.1:
            return 40

        # Check for good periodicity (autocorrelation)
        autocorr = np.correlate(normalized[:100], normalized[:100], mode='same')
        if len(autocorr) > 1:
            periodicity = np.max(autocorr[1:]) / autocorr[0] if autocorr[0] != 0 else 0
        else:
            periodicity = 0

        quality = 50 + periodicity * 50
        return min(100, quality)

    def _assess_feature_consistency(self, features):
        """Check if extracted features are consistent"""
        consistency_score = 100

        # Check heart rate is reasonable
        hr = features.get('heart_rate', 75)
        if hr < 40 or hr > 120:
            consistency_score -= 30
        elif hr < 50 or hr > 100:
            consistency_score -= 15

        # Check augmentation index
        ai = features.get('augmentation_index', 0.4)
        if ai < 0.1 or ai > 0.8:
            consistency_score -= 20

        # Check stiffness index
        si = features.get('stiffness_index', 8)
        if si < 3 or si > 15:
            consistency_score -= 20

        return max(0, consistency_score)

    def _assess_demographic_alignment(self, features, demographics):
        """Check if features align with demographics"""
        age = demographics.get('age', 40)

        # Expected ranges based on age
        if age < 30:
            expected_ai = (0.15, 0.35)
            expected_si = (4, 7)
        elif age < 50:
            expected_ai = (0.25, 0.45)
            expected_si = (6, 10)
        elif age < 70:
            expected_ai = (0.35, 0.55)
            expected_si = (8, 12)
        else:
            expected_ai = (0.45, 0.65)
            expected_si = (10, 15)

        alignment_score = 100

        # Check if AI is in expected range
        ai = features.get('augmentation_index', 0.4)
        if not (expected_ai[0] <= ai <= expected_ai[1]):
            deviation = min(abs(ai - expected_ai[0]), abs(ai - expected_ai[1]))
            alignment_score -= min(40, deviation * 100)

        # Check if SI is in expected range
        si = features.get('stiffness_index', 8)
        if not (expected_si[0] <= si <= expected_si[1]):
            deviation = min(abs(si - expected_si[0]), abs(si - expected_si[1]))
            alignment_score -= min(40, deviation * 5)

        return max(0, alignment_score)

    def _assess_physiological_plausibility(self, features):
        """Check if features are physiologically plausible"""
        plausibility = 100

        # Check blood pressure
        systolic = features.get('systolic_bp', 120)
        diastolic = features.get('diastolic_bp', 80)

        # Pulse pressure should be 20-80
        pulse_pressure = systolic - diastolic
        if pulse_pressure < 20 or pulse_pressure > 80:
            plausibility -= 30

        # Check HRV
        hrv = features.get('hrv', 50)
        if hrv < 10 or hrv > 150:
            plausibility -= 20

        # Check relationships between features
        # Higher AI should correlate with higher SI
        ai = features.get('augmentation_index', 0.4)
        si = features.get('stiffness_index', 8)

        # Normalize to 0-1 scale
        ai_norm = ai
        si_norm = (si - 3) / 12  # SI range 3-15

        # They should be somewhat correlated
        if abs(ai_norm - si_norm) > 0.5:
            plausibility -= 15

        return max(0, plausibility)

    def _get_recommendations(self, confidence_factors):
        """Get recommendations to improve confidence"""
        recommendations = []

        for name, score, _ in confidence_factors:
            if score < 60:
                if name == 'Signal Quality':
                    recommendations.append("Ensure good lighting and keep finger still")
                elif name == 'Feature Consistency':
                    recommendations.append("Try recording again with stable heart rate")
                elif name == 'Demographic Match':
                    recommendations.append("Features may indicate need for calibration")
                elif name == 'Plausibility':
                    recommendations.append("Results may need clinical validation")

        if not recommendations:
            recommendations.append("Good quality measurement")

        return recommendations

    def optimize_prediction(self, vascular_age, features, demographics, ppg_signal):
        """
        Apply final optimizations to vascular age prediction

        Args:
            vascular_age: Initial prediction
            features: Extracted features
            demographics: Patient info
            ppg_signal: Raw signal

        Returns:
            Optimized vascular age with confidence
        """
        # Get confidence metrics
        confidence = self.calculate_confidence(ppg_signal, features, demographics)

        # Apply confidence-based adjustment
        if confidence['score'] < 50:
            # Low confidence - pull toward chronological age
            chronological_age = demographics.get('age', 40)
            adjustment_factor = (50 - confidence['score']) / 100
            vascular_age = vascular_age * (1 - adjustment_factor) + chronological_age * adjustment_factor

        # Add uncertainty bounds based on confidence
        if confidence['score'] >= 80:
            uncertainty = 3  # ±3 years
        elif confidence['score'] >= 60:
            uncertainty = 5  # ±5 years
        else:
            uncertainty = 8  # ±8 years

        return {
            'optimized_age': round(vascular_age, 1),
            'confidence': confidence,
            'uncertainty': uncertainty,
            'prediction_range': {
                'lower': round(vascular_age - uncertainty, 1),
                'upper': round(vascular_age + uncertainty, 1)
            }
        }