"""
Measurement Stabilizer for Consistent Health Predictions
=========================================================
Implements averaging, outlier detection, and consistency checking
to reduce measurement variability.
"""

import numpy as np
import logging
from collections import deque
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MeasurementStabilizer:
    """Stabilizes health measurements across multiple readings"""

    def __init__(self, history_size=3):
        """
        Initialize the stabilizer

        Args:
            history_size: Number of recent measurements to keep for averaging
        """
        self.history_size = history_size
        self.measurement_history = {
            'heart_rate': deque(maxlen=history_size),
            'systolic': deque(maxlen=history_size),
            'diastolic': deque(maxlen=history_size),
            'glucose': deque(maxlen=history_size),
            'cholesterol': deque(maxlen=history_size),
            'vascular_age': deque(maxlen=history_size),
            'signal_quality': deque(maxlen=history_size)
        }

        # Acceptable ranges for outlier detection
        self.valid_ranges = {
            'heart_rate': (40, 150),
            'systolic': (80, 180),
            'diastolic': (50, 110),
            'glucose': (60, 300),
            'cholesterol': (100, 400),
            'vascular_age': (20, 80),
            'signal_quality': (0, 1)
        }

        # Maximum allowed deviation from median
        self.max_deviations = {
            'heart_rate': 30,  # BPM
            'systolic': 30,     # mmHg
            'diastolic': 20,    # mmHg
            'glucose': 50,      # mg/dL
            'cholesterol': 50,  # mg/dL
            'vascular_age': 15, # years
            'signal_quality': 0.5
        }

    def add_measurement(self, metrics: Dict) -> Dict:
        """
        Add a new measurement and return stabilized values

        Args:
            metrics: Dictionary containing measurement values

        Returns:
            Stabilized metrics dictionary
        """
        # Extract and validate measurements
        current_measurements = self._extract_measurements(metrics)

        # Check for outliers before adding to history
        filtered_measurements = self._filter_outliers(current_measurements)

        # Add to history
        for key, value in filtered_measurements.items():
            if value is not None:
                self.measurement_history[key].append(value)

        # Calculate stabilized values
        stabilized = self._calculate_stabilized_values()

        # Apply consistency checks
        stabilized = self._apply_consistency_checks(stabilized)

        # Log stabilization results
        self._log_stabilization(current_measurements, stabilized)

        return stabilized

    def _extract_measurements(self, metrics: Dict) -> Dict:
        """Extract relevant measurements from metrics"""
        measurements = {}

        # Heart rate
        if 'heart_rate_enhanced' in metrics:
            measurements['heart_rate'] = metrics['heart_rate_enhanced'].get('heart_rate')
        elif 'heart_rate' in metrics:
            measurements['heart_rate'] = metrics['heart_rate']

        # Blood pressure
        measurements['systolic'] = metrics.get('systolic')
        measurements['diastolic'] = metrics.get('diastolic')

        # Other metrics
        measurements['glucose'] = metrics.get('glucose')
        measurements['cholesterol'] = metrics.get('cholesterol_total_original')
        measurements['vascular_age'] = metrics.get('vascular_age')

        # Signal quality
        if 'signal_quality' in metrics:
            measurements['signal_quality'] = metrics['signal_quality'].get('score')

        return measurements

    def _filter_outliers(self, measurements: Dict) -> Dict:
        """Filter out outlier measurements"""
        filtered = {}

        for key, value in measurements.items():
            if value is None:
                filtered[key] = None
                continue

            # Check if within valid range
            min_val, max_val = self.valid_ranges.get(key, (None, None))
            if min_val is not None and max_val is not None:
                if value < min_val or value > max_val:
                    logger.warning(f"Outlier detected for {key}: {value} (range: {min_val}-{max_val})")
                    filtered[key] = None
                    continue

            # Check deviation from history median if we have history
            if len(self.measurement_history[key]) > 0:
                history_median = np.median(list(self.measurement_history[key]))
                deviation = abs(value - history_median)
                max_deviation = self.max_deviations.get(key, float('inf'))

                if deviation > max_deviation:
                    logger.warning(f"Large deviation for {key}: {value} (median: {history_median:.1f}, dev: {deviation:.1f})")
                    # Use weighted value instead of rejecting completely
                    weight = max(0, 1 - (deviation - max_deviation) / max_deviation)
                    filtered[key] = history_median * (1 - weight) + value * weight
                else:
                    filtered[key] = value
            else:
                filtered[key] = value

        return filtered

    def _calculate_stabilized_values(self) -> Dict:
        """Calculate stabilized values from history"""
        stabilized = {}

        for key, history in self.measurement_history.items():
            if len(history) == 0:
                stabilized[key] = None
            elif len(history) == 1:
                stabilized[key] = history[0]
            else:
                # Use weighted average with more recent measurements having higher weight
                weights = np.array([0.5 ** i for i in range(len(history) - 1, -1, -1)])
                weights = weights / weights.sum()

                values = np.array(list(history))

                # Remove extreme outliers before averaging
                if len(values) >= 3:
                    q1 = np.percentile(values, 25)
                    q3 = np.percentile(values, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    mask = (values >= lower_bound) & (values <= upper_bound)
                    if mask.sum() > 0:
                        values = values[mask]
                        weights = weights[mask]
                        weights = weights / weights.sum()

                stabilized[key] = np.average(values, weights=weights)

        return stabilized

    def _apply_consistency_checks(self, stabilized: Dict) -> Dict:
        """Apply medical consistency checks"""

        # Ensure pulse pressure is reasonable
        if stabilized.get('systolic') and stabilized.get('diastolic'):
            pulse_pressure = stabilized['systolic'] - stabilized['diastolic']

            if pulse_pressure < 25:
                # Adjust to maintain minimum pulse pressure
                adjustment = (25 - pulse_pressure) / 2
                stabilized['systolic'] += adjustment
                stabilized['diastolic'] -= adjustment
                logger.info(f"Adjusted BP for pulse pressure: {stabilized['systolic']:.0f}/{stabilized['diastolic']:.0f}")

            elif pulse_pressure > 70:
                # Adjust to maintain maximum pulse pressure
                adjustment = (pulse_pressure - 70) / 2
                stabilized['systolic'] -= adjustment
                stabilized['diastolic'] += adjustment
                logger.info(f"Adjusted BP for pulse pressure: {stabilized['systolic']:.0f}/{stabilized['diastolic']:.0f}")

        # Ensure vascular age is reasonable relative to chronological age
        # This is handled in the vascular age predictor itself

        return stabilized

    def _log_stabilization(self, current: Dict, stabilized: Dict):
        """Log stabilization effects"""
        changes = []

        for key in ['heart_rate', 'systolic', 'diastolic', 'vascular_age']:
            if current.get(key) and stabilized.get(key):
                change = abs(current[key] - stabilized[key])
                if change > 0.1:
                    changes.append(f"{key}: {current[key]:.1f} → {stabilized[key]:.1f}")

        if changes:
            logger.info(f"Stabilization applied: {', '.join(changes)}")

    def reset(self):
        """Reset measurement history"""
        for history in self.measurement_history.values():
            history.clear()
        logger.info("Measurement history reset")

    def get_confidence(self) -> float:
        """
        Calculate overall confidence based on measurement consistency

        Returns:
            Confidence score between 0 and 1
        """
        if not any(len(h) > 0 for h in self.measurement_history.values()):
            return 0.5

        confidences = []

        for key, history in self.measurement_history.items():
            if len(history) >= 2 and key != 'signal_quality':
                values = list(history)
                cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 1
                # Lower CV = higher confidence
                confidence = max(0, 1 - cv)
                confidences.append(confidence)

        # Include signal quality in confidence
        if len(self.measurement_history['signal_quality']) > 0:
            avg_quality = np.mean(list(self.measurement_history['signal_quality']))
            confidences.append(avg_quality)

        if confidences:
            return np.mean(confidences)
        else:
            return 0.5

    def get_measurement_count(self) -> int:
        """Get the number of measurements in history"""
        return max(len(h) for h in self.measurement_history.values())

    def should_retry(self) -> bool:
        """
        Determine if another measurement should be taken

        Returns:
            True if retry is recommended
        """
        # Check if we have enough measurements
        if self.get_measurement_count() < 2:
            return True

        # Check if signal quality is consistently poor
        if len(self.measurement_history['signal_quality']) > 0:
            avg_quality = np.mean(list(self.measurement_history['signal_quality']))
            if avg_quality < 0.4:
                logger.info("Retry recommended due to poor signal quality")
                return True

        # Check if measurements are too variable
        confidence = self.get_confidence()
        if confidence < 0.5:
            logger.info("Retry recommended due to high variability")
            return True

        return False