"""
Model Validation and Sanity Checks
====================================
Ensures all health predictions are within reasonable bounds.
Implements best practices for model deployment.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class ModelValidator:
    """Validates and sanitizes all model predictions"""

    def __init__(self):
        """Initialize validation bounds for all metrics"""
        # Vascular age bounds relative to chronological age
        self.max_vascular_age_deviation = 20  # Maximum +/- years from chronological

        # Absolute bounds for all metrics
        self.bounds = {
            'vascular_age': (18, 95),
            'systolic_bp': (70, 200),
            'diastolic_bp': (40, 130),
            'glucose': (60, 400),
            'cholesterol_total': (100, 400),
            'ldl': (40, 300),
            'hdl': (20, 100),
            'heart_rate': (40, 200),
            'hrv': (5, 200),
            'augmentation_index': (0.0, 1.0),
            'stiffness_index': (3.0, 20.0)
        }

    def validate_vascular_age(self, vascular_age, chronological_age, features=None):
        """
        Validate and correct vascular age predictions

        Args:
            vascular_age: Predicted vascular age
            chronological_age: Patient's actual age
            features: Optional dict of features used for prediction

        Returns:
            Validated vascular age
        """
        original_age = vascular_age

        # Step 1: Apply absolute bounds
        min_age, max_age = self.bounds['vascular_age']
        vascular_age = np.clip(vascular_age, min_age, max_age)

        # Step 2: Check deviation from chronological age
        deviation = vascular_age - chronological_age

        if abs(deviation) > self.max_vascular_age_deviation:
            # Log extreme prediction for debugging
            logger.warning(f"Extreme vascular age detected: {original_age:.1f} "
                          f"(chronological: {chronological_age})")

            # Apply intelligent correction based on features
            if features:
                correction_factor = self._calculate_correction_factor(features, chronological_age)
            else:
                correction_factor = 0.5  # Default conservative correction

            # Pull towards chronological age
            vascular_age = chronological_age + np.sign(deviation) * self.max_vascular_age_deviation * correction_factor

        # Step 3: Final sanity check
        vascular_age = np.clip(vascular_age,
                               chronological_age - self.max_vascular_age_deviation,
                               chronological_age + self.max_vascular_age_deviation)

        if original_age != vascular_age:
            logger.info(f"Vascular age corrected: {original_age:.1f} -> {vascular_age:.1f}")

        return round(vascular_age, 1)

    def _calculate_correction_factor(self, features, age):
        """
        Calculate correction factor based on health indicators

        Better health metrics -> allow more deviation in favorable direction
        Poor health metrics -> allow more deviation in unfavorable direction
        """
        correction = 0.5  # Base correction

        # Check blood pressure
        systolic = features.get('systolic_bp', 120)
        if systolic < 110:
            # Low BP is protective, allow younger vascular age
            correction = 0.7
        elif systolic > 140:
            # High BP is risky, allow older vascular age
            correction = 0.7

        # Check HRV (higher is better)
        hrv = features.get('hrv', 50)
        if hrv > 60:
            correction = max(correction, 0.6)
        elif hrv < 30:
            correction = max(correction, 0.6)

        # Check augmentation index (lower is better)
        ai = features.get('augmentation_index', 0.4)
        if ai < 0.3:
            correction = max(correction, 0.65)
        elif ai > 0.5:
            correction = max(correction, 0.65)

        return correction

    def validate_blood_pressure(self, systolic, diastolic):
        """Validate blood pressure values"""
        # Ensure systolic > diastolic
        if systolic <= diastolic:
            systolic = diastolic + 20

        # Apply bounds
        systolic = np.clip(systolic, *self.bounds['systolic_bp'])
        diastolic = np.clip(diastolic, *self.bounds['diastolic_bp'])

        # Ensure pulse pressure is reasonable (20-80 mmHg)
        pulse_pressure = systolic - diastolic
        if pulse_pressure < 20:
            systolic = diastolic + 20
        elif pulse_pressure > 80:
            systolic = diastolic + 80

        return systolic, diastolic

    def validate_cholesterol(self, total=None, ldl=None, hdl=None):
        """Validate cholesterol values and ensure consistency"""
        results = {}

        if total is not None:
            results['total'] = np.clip(total, *self.bounds['cholesterol_total'])

        if ldl is not None:
            results['ldl'] = np.clip(ldl, *self.bounds['ldl'])

        if hdl is not None:
            results['hdl'] = np.clip(hdl, *self.bounds['hdl'])

        # Ensure components sum reasonably to total
        if 'ldl' in results and 'hdl' in results:
            vldl_estimate = 30  # Typical VLDL
            calculated_total = results['ldl'] + results['hdl'] + vldl_estimate

            if 'total' in results:
                # Check consistency
                if abs(results['total'] - calculated_total) > 50:
                    logger.warning(f"Cholesterol components don't sum correctly: "
                                 f"Total={results['total']}, Sum={calculated_total}")
                    # Average them for consistency
                    results['total'] = (results['total'] + calculated_total) / 2
            else:
                results['total'] = calculated_total

        return results

    def validate_glucose(self, glucose):
        """Validate glucose level"""
        return np.clip(glucose, *self.bounds['glucose'])

    def validate_ppg_features(self, features):
        """Validate PPG-derived features"""
        validated = features.copy()

        # Validate augmentation index
        if 'augmentation_index' in validated:
            validated['augmentation_index'] = np.clip(
                validated['augmentation_index'],
                *self.bounds['augmentation_index']
            )

        # Validate stiffness index
        if 'stiffness_index' in validated:
            validated['stiffness_index'] = np.clip(
                validated['stiffness_index'],
                *self.bounds['stiffness_index']
            )

        # Validate heart rate
        if 'heart_rate' in validated:
            validated['heart_rate'] = np.clip(
                validated['heart_rate'],
                *self.bounds['heart_rate']
            )

        # Validate HRV
        if 'hrv' in validated:
            validated['hrv'] = np.clip(
                validated['hrv'],
                *self.bounds['hrv']
            )

        return validated

    def validate_all_predictions(self, predictions, demographics):
        """
        Validate all predictions in a results dictionary

        Args:
            predictions: Dict of all predictions
            demographics: Patient demographics

        Returns:
            Validated predictions dict
        """
        validated = predictions.copy()

        # Validate vascular age if present
        if 'vascular_age_data' in validated:
            vascular_data = validated['vascular_age_data']
            chronological_age = demographics.get('age', 40)

            # Extract features for intelligent correction
            features = {
                'systolic_bp': validated.get('systolic', 120),
                'diastolic_bp': validated.get('diastolic', 80),
                'hrv': vascular_data.get('features_used', {}).get('hrv', 50),
                'augmentation_index': vascular_data.get('features_used', {}).get('augmentation_index', 0.4)
            }

            # Validate vascular age
            vascular_data['vascular_age'] = self.validate_vascular_age(
                vascular_data['vascular_age'],
                chronological_age,
                features
            )

            # Recalculate age difference
            vascular_data['age_difference'] = vascular_data['vascular_age'] - chronological_age

            # Update status based on new age difference
            age_diff = vascular_data['age_difference']
            if age_diff <= -5:
                vascular_data['status'] = "Excellent"
                vascular_data['risk_level'] = "Low"
            elif age_diff <= 0:
                vascular_data['status'] = "Good"
                vascular_data['risk_level'] = "Low"
            elif age_diff <= 5:
                vascular_data['status'] = "Normal"
                vascular_data['risk_level'] = "Moderate"
            elif age_diff <= 10:
                vascular_data['status'] = "Accelerated Aging"
                vascular_data['risk_level'] = "Elevated"
            else:
                vascular_data['status'] = "Significant Aging"
                vascular_data['risk_level'] = "High"

            validated['vascular_age'] = vascular_data['vascular_age']

        # Validate ML vascular age if present
        if 'vascular_age_ml_data' in validated:
            ml_data = validated['vascular_age_ml_data']
            chronological_age = demographics.get('age', 40)

            ml_data['vascular_age_ml'] = self.validate_vascular_age(
                ml_data.get('vascular_age_ml', chronological_age),
                chronological_age
            )
            ml_data['age_difference_ml'] = ml_data['vascular_age_ml'] - chronological_age

        # Validate blood pressure
        if 'systolic' in validated and 'diastolic' in validated:
            validated['systolic'], validated['diastolic'] = self.validate_blood_pressure(
                validated['systolic'],
                validated['diastolic']
            )

        # Validate glucose
        if 'glucose' in validated:
            validated['glucose'] = self.validate_glucose(validated['glucose'])

        # Validate cholesterol
        chol_results = self.validate_cholesterol(
            total=validated.get('cholesterol_total_original'),
            ldl=validated.get('ldl'),
            hdl=validated.get('hdl')
        )
        validated.update(chol_results)

        return validated