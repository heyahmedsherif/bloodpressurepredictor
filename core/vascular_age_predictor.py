"""
Vascular Age Predictor from PPG Signals
========================================
Estimates vascular age based on arterial stiffness markers extracted from PPG.

Based on research showing correlation between:
- Augmentation Index (AI) and vascular aging
- Arterial Stiffness Index (SI) and age
- Pulse Wave Velocity and cardiovascular age
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

class VascularAgePredictor:
    """Predict vascular age from PPG-derived arterial stiffness features"""

    def __init__(self):
        """Initialize vascular age predictor with reference values"""
        # Reference values from literature - ADJUSTED FOR REALISM
        self.ai_baseline = 0.30  # Baseline AI for age 40
        self.ai_change_per_year = 0.01  # AI increases ~1% per year (was too sensitive)

        self.si_baseline = 7.0  # Baseline SI for age 40 (adjusted)
        self.si_change_per_year = 0.2  # SI increases with age (less sensitive)

        self.sbp_baseline = 120  # Baseline systolic BP
        self.sbp_change_per_year = 1.0  # SBP increases ~1 mmHg/year (was too sensitive)

    def predict_vascular_age(self, features, demographics):
        """
        Predict vascular age from PPG features

        Args:
            features: Dict containing:
                - augmentation_index (AI)
                - stiffness_index (SI)
                - heart_rate
                - hrv (heart rate variability)
                - systolic_bp
                - diastolic_bp
            demographics: Dict containing:
                - age (chronological age)
                - gender
                - bmi

        Returns:
            Dict with vascular age metrics
        """
        try:
            # Extract features
            ai = features.get('augmentation_index', 0.4)
            si = features.get('stiffness_index', 8.0)
            hrv = features.get('hrv', 50)
            systolic = features.get('systolic_bp', 120)
            heart_rate = features.get('heart_rate', 75)

            # Demographics
            chronological_age = demographics.get('age', 40)
            gender = demographics.get('gender', 'M')
            bmi = demographics.get('bmi', 25)

            # Calculate vascular age components

            # 1. AI-based age (weighted 40%)
            # Center around age 40, not 30
            ai_age = 40 + (ai - self.ai_baseline) / self.ai_change_per_year
            ai_age = np.clip(ai_age, 20, 80)  # Reasonable bounds

            # 2. SI-based age (weighted 30%)
            si_age = 40 + (si - self.si_baseline) / self.si_change_per_year
            si_age = np.clip(si_age, 20, 80)  # Reasonable bounds

            # 3. BP-based age (weighted 20%)
            # Note: Low BP is protective, shouldn't increase vascular age
            if systolic < 110:
                bp_age = 40 - (110 - systolic) * 0.2  # Lower BP = younger vascular age
            else:
                bp_age = 40 + (systolic - self.sbp_baseline) / self.sbp_change_per_year
            bp_age = np.clip(bp_age, 20, 80)  # Reasonable bounds

            # 4. HRV-based age (weighted 10%)
            # HRV decreases with age: young (~60-100ms), old (~20-40ms)
            # Better HRV = younger vascular age
            if hrv > 60:
                hrv_age = 40 - (hrv - 60) * 0.1  # High HRV = younger (less sensitive)
            elif hrv > 40:
                hrv_age = 40 + (60 - hrv) * 0.3
            else:
                hrv_age = 45 + (40 - hrv) * 0.5
            hrv_age = np.clip(hrv_age, 20, 80)  # Reasonable bounds

            # Adjust for gender (women typically have lower vascular age)
            gender_adjustment = -2 if gender == 'F' else 0

            # Adjust for BMI (higher BMI increases vascular age)
            bmi_adjustment = max(0, (bmi - 25) * 0.3)  # Only penalize if BMI > 25

            # Weighted combination
            vascular_age = (
                ai_age * 0.40 +
                si_age * 0.30 +
                bp_age * 0.20 +
                hrv_age * 0.10
            ) + gender_adjustment + bmi_adjustment

            # Sanity check: vascular age should be within reasonable range of chronological age
            # Most healthy people are within +/- 10 years of their chronological age
            # With your good health metrics, bias towards chronological age
            max_deviation = 15
            if abs(vascular_age - chronological_age) > max_deviation:
                # Pull towards chronological age if calculation seems unrealistic
                vascular_age = chronological_age + np.sign(vascular_age - chronological_age) * max_deviation * 0.8

            # Ensure reasonable bounds
            vascular_age = np.clip(vascular_age, 20, 90)

            # Calculate difference from chronological age
            age_difference = vascular_age - chronological_age

            # Determine vascular health status
            if age_difference <= -5:
                status = "Excellent"
                risk_level = "Low"
            elif age_difference <= 0:
                status = "Good"
                risk_level = "Low"
            elif age_difference <= 5:
                status = "Normal"
                risk_level = "Moderate"
            elif age_difference <= 10:
                status = "Accelerated Aging"
                risk_level = "Elevated"
            else:
                status = "Significant Aging"
                risk_level = "High"

            # Calculate vascular health score (0-100)
            # Perfect health = vascular age 10 years younger than chronological
            # Poor health = vascular age 20 years older than chronological
            health_score = max(0, min(100, 80 - age_difference * 2.5))

            results = {
                'vascular_age': round(vascular_age, 1),
                'chronological_age': chronological_age,
                'age_difference': round(age_difference, 1),
                'status': status,
                'risk_level': risk_level,
                'health_score': round(health_score, 1),
                'components': {
                    'ai_age': round(ai_age, 1),
                    'si_age': round(si_age, 1),
                    'bp_age': round(bp_age, 1),
                    'hrv_age': round(hrv_age, 1)
                },
                'features_used': {
                    'augmentation_index': round(ai, 3),
                    'stiffness_index': round(si, 2),
                    'systolic_bp': round(systolic, 1),
                    'hrv': round(hrv, 1)
                }
            }

            logger.info(f"Vascular age: {vascular_age:.1f} years "
                       f"(chronological: {chronological_age}, diff: {age_difference:+.1f})")

            return results

        except Exception as e:
            logger.error(f"Error predicting vascular age: {e}")
            return {
                'vascular_age': demographics.get('age', 40),
                'chronological_age': demographics.get('age', 40),
                'age_difference': 0,
                'status': 'Unknown',
                'risk_level': 'Unknown',
                'health_score': 50
            }

    def get_recommendations(self, vascular_age_results):
        """
        Get personalized recommendations based on vascular age

        Args:
            vascular_age_results: Dict from predict_vascular_age()

        Returns:
            List of recommendation strings
        """
        recommendations = []
        age_diff = vascular_age_results['age_difference']

        if age_diff > 10:
            recommendations.extend([
                "⚠️ Consult with a healthcare provider about cardiovascular risk",
                "🏃‍♂️ Increase aerobic exercise to 150+ minutes per week",
                "🥗 Adopt a heart-healthy diet (Mediterranean or DASH)",
                "💊 Consider medication if blood pressure is elevated"
            ])
        elif age_diff > 5:
            recommendations.extend([
                "🏃‍♂️ Regular moderate exercise (30 min/day, 5 days/week)",
                "🧘‍♂️ Practice stress reduction techniques",
                "🚭 Avoid smoking and limit alcohol",
                "📊 Monitor blood pressure regularly"
            ])
        elif age_diff > 0:
            recommendations.extend([
                "✅ Maintain current healthy habits",
                "🏃‍♂️ Continue regular physical activity",
                "🥗 Keep a balanced diet with fruits and vegetables",
                "😴 Ensure 7-8 hours of quality sleep"
            ])
        else:
            recommendations.extend([
                "🎉 Excellent vascular health!",
                "✅ Continue your current lifestyle",
                "📊 Annual health check-ups recommended",
                "🏃‍♂️ Maintain regular exercise routine"
            ])

        return recommendations