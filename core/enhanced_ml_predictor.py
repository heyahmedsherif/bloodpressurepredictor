"""
Enhanced ML Health Predictor with Dual Cholesterol Models
===========================================================
Provides both total cholesterol and LDL/HDL breakdown predictions.
"""

import pickle
import numpy as np
import logging
from pathlib import Path
from .ppg_feature_extractor import PPGFeatureExtractor
from .ppg_cholesterol_extractor import CholesterolFeatureExtractor

logger = logging.getLogger(__name__)

class EnhancedMLHealthPredictor:
    """Predict health metrics with both original and detailed cholesterol models"""

    def __init__(self):
        """Initialize and load all ML models including LDL/HDL"""
        self.models_loaded = False
        self.ldl_hdl_models_loaded = False
        self.feature_extractor = PPGFeatureExtractor()
        self.cholesterol_extractor = CholesterolFeatureExtractor()

        # Load original models
        try:
            # Glucose models
            with open('models/glucose/glucose_model.pkl', 'rb') as f:
                self.glucose_model = pickle.load(f)
            with open('models/glucose/glucose_scaler.pkl', 'rb') as f:
                self.glucose_scaler = pickle.load(f)

            # Original cholesterol model (total)
            with open('models/cholesterol/cholesterol_model.pkl', 'rb') as f:
                self.cholesterol_model = pickle.load(f)
            with open('models/cholesterol/cholesterol_scaler.pkl', 'rb') as f:
                self.cholesterol_scaler = pickle.load(f)

            # Blood pressure models
            with open('models/blood_pressure/sbp_model.pkl', 'rb') as f:
                self.systolic_model = pickle.load(f)
            with open('models/blood_pressure/dbp_model.pkl', 'rb') as f:
                self.diastolic_model = pickle.load(f)
            with open('models/blood_pressure/bp_scaler.pkl', 'rb') as f:
                self.bp_scaler = pickle.load(f)
            with open('models/blood_pressure/bp_poly.pkl', 'rb') as f:
                self.bp_poly = pickle.load(f)

            self.models_loaded = True
            logger.info("✅ Original ML models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading original models: {e}")
            self.models_loaded = False

        # Load LDL/HDL models
        try:
            ldl_path = Path('models/cholesterol_detailed')
            if ldl_path.exists():
                with open(ldl_path / 'ldl_model.pkl', 'rb') as f:
                    self.ldl_model = pickle.load(f)
                with open(ldl_path / 'hdl_model.pkl', 'rb') as f:
                    self.hdl_model = pickle.load(f)
                with open(ldl_path / 'cholesterol_scaler.pkl', 'rb') as f:
                    self.ldl_hdl_scaler = pickle.load(f)

                self.ldl_hdl_models_loaded = True
                logger.info("✅ LDL/HDL models loaded successfully")
            else:
                logger.info("LDL/HDL models not found, will use estimation")

        except Exception as e:
            logger.error(f"Error loading LDL/HDL models: {e}")
            self.ldl_hdl_models_loaded = False

    def predict_all_metrics(self, ppg_signal, demographics):
        """
        Predict all health metrics including both cholesterol methods

        Args:
            ppg_signal: Array of PPG values
            demographics: Dict with age, gender, height, weight

        Returns:
            Dict with all predictions including comparison
        """
        results = {}

        # Calculate BMI
        height_m = demographics.get('height', 170) / 100
        weight_kg = demographics.get('weight', 70)
        bmi = weight_kg / (height_m ** 2)
        demographics['bmi'] = bmi

        # Extract features for both models
        basic_features = self.feature_extractor.extract_features(ppg_signal)
        advanced_features = self.cholesterol_extractor.extract_all_features(
            ppg_signal, demographics
        )

        # Blood Pressure Prediction
        bp_results = self._predict_blood_pressure(advanced_features, demographics)
        results['systolic'] = bp_results['systolic']
        results['diastolic'] = bp_results['diastolic']

        # Update features with BP for other predictions
        advanced_features['sbp'] = results['systolic']
        advanced_features['dbp'] = results['diastolic']

        # Glucose Prediction
        results['glucose'] = self._predict_glucose(advanced_features, bp_results)

        # Original Cholesterol Prediction (Total)
        results['cholesterol_total_original'] = self._predict_cholesterol_original(
            basic_features, demographics, bp_results
        )

        # New Cholesterol Prediction (LDL/HDL)
        cholesterol_detailed = self._predict_cholesterol_detailed(
            advanced_features, demographics
        )
        results.update(cholesterol_detailed)

        # Comparison and validation
        results['comparison'] = self._compare_cholesterol_methods(results)

        return results

    def _predict_blood_pressure(self, features, demographics):
        """Predict blood pressure using ML models"""
        if not self.models_loaded:
            # Fallback calculation
            return {
                'systolic': 120 + demographics.get('age', 40) * 0.3,
                'diastolic': 80 + demographics.get('age', 40) * 0.1
            }

        try:
            # Prepare features for BP model
            bp_features = [
                features.get('ppg_amplitude', 0.5),
                features.get('heart_rate', 75),
                features.get('ppg_width', 0.3),
                demographics.get('age', 40),
                demographics.get('bmi', 25),
                features.get('hrv', 50)
            ]

            # Apply polynomial features and scaling
            X = np.array(bp_features).reshape(1, -1)
            X_poly = self.bp_poly.transform(X)
            X_scaled = self.bp_scaler.transform(X_poly)

            # Predict
            systolic = self.systolic_model.predict(X_scaled)[0]
            diastolic = self.diastolic_model.predict(X_scaled)[0]

            # Ensure reasonable ranges
            systolic = np.clip(systolic, 90, 200)
            diastolic = np.clip(diastolic, 60, 120)

            return {'systolic': systolic, 'diastolic': diastolic}

        except Exception as e:
            logger.error(f"BP prediction error: {e}")
            return {'systolic': 120, 'diastolic': 80}

    def _predict_glucose(self, features, bp_results):
        """Predict glucose levels"""
        if not self.models_loaded:
            return 95.0

        try:
            # Prepare features
            glucose_features = [
                features.get('ppg_amplitude', 0.5),
                features.get('heart_rate', 75),
                bp_results['systolic'],
                bp_results['diastolic'],
                features.get('age', 40),
                features.get('bmi', 25)
            ]

            X = np.array(glucose_features).reshape(1, -1)
            X_scaled = self.glucose_scaler.transform(X)

            glucose = self.glucose_model.predict(X_scaled)[0]
            return np.clip(glucose, 70, 300)

        except Exception as e:
            logger.error(f"Glucose prediction error: {e}")
            return 95.0

    def _predict_cholesterol_original(self, features, demographics, bp_results):
        """Predict total cholesterol using original model"""
        if not self.models_loaded:
            return 180.0

        try:
            # Prepare features for original cholesterol model
            chol_features = [
                demographics.get('age', 40),
                features.get('heart_rate', 75),
                bp_results['systolic'],
                bp_results['diastolic'],
                demographics.get('bmi', 25),
                features.get('ppg_variability', 0.05)
            ]

            X = np.array(chol_features).reshape(1, -1)
            X_scaled = self.cholesterol_scaler.transform(X)

            cholesterol = self.cholesterol_model.predict(X_scaled)[0]
            return np.clip(cholesterol, 120, 400)

        except Exception as e:
            logger.error(f"Original cholesterol prediction error: {e}")
            return 180.0

    def _predict_cholesterol_detailed(self, features, demographics):
        """Predict LDL and HDL separately"""
        results = {}

        if self.ldl_hdl_models_loaded:
            try:
                # Prepare features for LDL/HDL models
                ldl_hdl_features = [
                    features.get('heart_rate', 75),
                    features.get('hrv', 50),
                    features.get('mean_amplitude', 0.5),
                    features.get('std_amplitude', 0.1),
                    features.get('mean_width', 0.3),
                    features.get('std_width', 0.05),
                    features.get('mean_ai', 0.4),
                    features.get('std_ai', 0.1),
                    features.get('stiffness_index', 10.0),
                    features.get('reflection_index', 4.0),
                    demographics.get('age', 40),
                    features.get('sex', 0)
                ]

                X = np.array(ldl_hdl_features).reshape(1, -1)
                X_scaled = self.ldl_hdl_scaler.transform(X)

                # Predict LDL and HDL
                ldl = self.ldl_model.predict(X_scaled)[0]
                hdl = self.hdl_model.predict(X_scaled)[0]

                # Ensure reasonable ranges
                ldl = np.clip(ldl, 50, 300)
                hdl = np.clip(hdl, 30, 100)

                # Calculate total and other metrics
                vldl = 30  # Typical VLDL estimate (triglycerides/5)
                total = ldl + hdl + vldl

                results['ldl'] = ldl
                results['hdl'] = hdl
                results['vldl_estimate'] = vldl
                results['cholesterol_total_new'] = total
                results['ldl_hdl_ratio'] = ldl / hdl if hdl > 0 else 3.5

                # Risk assessment
                risk_calculator = CholesterolFeatureExtractor()
                results['cardiovascular_risk'] = risk_calculator.calculate_risk_category(ldl, hdl)

                logger.info(f"LDL/HDL prediction: LDL={ldl:.1f}, HDL={hdl:.1f}, Total={total:.1f}")

            except Exception as e:
                logger.error(f"LDL/HDL prediction error: {e}")
                results = self._estimate_ldl_hdl_fallback(features, demographics)

        else:
            # Use estimation based on total cholesterol
            results = self._estimate_ldl_hdl_fallback(features, demographics)

        return results

    def _estimate_ldl_hdl_fallback(self, features, demographics):
        """Fallback LDL/HDL estimation when models not available"""
        # Use typical ratios
        total = 180  # Default

        # Gender-based estimates
        if demographics.get('gender', 'M') == 'F':
            hdl = 55
            ldl_ratio = 0.60
        else:
            hdl = 45
            ldl_ratio = 0.65

        # Adjust based on features
        hrv = features.get('hrv', 50)
        hdl = hdl * (1 + (hrv - 50) / 100)  # Better HRV -> higher HDL

        ldl = total * ldl_ratio
        vldl = 30

        return {
            'ldl': ldl,
            'hdl': hdl,
            'vldl_estimate': vldl,
            'cholesterol_total_new': ldl + hdl + vldl,
            'ldl_hdl_ratio': ldl / hdl if hdl > 0 else 3.5,
            'cardiovascular_risk': "Moderate"
        }

    def _compare_cholesterol_methods(self, results):
        """Compare original vs new cholesterol predictions"""
        comparison = {}

        total_original = results.get('cholesterol_total_original', 0)
        total_new = results.get('cholesterol_total_new', 0)
        ldl = results.get('ldl', 0)
        hdl = results.get('hdl', 0)
        vldl = results.get('vldl_estimate', 0)

        # Calculate difference
        difference = abs(total_original - total_new)
        percentage_diff = (difference / total_original * 100) if total_original > 0 else 0

        comparison['total_original'] = total_original
        comparison['total_new'] = total_new
        comparison['difference'] = difference
        comparison['percentage_difference'] = percentage_diff

        # Check if parts sum to whole
        parts_sum = ldl + hdl + vldl
        comparison['parts_sum'] = parts_sum
        comparison['sum_validation'] = "✓ Match" if percentage_diff < 10 else "⚠ Mismatch"

        # Breakdown percentages
        if total_new > 0:
            comparison['ldl_percentage'] = (ldl / total_new) * 100
            comparison['hdl_percentage'] = (hdl / total_new) * 100
            comparison['vldl_percentage'] = (vldl / total_new) * 100

        logger.info(f"Cholesterol comparison: Original={total_original:.1f}, "
                   f"New={total_new:.1f} (LDL={ldl:.1f} + HDL={hdl:.1f} + VLDL={vldl:.1f}), "
                   f"Diff={percentage_diff:.1f}%")

        return comparison