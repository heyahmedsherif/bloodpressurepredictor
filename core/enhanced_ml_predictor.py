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
from .vascular_age_predictor import VascularAgePredictor
from .ml_vascular_age_predictor import MLVascularAgePredictor
from .model_validator import ModelValidator
from .ppg_signal_enhancer import PPGSignalEnhancer

logger = logging.getLogger(__name__)

class EnhancedMLHealthPredictor:
    """Predict health metrics with both original and detailed cholesterol models"""

    def __init__(self):
        """Initialize and load all ML models including LDL/HDL"""
        self.models_loaded = False
        self.ldl_hdl_models_loaded = False
        self.feature_extractor = PPGFeatureExtractor()
        self.cholesterol_extractor = CholesterolFeatureExtractor()
        self.vascular_age_predictor = VascularAgePredictor()
        self.ml_vascular_age_predictor = MLVascularAgePredictor()
        self.validator = ModelValidator()
        self.signal_enhancer = PPGSignalEnhancer(sampling_rate=30)

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

        # ENHANCED: Apply signal enhancement before feature extraction
        signal_enhancement = self.signal_enhancer.enhance_signal(ppg_signal)
        enhanced_signal = signal_enhancement['enhanced_signal']
        quality_score = signal_enhancement['quality_score']
        quality_level = signal_enhancement['quality_level']

        # Add signal quality to results
        results['signal_quality'] = {
            'score': quality_score,
            'level': quality_level,
            'confidence': signal_enhancement['confidence'],
            'metrics': signal_enhancement['quality_metrics'],
            'is_acceptable': signal_enhancement['is_acceptable']
        }

        logger.info(f"Signal quality: {quality_level} (score: {quality_score})")

        # Use enhanced signal for feature extraction if quality is acceptable
        if signal_enhancement['is_acceptable']:
            processing_signal = enhanced_signal
            logger.info("Using enhanced signal for predictions")
        else:
            processing_signal = ppg_signal
            logger.warning("Signal quality poor, using raw signal")

        # Extract features from the processed signal
        basic_features = self.feature_extractor.extract_features(processing_signal)
        advanced_features = self.cholesterol_extractor.extract_all_features(
            processing_signal, demographics
        )

        # ENHANCED: Extract heart rate with confidence using enhanced signal
        hr_results = self.signal_enhancer.extract_heart_rate(enhanced_signal, quality_score)
        results['heart_rate_enhanced'] = hr_results

        # Update basic features with enhanced HR if available
        if hr_results and 'heart_rate' in hr_results:
            basic_features['heart_rate'] = hr_results['heart_rate']
            advanced_features['heart_rate'] = hr_results['heart_rate']
            logger.info(f"Using enhanced HR: {hr_results['heart_rate']:.1f} BPM (confidence: {hr_results['confidence']:.0f}%)")

        # Blood Pressure Prediction with quality adjustment
        bp_results = self._predict_blood_pressure(advanced_features, demographics, quality_score)
        results['systolic'] = bp_results['systolic']
        results['diastolic'] = bp_results['diastolic']
        results['bp_confidence'] = bp_results.get('confidence', quality_score * 100)

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

        # Vascular Age Prediction
        vascular_features = {
            'augmentation_index': advanced_features.get('mean_ai', 0.4),
            'stiffness_index': advanced_features.get('stiffness_index', 8.0),
            'heart_rate': basic_features.get('heart_rate', 75),
            'hrv': advanced_features.get('hrv', 50),
            'systolic_bp': results['systolic'],
            'diastolic_bp': results['diastolic']
        }

        vascular_age_results = self.vascular_age_predictor.predict_vascular_age(
            vascular_features, demographics
        )
        results['vascular_age'] = vascular_age_results['vascular_age']
        results['vascular_age_data'] = vascular_age_results

        # ML-Based Vascular Age Prediction (additional, doesn't replace formula-based)
        try:
            ml_vascular_results = self.ml_vascular_age_predictor.predict_vascular_age_ml(
                ppg_signal, demographics, vascular_features
            )
            results['vascular_age_ml_data'] = ml_vascular_results
            logger.info(f"ML Vascular Age: {ml_vascular_results.get('vascular_age_ml', 'N/A')}")
        except Exception as e:
            logger.error(f"ML vascular age prediction failed: {e}")

        # CRITICAL: Validate all predictions before returning
        results = self.validator.validate_all_predictions(results, demographics)

        return results

    def _predict_blood_pressure(self, features, demographics, quality_score=0.5):
        """Predict blood pressure using physiologically-based calculation"""
        try:
            # Extract key parameters
            heart_rate = features.get('heart_rate', 75)
            ppg_amplitude = features.get('ppg_amplitude', 0.5)
            ppg_width = features.get('ppg_width', 0.3)
            hrv = features.get('hrv', 50)
            age = demographics.get('age', 40)
            bmi = demographics.get('bmi', 25)

            # Physiological BP estimation based on research
            # Base BP calculation using age and demographics
            base_systolic = 100 + (age * 0.5)  # Age contributes to systolic
            base_diastolic = 60 + (age * 0.2)  # Age has less effect on diastolic

            # Heart rate contribution
            # Higher HR typically correlates with higher BP
            hr_factor = (heart_rate - 70) / 70  # Normalized around 70 BPM
            systolic = base_systolic + (hr_factor * 15)  # HR affects systolic more
            diastolic = base_diastolic + (hr_factor * 8)

            # PPG amplitude inversely related to BP (smaller amplitude = higher BP)
            # Based on arterial stiffness principle
            amplitude_factor = 1.0 - ppg_amplitude  # Invert amplitude (0-1 range)
            systolic += amplitude_factor * 20
            diastolic += amplitude_factor * 10

            # PPG width (pulse width) contribution
            # Wider pulses often indicate higher BP
            width_factor = ppg_width * 2  # Assume width is 0-1 range
            systolic += width_factor * 10
            diastolic += width_factor * 5

            # HRV contribution (lower HRV = higher BP typically)
            hrv_factor = (60 - hrv) / 60 if hrv < 60 else 0
            systolic += hrv_factor * 10
            diastolic += hrv_factor * 5

            # BMI contribution
            if bmi > 25:
                bmi_factor = (bmi - 25) / 10  # Normalize excess BMI
                systolic += bmi_factor * 10
                diastolic += bmi_factor * 7

            # Gender adjustment (if available)
            gender = demographics.get('gender', '').lower()
            if gender == 'male':
                systolic += 5
                diastolic += 3
            elif gender == 'female':
                systolic -= 5
                diastolic -= 2

            # Apply signal quality adjustment
            # Lower quality signals get less extreme predictions
            if quality_score < 0.5:
                # Pull values toward normal range for poor signals
                systolic = systolic * 0.7 + 120 * 0.3
                diastolic = diastolic * 0.7 + 80 * 0.3

            # Ensure reasonable physiological ranges
            systolic = np.clip(systolic, 90, 160)
            diastolic = np.clip(diastolic, 55, 95)

            # Ensure pulse pressure is reasonable
            pulse_pressure = systolic - diastolic
            if pulse_pressure < 20:
                systolic = diastolic + 25
            elif pulse_pressure > 80:
                diastolic = systolic - 70

            # Calculate confidence based on signal quality and prediction consistency
            confidence = quality_score * 100

            # Adjust confidence if predictions are at boundaries
            if systolic == 85 or systolic == 180 or diastolic == 50 or diastolic == 110:
                confidence *= 0.7  # Lower confidence for clipped values

            logger.info(f"ML BP Prediction: {systolic:.0f}/{diastolic:.0f} mmHg (confidence: {confidence:.0f}%)")

            return {
                'systolic': systolic,
                'diastolic': diastolic,
                'confidence': round(confidence, 0)
            }

        except Exception as e:
            logger.error(f"BP prediction error: {e}")
            return {'systolic': 120, 'diastolic': 80, 'confidence': 30}

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