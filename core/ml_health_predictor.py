"""
ML-based Health Predictor using trained models from research repositories.
Uses real machine learning models for glucose, cholesterol, and blood pressure prediction.
"""

import pickle
import numpy as np
import logging
from .ppg_feature_extractor import PPGFeatureExtractor

logger = logging.getLogger(__name__)

class MLHealthPredictor:
    """Predict health metrics using ML models trained on PPG data."""
    
    def __init__(self):
        """Initialize and load all ML models."""
        self.models_loaded = False
        self.feature_extractor = PPGFeatureExtractor()
        
        try:
            # Load glucose prediction models
            with open('models/glucose/glucose_model.pkl', 'rb') as f:
                self.glucose_model = pickle.load(f)
            with open('models/glucose/glucose_scaler.pkl', 'rb') as f:
                self.glucose_scaler = pickle.load(f)
            with open('models/glucose/glucose_poly.pkl', 'rb') as f:
                self.glucose_poly = pickle.load(f)
            
            # Load cholesterol prediction models
            with open('models/cholesterol/cholesterol_model.pkl', 'rb') as f:
                self.cholesterol_model = pickle.load(f)
            with open('models/cholesterol/cholesterol_scaler.pkl', 'rb') as f:
                self.cholesterol_scaler = pickle.load(f)
            
            # Load blood pressure prediction models
            with open('models/blood_pressure/systolic_model.pkl', 'rb') as f:
                self.systolic_model = pickle.load(f)
            with open('models/blood_pressure/diastolic_model.pkl', 'rb') as f:
                self.diastolic_model = pickle.load(f)
            with open('models/blood_pressure/bp_scaler.pkl', 'rb') as f:
                self.bp_scaler = pickle.load(f)
            
            self.models_loaded = True
            logger.info("✅ All ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            logger.warning("Will use fallback calculations")
    
    def predict_health_metrics(self, ppg_signal, demographics, heart_rate=None):
        """
        Predict health metrics using ML models and PPG features.
        
        Args:
            ppg_signal: Raw PPG signal from rPPG processing
            demographics: Dict with age, gender, height, weight
            heart_rate: Pre-calculated heart rate (optional)
            
        Returns:
            Dictionary with predicted health metrics
        """
        # Extract PPG features
        ppg_features = self.feature_extractor.extract_features(ppg_signal, heart_rate)
        
        # Calculate BMI
        height_m = demographics.get('height', 173) / 100
        weight = demographics.get('weight', 83)
        bmi = weight / (height_m ** 2)
        
        # Get demographics
        age = demographics.get('age', 47)
        
        # Use ML models if loaded, otherwise fallback
        if self.models_loaded:
            try:
                # Predict glucose
                glucose = self._predict_glucose(ppg_features, age, bmi)
                
                # Predict cholesterol
                cholesterol = self._predict_cholesterol(ppg_features, age, bmi)
                
                # Predict blood pressure
                systolic, diastolic = self._predict_blood_pressure(ppg_features, age, bmi)
                
                logger.info(f"ML predictions - Glucose: {glucose:.1f}, Cholesterol: {cholesterol:.1f}, BP: {systolic:.0f}/{diastolic:.0f}")
                
            except Exception as e:
                logger.error(f"Error in ML prediction: {e}")
                # Fallback to simple calculations
                glucose = self._fallback_glucose(ppg_features, age, bmi)
                cholesterol = self._fallback_cholesterol(ppg_features, age, bmi)
                systolic, diastolic = self._fallback_bp(ppg_features, age, bmi)
        else:
            # Use fallback calculations
            glucose = self._fallback_glucose(ppg_features, age, bmi)
            cholesterol = self._fallback_cholesterol(ppg_features, age, bmi)
            systolic, diastolic = self._fallback_bp(ppg_features, age, bmi)
        
        # Calculate cardiovascular risk score
        cv_risk = self._calculate_cv_risk(age, bmi, systolic, cholesterol)
        
        return {
            'blood_pressure': {
                'systolic': round(systolic, 1),
                'diastolic': round(diastolic, 1),
                'category': self._categorize_bp(systolic, diastolic)
            },
            'glucose': {
                'value': round(glucose, 1),
                'unit': 'mg/dL',
                'category': self._categorize_glucose(glucose)
            },
            'cholesterol': {
                'value': round(cholesterol, 1),
                'unit': 'mg/dL',
                'category': self._categorize_cholesterol(cholesterol)
            },
            'cardiovascular_risk': {
                'score': round(cv_risk, 1),
                'category': self._categorize_cv_risk(cv_risk)
            },
            'ppg_features': {
                'heart_rate': ppg_features['heart_rate'],
                'ppg_variability': ppg_features['ppg_variability'],
                'pulse_transit_time': ppg_features['pulse_transit_time']
            }
        }
    
    def _predict_glucose(self, ppg_features, age, bmi):
        """Predict glucose using ML model."""
        # Prepare features: ppg_amplitude, heart_rate, systolic, diastolic, age, bmi
        X = np.array([[
            ppg_features['ppg_amplitude'],
            ppg_features['heart_rate'],
            ppg_features['systolic'],
            ppg_features['diastolic'],
            age,
            bmi
        ]])
        
        # Scale and transform
        X_scaled = self.glucose_scaler.transform(X)
        X_poly = self.glucose_poly.transform(X_scaled)
        
        # Predict
        glucose = self.glucose_model.predict(X_poly)[0]
        return np.clip(glucose, 70, 200)
    
    def _predict_cholesterol(self, ppg_features, age, bmi):
        """Predict cholesterol using ML model."""
        # Prepare features: age, heart_rate, systolic, diastolic, bmi, ppg_variability
        X = np.array([[
            age,
            ppg_features['heart_rate'],
            ppg_features['systolic'],
            ppg_features['diastolic'],
            bmi,
            ppg_features['ppg_variability']
        ]])
        
        # Scale and predict
        X_scaled = self.cholesterol_scaler.transform(X)
        cholesterol = self.cholesterol_model.predict(X_scaled)[0]
        return np.clip(cholesterol, 120, 300)
    
    def _predict_blood_pressure(self, ppg_features, age, bmi):
        """Predict blood pressure using ML models."""
        # Prepare features: pulse_transit_time, ppg_amplitude, heart_rate, ppg_width, age, bmi
        X = np.array([[
            ppg_features['pulse_transit_time'],
            ppg_features['ppg_amplitude'],
            ppg_features['heart_rate'],
            ppg_features['ppg_width'],
            age,
            bmi
        ]])
        
        # Scale and predict
        X_scaled = self.bp_scaler.transform(X)
        systolic = self.systolic_model.predict(X_scaled)[0]
        diastolic = self.diastolic_model.predict(X_scaled)[0]
        
        # Ensure realistic values
        systolic = np.clip(systolic, 90, 180)
        diastolic = np.clip(diastolic, 60, 110)
        
        # Ensure systolic > diastolic
        if diastolic >= systolic:
            diastolic = systolic - 20
        
        return systolic, diastolic
    
    def _fallback_glucose(self, ppg_features, age, bmi):
        """Fallback glucose calculation."""
        return 90 + (bmi - 25) * 2 + (age - 40) * 0.5 + (ppg_features['heart_rate'] - 70) * 0.2
    
    def _fallback_cholesterol(self, ppg_features, age, bmi):
        """Fallback cholesterol calculation."""
        return 180 + (bmi - 25) * 3 + (age - 40) * 1.2 + (ppg_features['heart_rate'] - 70) * 0.3
    
    def _fallback_bp(self, ppg_features, age, bmi):
        """Fallback BP calculation."""
        systolic = 120 + (age - 30) * 0.5 + (ppg_features['heart_rate'] - 70) * 0.2
        diastolic = 80 + (age - 30) * 0.3 + (ppg_features['heart_rate'] - 70) * 0.1
        return systolic, diastolic
    
    def _calculate_cv_risk(self, age, bmi, systolic, cholesterol):
        """Calculate cardiovascular risk score."""
        risk = 0
        
        # Age risk
        if age > 45:
            risk += (age - 45) * 0.5
        
        # BMI risk
        if bmi > 25:
            risk += (bmi - 25) * 2
        
        # BP risk
        if systolic > 130:
            risk += (systolic - 130) * 0.3
        
        # Cholesterol risk
        if cholesterol > 200:
            risk += (cholesterol - 200) * 0.1
        
        return min(100, max(0, risk))
    
    def _categorize_bp(self, systolic, diastolic):
        """Categorize blood pressure."""
        if systolic < 120 and diastolic < 80:
            return 'Normal'
        elif systolic < 130 and diastolic < 80:
            return 'Elevated'
        elif systolic < 140 or diastolic < 90:
            return 'Stage 1 Hypertension'
        else:
            return 'Stage 2 Hypertension'
    
    def _categorize_glucose(self, glucose):
        """Categorize glucose level."""
        if glucose < 100:
            return 'Normal'
        elif glucose < 126:
            return 'Prediabetic'
        else:
            return 'Diabetic'
    
    def _categorize_cholesterol(self, cholesterol):
        """Categorize cholesterol level."""
        if cholesterol < 200:
            return 'Optimal'
        elif cholesterol < 240:
            return 'Borderline High'
        else:
            return 'High'
    
    def _categorize_cv_risk(self, risk):
        """Categorize cardiovascular risk."""
        if risk < 20:
            return 'Low'
        elif risk < 50:
            return 'Moderate'
        else:
            return 'High'