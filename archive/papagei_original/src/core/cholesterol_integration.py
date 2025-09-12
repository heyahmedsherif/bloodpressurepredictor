"""
Cholesterol and Cardiovascular Risk Prediction Integration Module

This module integrates cardiovascular risk prediction functionality using machine learning models
based on the research from external/cholesterol-cvd-prediction repository (Framingham Heart Study).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

class CardiovascularRiskPredictor:
    """
    Cardiovascular risk predictor using clinical features and PPG-derived metrics.
    Based on Framingham Heart Study approach from external/cholesterol-cvd-prediction.
    """
    
    def __init__(self, model_type: str = 'neural_network'):
        """
        Initialize cardiovascular risk predictor.
        
        Args:
            model_type: Type of ML model ('logistic_regression', 'neural_network', 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        
        # Key features identified from Framingham Heart Study
        self.feature_names = [
            'age', 'sex', 'education', 'cigs_per_day', 'bp_meds', 
            'prevalent_stroke', 'prevalent_hyp', 'diabetes', 
            'total_cholesterol', 'bmi', 'heart_rate', 'glucose', 'pulse_pressure'
        ]
        self.is_trained = False
        
    def _create_model(self):
        """Create the appropriate ML model based on model_type."""
        if self.model_type == 'logistic_regression':
            return LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'neural_network':
            # Tuned neural network based on research results
            return MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=1000,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def create_feature_vector(self, patient_data: Dict[str, Union[int, float]], 
                             ppg_metrics: Dict[str, float]) -> np.ndarray:
        """
        Create feature vector from patient data and PPG-derived metrics.
        
        Args:
            patient_data: Dictionary containing patient clinical information
            ppg_metrics: Dictionary containing PPG-derived cardiovascular metrics
        
        Returns:
            Feature vector for cardiovascular risk prediction
        """
        # Calculate pulse pressure from blood pressure readings
        systolic = ppg_metrics.get('systolic_bp', patient_data.get('systolic_bp', 120))
        diastolic = ppg_metrics.get('diastolic_bp', patient_data.get('diastolic_bp', 80))
        pulse_pressure = systolic - diastolic
        
        # Estimate cholesterol if not provided (based on age/risk factors)
        total_cholesterol = patient_data.get('total_cholesterol')
        if total_cholesterol is None:
            total_cholesterol = self._estimate_cholesterol(patient_data, ppg_metrics)
        
        # Create feature vector matching Framingham study
        features = [
            patient_data.get('age', 30),                                    # age
            1 if patient_data.get('gender', '').lower() == 'male' else 0,   # sex (1=male, 0=female)  
            patient_data.get('education', 2),                               # education level (1-4)
            patient_data.get('cigarettes_per_day', 0),                      # cigs_per_day
            1 if patient_data.get('bp_medication', False) else 0,           # bp_meds
            1 if patient_data.get('previous_stroke', False) else 0,         # prevalent_stroke
            1 if patient_data.get('hypertension', False) else 0,            # prevalent_hyp
            1 if patient_data.get('diabetes', False) else 0,                # diabetes
            total_cholesterol,                                              # total_cholesterol
            self._calculate_bmi(patient_data),                              # bmi
            ppg_metrics.get('heart_rate', patient_data.get('heart_rate', 75)), # heart_rate
            patient_data.get('glucose_level', 90),                          # glucose
            pulse_pressure                                                  # pulse_pressure
        ]
        
        return np.array(features).reshape(1, -1)
    
    def _calculate_bmi(self, patient_data: Dict) -> float:
        """Calculate BMI from height and weight."""
        height_m = patient_data.get('height_cm', 170) / 100
        weight_kg = patient_data.get('weight_kg', 70)
        return weight_kg / (height_m ** 2)
    
    def _estimate_cholesterol(self, patient_data: Dict, ppg_metrics: Dict) -> float:
        """Estimate total cholesterol based on age, gender, and cardiovascular metrics."""
        age = patient_data.get('age', 30)
        is_male = patient_data.get('gender', '').lower() == 'male'
        
        # Base cholesterol levels by demographic
        if is_male:
            base_cholesterol = 180 + (age - 30) * 0.8
        else:
            base_cholesterol = 170 + (age - 30) * 1.2
        
        # Adjust for risk factors
        if patient_data.get('diabetes', False):
            base_cholesterol += 20
        if patient_data.get('hypertension', False):
            base_cholesterol += 15
        if patient_data.get('cigarettes_per_day', 0) > 0:
            base_cholesterol += 10
            
        result = min(max(float(base_cholesterol), 120.0), 350.0)  # Physiological bounds
        return result
    
    def train_model(self, training_data: pd.DataFrame, target_column: str = 'TenYearCHD'):
        """
        Train the cardiovascular risk prediction model.
        
        Args:
            training_data: DataFrame with Framingham Heart Study format features
            target_column: Target column indicating 10-year CHD risk
        """
        # Prepare features and target
        X = training_data[self.feature_names]
        y = training_data[target_column]
        
        # Handle missing values
        X_imputed = self.imputer.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics (focusing on recall as per research)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Model Training Complete - {self.model_type}")
        print(f"AUC Score: {auc_score:.3f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        return {'auc': auc_score}
    
    def predict_cardiovascular_risk(self, patient_data: Dict[str, Union[int, float]], 
                                  ppg_metrics: Dict[str, float]) -> Dict[str, Union[float, str]]:
        """
        Predict 10-year cardiovascular risk from patient data and PPG metrics.
        
        Args:
            patient_data: Patient clinical and demographic information
            ppg_metrics: PPG-derived cardiovascular metrics
        
        Returns:
            Dictionary with risk prediction and clinical insights
        """
        # Always use Framingham Risk Score since we don't have trained models yet
        return self._framingham_risk_score(patient_data, ppg_metrics)
    
    def _framingham_risk_score(self, patient_data: Dict, ppg_metrics: Dict) -> Dict[str, Union[float, str]]:
        """
        Fallback: Calculate traditional Framingham Risk Score when no trained model available.
        """
        age = patient_data.get('age', 30)
        is_male = patient_data.get('gender', '').lower() == 'male'
        
        # Ensure all values are numbers, not None
        total_chol = patient_data.get('total_cholesterol')
        if total_chol is None:
            total_chol = self._estimate_cholesterol(patient_data, ppg_metrics)
        
        hdl = patient_data.get('hdl_cholesterol')
        if hdl is None:
            hdl = 50  # Default HDL value
        
        systolic = ppg_metrics.get('systolic_bp')
        if systolic is None:
            systolic = 120  # Default systolic BP
        
        smoking = patient_data.get('cigarettes_per_day', 0) > 0
        diabetes = patient_data.get('diabetes', False)
        
        # Simplified Framingham calculation
        points = 0
        
        # Ensure all values are properly typed
        age = float(age) if age is not None else 30.0
        total_chol = float(total_chol) if total_chol is not None else 200.0
        hdl = float(hdl) if hdl is not None else 50.0
        systolic = float(systolic) if systolic is not None else 120.0
        
        # Age points
        if is_male:
            points += max(0, (age - 35) // 5 * 2)
        else:
            points += max(0, (age - 35) // 5 * 3)
        
        # Cholesterol points
        points += max(0, (total_chol - 160) // 40)
        
        # HDL points (protective)
        if hdl >= 60:
            points -= 1
        elif hdl < 40:
            points += 2
        
        # Blood pressure points
        points += max(0, (systolic - 120) // 20)
        
        # Risk factors
        if smoking:
            points += 4 if is_male else 3
        if diabetes:
            points += 3 if is_male else 4
        
        # Convert points to risk probability (approximation)
        risk_probability = min(0.95, max(0.01, points * 0.03))
        
        return {
            '10_year_chd_risk_probability': round(risk_probability, 3),
            'high_risk_prediction': risk_probability > 0.20,
            'risk_category': self._categorize_risk(risk_probability),
            'recommendations': self._get_recommendations(risk_probability, patient_data),
            'model_used': 'framingham_risk_score',
            'confidence_level': 'moderate'
        }
    
    def _categorize_risk(self, risk_probability: float) -> str:
        """Categorize cardiovascular risk based on probability."""
        if risk_probability < 0.075:
            return "Low Risk (<7.5%)"
        elif risk_probability < 0.20:
            return "Intermediate Risk (7.5-20%)"
        else:
            return "High Risk (>20%)"
    
    def _get_recommendations(self, risk_probability: float, patient_data: Dict) -> List[str]:
        """Generate personalized recommendations based on risk level."""
        recommendations = []
        
        if risk_probability >= 0.20:
            recommendations.extend([
                "Consider statin therapy consultation with physician",
                "Implement intensive lifestyle modifications",
                "Regular cardiovascular monitoring recommended"
            ])
        elif risk_probability >= 0.075:
            recommendations.extend([
                "Consider statin therapy if additional risk factors present",
                "Lifestyle modifications strongly recommended",
                "Monitor blood pressure and cholesterol regularly"
            ])
        else:
            recommendations.extend([
                "Continue healthy lifestyle practices",
                "Regular check-ups for cardiovascular health",
                "Maintain current health management approach"
            ])
        
        # Specific recommendations based on risk factors
        if patient_data.get('cigarettes_per_day', 0) > 0:
            recommendations.append("Smoking cessation is critical for cardiovascular health")
        
        if patient_data.get('diabetes', False):
            recommendations.append("Optimize diabetes management for cardiovascular protection")
        
        if patient_data.get('hypertension', False):
            recommendations.append("Blood pressure control is essential")
        
        return recommendations
    
    def save_model(self, filepath: str):
        """Save the trained model to file."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load a pre-trained model from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.imputer = model_data['imputer']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_trained = True

class CholesterolCardiovascularIntegration:
    """
    High-level integration class for cholesterol and cardiovascular risk prediction in the PaPaGei system.
    """
    
    def __init__(self):
        self.risk_predictor = CardiovascularRiskPredictor(model_type='neural_network')
        self.supported_features = [
            'Framingham Heart Study Model',
            'Neural Network Prediction', 
            'PPG-Derived Metrics Integration',
            'Clinical Risk Categorization',
            'Personalized Recommendations'
        ]
    
    def predict_from_papagei_format(self, papagei_data: Dict) -> Dict[str, Union[float, str, List[str]]]:
        """
        Predict cardiovascular risk from PaPaGei format data.
        
        Args:
            papagei_data: Dictionary containing patient and PPG data in PaPaGei format
        
        Returns:
            Cardiovascular risk prediction results
        """
        # Extract patient clinical data
        patient_data = {
            'age': papagei_data.get('age', 30),
            'gender': papagei_data.get('gender', ''),
            'height_cm': papagei_data.get('height_cm', 170),
            'weight_kg': papagei_data.get('weight_kg', 70),
            'education': papagei_data.get('education_level', 2),
            'cigarettes_per_day': papagei_data.get('cigarettes_per_day', 0),
            'bp_medication': papagei_data.get('bp_medication', False),
            'previous_stroke': papagei_data.get('previous_stroke', False),
            'hypertension': papagei_data.get('hypertension', False),
            'diabetes': papagei_data.get('diabetes', False),
            'total_cholesterol': papagei_data.get('total_cholesterol'),
            'hdl_cholesterol': papagei_data.get('hdl_cholesterol'),
            'heart_rate': papagei_data.get('heart_rate', 75),
            'glucose_level': papagei_data.get('glucose_level', 90)
        }
        
        # Extract PPG-derived metrics
        ppg_metrics = {
            'heart_rate': papagei_data.get('heart_rate', 75),
            'systolic_bp': papagei_data.get('blood_pressure', {}).get('systolic', 120),
            'diastolic_bp': papagei_data.get('blood_pressure', {}).get('diastolic', 80)
        }
        
        # Add glucose prediction if available
        if 'predicted_glucose' in papagei_data:
            patient_data['glucose_level'] = papagei_data['predicted_glucose']
        
        return self.risk_predictor.predict_cardiovascular_risk(patient_data, ppg_metrics)
    
    def calculate_cholesterol_ratios(self, lipid_panel: Dict[str, float]) -> Dict[str, Union[float, str]]:
        """
        Calculate important cholesterol ratios and interpretations.
        
        Args:
            lipid_panel: Dictionary containing lipid measurements
        
        Returns:
            Cholesterol ratios and interpretations
        """
        total_chol = lipid_panel.get('total_cholesterol', 200)
        hdl = lipid_panel.get('hdl_cholesterol', 50)
        ldl = lipid_panel.get('ldl_cholesterol', 100)
        triglycerides = lipid_panel.get('triglycerides', 150)
        
        # Calculate ratios
        tc_hdl_ratio = total_chol / hdl if hdl > 0 else float('inf')
        ldl_hdl_ratio = ldl / hdl if hdl > 0 else float('inf')
        
        return {
            'total_cholesterol_hdl_ratio': round(tc_hdl_ratio, 2),
            'ldl_hdl_ratio': round(ldl_hdl_ratio, 2),
            'total_cholesterol_status': self._interpret_cholesterol(total_chol),
            'hdl_status': self._interpret_hdl(hdl),
            'ldl_status': self._interpret_ldl(ldl),
            'triglycerides_status': self._interpret_triglycerides(triglycerides),
            'overall_lipid_risk': self._assess_lipid_risk(tc_hdl_ratio, ldl, hdl)
        }
    
    def _interpret_cholesterol(self, total_chol: float) -> str:
        """Interpret total cholesterol levels."""
        if total_chol < 200:
            return "Desirable"
        elif total_chol < 240:
            return "Borderline High"
        else:
            return "High"
    
    def _interpret_hdl(self, hdl: float) -> str:
        """Interpret HDL cholesterol levels."""
        if hdl >= 60:
            return "High (Protective)"
        elif hdl >= 40:
            return "Normal"
        else:
            return "Low (Risk Factor)"
    
    def _interpret_ldl(self, ldl: float) -> str:
        """Interpret LDL cholesterol levels."""
        if ldl < 100:
            return "Optimal"
        elif ldl < 130:
            return "Near Optimal"
        elif ldl < 160:
            return "Borderline High"
        else:
            return "High"
    
    def _interpret_triglycerides(self, trig: float) -> str:
        """Interpret triglyceride levels."""
        if trig < 150:
            return "Normal"
        elif trig < 200:
            return "Borderline High"
        elif trig < 500:
            return "High"
        else:
            return "Very High"
    
    def _assess_lipid_risk(self, tc_hdl_ratio: float, ldl: float, hdl: float) -> str:
        """Assess overall lipid-related cardiovascular risk."""
        risk_factors = 0
        
        if tc_hdl_ratio > 5:
            risk_factors += 2
        elif tc_hdl_ratio > 4:
            risk_factors += 1
            
        if ldl >= 160:
            risk_factors += 2
        elif ldl >= 130:
            risk_factors += 1
            
        if hdl < 40:
            risk_factors += 1
        
        if risk_factors >= 3:
            return "High Risk"
        elif risk_factors >= 1:
            return "Moderate Risk"
        else:
            return "Low Risk"
    
    def get_feature_requirements(self) -> Dict[str, Dict]:
        """Get information about required features for cardiovascular prediction."""
        return {
            'required': {
                'age': 'Patient age in years',
                'gender': 'Patient gender (male/female)',
                'blood_pressure': 'Blood pressure readings (systolic/diastolic)'
            },
            'optional': {
                'total_cholesterol': 'Total cholesterol level (mg/dL)',
                'hdl_cholesterol': 'HDL cholesterol level (mg/dL)',
                'cigarettes_per_day': 'Number of cigarettes smoked per day',
                'diabetes': 'Diabetes status (boolean)',
                'hypertension': 'Hypertension status (boolean)',
                'previous_stroke': 'Previous stroke history (boolean)',
                'bp_medication': 'Blood pressure medication use (boolean)'
            },
            'output': {
                '10_year_chd_risk_probability': 'Probability of coronary heart disease in 10 years',
                'risk_category': 'Risk categorization (Low/Intermediate/High)',
                'recommendations': 'Personalized clinical recommendations',
                'cholesterol_analysis': 'Lipid panel interpretation if provided'
            }
        }