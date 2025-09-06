"""
Glucose Prediction Integration Module

This module integrates PPG-based glucose prediction functionality using machine learning models
based on the research from external/glucose-prediction repository.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import joblib
import os
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class GlucosePredictorFromPPG:
    """
    Glucose level predictor using PPG signal features and demographic information.
    Based on the approach from external/glucose-prediction repository.
    """
    
    def __init__(self, model_type: str = 'polynomial_regression'):
        """
        Initialize glucose predictor.
        
        Args:
            model_type: Type of ML model ('linear_regression', 'decision_tree', 'polynomial_regression')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.poly_features = None
        self.feature_names = [
            'ppg_signal_mv', 'heart_rate_bpm', 'systolic_peak_mmhg', 
            'diastolic_peak_mmhg', 'pulse_area', 'gender', 
            'height_cm', 'weight_kg', 'age_range'
        ]
        self.is_trained = False
        
    def _create_model(self):
        """Create the appropriate ML model based on model_type."""
        if self.model_type == 'linear_regression':
            return LinearRegression()
        elif self.model_type == 'decision_tree':
            return DecisionTreeRegressor(random_state=42)
        elif self.model_type == 'polynomial_regression':
            self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
            return LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def extract_ppg_features(self, ppg_signal: np.ndarray, 
                           demographic_info: Dict[str, Union[int, float]]) -> np.ndarray:
        """
        Extract features from PPG signal and combine with demographic information.
        
        Args:
            ppg_signal: PPG signal array (mV)
            demographic_info: Dictionary containing demographic and physiological info
                - heart_rate_bpm: Heart rate in beats per minute
                - systolic_peak_mmhg: Systolic blood pressure
                - diastolic_peak_mmhg: Diastolic blood pressure
                - gender: 1 for Male, 0 for Female
                - height_cm: Height in centimeters
                - weight_kg: Weight in kilograms
                - age_range: Age category (encoded as integer)
        
        Returns:
            Feature vector for glucose prediction
        """
        # PPG signal features
        ppg_mean = np.mean(ppg_signal)
        pulse_area = np.trapz(ppg_signal)  # Area under curve
        
        # Create feature vector
        features = [
            ppg_mean,  # PPG Signal (mV)
            demographic_info.get('heart_rate_bpm', 75),
            demographic_info.get('systolic_peak_mmhg', 120),
            demographic_info.get('diastolic_peak_mmhg', 80),
            pulse_area,  # Pulse Area
            demographic_info.get('gender', 0),  # 0=Female, 1=Male
            demographic_info.get('height_cm', 170),
            demographic_info.get('weight_kg', 70),
            demographic_info.get('age_range', 2)  # Encoded age range
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train_model(self, training_data: pd.DataFrame, target_column: str = 'blood_glucose_level'):
        """
        Train the glucose prediction model using provided training data.
        
        Args:
            training_data: DataFrame with features and target glucose levels
            target_column: Name of the target column containing glucose levels
        """
        # Prepare features and target
        X = training_data[self.feature_names]
        y = training_data[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        self.model = self._create_model()
        
        if self.model_type == 'polynomial_regression':
            # Apply polynomial features
            X_train_poly = self.poly_features.fit_transform(X_train_scaled)
            X_test_poly = self.poly_features.transform(X_test_scaled)
            self.model.fit(X_train_poly, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_poly)
        else:
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Training Complete - {self.model_type}")
        print(f"Mean Absolute Error: {mae:.2f} mg/dL")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.3f}")
        
        self.is_trained = True
        return {'mae': mae, 'mse': mse, 'r2': r2}
    
    def predict_glucose(self, ppg_signal: np.ndarray, 
                       demographic_info: Dict[str, Union[int, float]]) -> Dict[str, Union[float, str]]:
        """
        Predict blood glucose level from PPG signal and demographic information.
        
        Args:
            ppg_signal: PPG signal array
            demographic_info: Dictionary with demographic and physiological information
        
        Returns:
            Dictionary with glucose prediction and confidence metrics
        """
        if not self.is_trained:
            # Use pre-trained fallback model with default parameters
            return self._fallback_prediction(ppg_signal, demographic_info)
        
        # Extract features
        features = self.extract_ppg_features(ppg_signal, demographic_info)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Apply polynomial features if needed
        if self.model_type == 'polynomial_regression' and self.poly_features:
            features_scaled = self.poly_features.transform(features_scaled)
        
        # Make prediction
        glucose_prediction = self.model.predict(features_scaled)[0]
        
        # Calculate confidence based on feature quality
        confidence = self._calculate_confidence(ppg_signal, demographic_info)
        
        # Interpret results
        interpretation = self._interpret_glucose_level(glucose_prediction)
        
        return {
            'predicted_glucose_mg_dl': round(float(glucose_prediction), 1),
            'confidence_score': round(confidence, 2),
            'interpretation': interpretation,
            'model_used': self.model_type,
            'features_used': len(self.feature_names)
        }
    
    def _fallback_prediction(self, ppg_signal: np.ndarray, 
                           demographic_info: Dict[str, Union[int, float]]) -> Dict[str, Union[float, str]]:
        """Fallback prediction using statistical approach when no trained model available."""
        # Simple heuristic-based prediction
        ppg_mean = np.mean(ppg_signal)
        age = demographic_info.get('age_range', 2) * 15 + 25  # Approximate age
        bmi = demographic_info.get('weight_kg', 70) / ((demographic_info.get('height_cm', 170) / 100) ** 2)
        
        # Basic glucose estimation based on physiological relationships
        base_glucose = 90  # Normal fasting glucose
        ppg_factor = (ppg_mean - 1.0) * 20  # PPG influence
        age_factor = (age - 40) * 0.5  # Age influence
        bmi_factor = (bmi - 22) * 2  # BMI influence
        
        predicted_glucose = base_glucose + ppg_factor + age_factor + bmi_factor
        predicted_glucose = max(70, min(300, predicted_glucose))  # Physiological bounds
        
        return {
            'predicted_glucose_mg_dl': round(float(predicted_glucose), 1),
            'confidence_score': 0.6,
            'interpretation': self._interpret_glucose_level(predicted_glucose),
            'model_used': 'heuristic_fallback',
            'features_used': 3
        }
    
    def _calculate_confidence(self, ppg_signal: np.ndarray, 
                            demographic_info: Dict[str, Union[int, float]]) -> float:
        """Calculate prediction confidence based on signal quality and completeness."""
        confidence = 0.8  # Base confidence
        
        # Reduce confidence for poor signal quality
        if len(ppg_signal) < 100:
            confidence -= 0.2
        
        # Reduce confidence for missing demographic info
        missing_info = sum(1 for key in ['heart_rate_bpm', 'height_cm', 'weight_kg'] 
                          if key not in demographic_info)
        confidence -= missing_info * 0.1
        
        # Signal quality metrics
        signal_std = np.std(ppg_signal)
        if signal_std < 0.1:  # Very low variability
            confidence -= 0.15
        
        return max(0.3, min(0.95, confidence))
    
    def _interpret_glucose_level(self, glucose_level: float) -> str:
        """Interpret glucose level according to medical standards."""
        if glucose_level < 70:
            return "Hypoglycemia (Low) - Consult healthcare provider"
        elif glucose_level <= 99:
            return "Normal fasting glucose"
        elif glucose_level <= 125:
            return "Prediabetes range - Monitor closely"
        else:
            return "Diabetes range - Consult healthcare provider"
    
    def save_model(self, filepath: str):
        """Save the trained model to file."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'poly_features': self.poly_features,
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
        self.poly_features = model_data.get('poly_features')
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_trained = True

class GlucoseIntegration:
    """
    High-level integration class for glucose prediction in the PaPaGei system.
    """
    
    def __init__(self):
        self.predictor = GlucosePredictorFromPPG(model_type='polynomial_regression')
        self.supported_features = [
            'PPG Signal Analysis',
            'Demographic Integration', 
            'Multiple ML Models',
            'Confidence Scoring',
            'Medical Interpretation'
        ]
    
    def predict_from_papagei_format(self, papagei_data: Dict) -> Dict[str, Union[float, str]]:
        """
        Predict glucose from PaPaGei format data.
        
        Args:
            papagei_data: Dictionary containing PPG data in PaPaGei format
        
        Returns:
            Glucose prediction results
        """
        # Extract PPG signal
        ppg_signal = np.array(papagei_data.get('ppg_signal', []))
        if len(ppg_signal) == 0:
            raise ValueError("No PPG signal data provided")
        
        # Extract demographic information
        demographic_info = {
            'heart_rate_bpm': papagei_data.get('heart_rate', 75),
            'gender': 1 if papagei_data.get('gender', '').lower() == 'male' else 0,
            'height_cm': papagei_data.get('height_cm', 170),
            'weight_kg': papagei_data.get('weight_kg', 70),
            'age_range': self._encode_age_range(papagei_data.get('age', 30))
        }
        
        # Add blood pressure if available from BP prediction
        if 'blood_pressure' in papagei_data:
            bp = papagei_data['blood_pressure']
            demographic_info['systolic_peak_mmhg'] = bp.get('systolic', 120)
            demographic_info['diastolic_peak_mmhg'] = bp.get('diastolic', 80)
        
        return self.predictor.predict_glucose(ppg_signal, demographic_info)
    
    def _encode_age_range(self, age: int) -> int:
        """Encode age into categorical ranges."""
        if age < 25:
            return 0
        elif age < 40:
            return 1
        elif age < 55:
            return 2
        elif age < 70:
            return 3
        else:
            return 4
    
    def get_feature_requirements(self) -> Dict[str, str]:
        """Get information about required features."""
        return {
            'required': {
                'ppg_signal': 'PPG waveform data (mV)',
            },
            'optional': {
                'heart_rate': 'Heart rate in BPM',
                'gender': 'Gender (male/female)',
                'height_cm': 'Height in centimeters',
                'weight_kg': 'Weight in kilograms',
                'age': 'Age in years',
                'blood_pressure': 'Blood pressure readings'
            },
            'output': {
                'predicted_glucose_mg_dl': 'Predicted blood glucose in mg/dL',
                'confidence_score': 'Prediction confidence (0-1)',
                'interpretation': 'Medical interpretation of glucose level',
                'model_used': 'ML model used for prediction'
            }
        }