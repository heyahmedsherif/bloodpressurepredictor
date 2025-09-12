"""
Health Predictor - Health metrics prediction from demographics and PPG data
"""
import numpy as np


class HealthPredictor:
    """Health metrics prediction using demographics and PPG data."""
    
    @staticmethod
    def predict_blood_pressure(age, gender, height, weight, heart_rate):
        """Predict blood pressure from demographics and heart rate."""
        # Simple predictive model (replace with actual trained models)
        bmi = weight / ((height/100) ** 2)
        
        # Gender factor
        gender_factor = 1.0 if gender.lower() == 'male' else 0.9
        
        # Base values
        systolic = 110 + (age - 30) * 0.6 + (heart_rate - 70) * 0.25 + (bmi - 25) * 1.2
        diastolic = 70 + (age - 30) * 0.4 + (heart_rate - 70) * 0.15 + (bmi - 25) * 0.8
        
        systolic *= gender_factor
        diastolic *= gender_factor
        
        # Clamp to reasonable ranges
        systolic = max(90, min(200, systolic))
        diastolic = max(60, min(120, diastolic))
        
        return round(systolic, 1), round(diastolic, 1)
    
    @staticmethod
    def predict_glucose(age, gender, height, weight, heart_rate):
        """Predict glucose level from demographics."""
        bmi = weight / ((height/100) ** 2)
        
        # Base glucose level
        glucose = 85 + (bmi - 25) * 2.5 + (age - 40) * 0.8
        
        # Gender adjustment
        if gender.lower() == 'female':
            glucose *= 0.95
        
        # Stress factor from elevated heart rate
        if heart_rate > 80:
            glucose += (heart_rate - 80) * 0.3
        
        return max(70, min(150, round(glucose, 1)))
    
    @staticmethod
    def predict_cholesterol(age, gender, height, weight, heart_rate):
        """Predict cholesterol level from demographics."""
        bmi = weight / ((height/100) ** 2)
        
        # Base cholesterol
        cholesterol = 160 + (age - 30) * 1.5 + (bmi - 25) * 4
        
        # Gender adjustment
        if gender.lower() == 'male':
            cholesterol += 15
        
        return max(120, min(300, round(cholesterol, 1)))
    
    @staticmethod
    def predict_cv_risk(age, gender, height, weight, heart_rate, systolic_bp):
        """Predict cardiovascular risk score."""
        bmi = weight / ((height/100) ** 2)
        
        # Risk factors
        age_risk = max(0, (age - 40) * 1.5)
        bmi_risk = max(0, (bmi - 25) * 2)
        bp_risk = max(0, (systolic_bp - 130) * 0.8)
        hr_risk = max(0, abs(heart_rate - 70) * 0.2)
        
        # Gender factor
        gender_risk = 5 if gender.lower() == 'male' else 0
        
        total_risk = age_risk + bmi_risk + bp_risk + hr_risk + gender_risk
        
        return max(0, min(100, round(total_risk, 1)))