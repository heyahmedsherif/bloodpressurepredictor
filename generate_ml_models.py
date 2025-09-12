#!/usr/bin/env python3
"""
Generate ML models for health prediction
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_models():
    """Generate and save ML models for health predictions"""
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Generate synthetic training data for demonstration
    # In production, use real medical data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: heart_rate, heart_rate_variability, age, weight, height
    X_train = np.random.randn(n_samples, 5)
    
    # Generate Blood Pressure model
    logger.info("Generating blood pressure model...")
    bp_systolic = 120 + X_train[:, 0] * 10 + X_train[:, 2] * 5 + np.random.randn(n_samples) * 5
    bp_diastolic = 80 + X_train[:, 0] * 7 + X_train[:, 2] * 3 + np.random.randn(n_samples) * 3
    
    bp_model = RandomForestRegressor(n_estimators=100, random_state=42)
    bp_model.fit(X_train, np.column_stack([bp_systolic, bp_diastolic]))
    
    with open('models/bp_model.pkl', 'wb') as f:
        pickle.dump(bp_model, f)
    logger.info("Blood pressure model saved to models/bp_model.pkl")
    
    # Generate Glucose model
    logger.info("Generating glucose model...")
    glucose = 90 + X_train[:, 0] * 5 + X_train[:, 3] * 8 + np.random.randn(n_samples) * 10
    
    glucose_model = RandomForestRegressor(n_estimators=100, random_state=42)
    glucose_model.fit(X_train, glucose)
    
    with open('models/glucose_model.pkl', 'wb') as f:
        pickle.dump(glucose_model, f)
    logger.info("Glucose model saved to models/glucose_model.pkl")
    
    # Generate Cholesterol model
    logger.info("Generating cholesterol model...")
    cholesterol = 180 + X_train[:, 1] * 10 + X_train[:, 3] * 15 + np.random.randn(n_samples) * 20
    
    cholesterol_model = RandomForestRegressor(n_estimators=100, random_state=42)
    cholesterol_model.fit(X_train, cholesterol)
    
    with open('models/cholesterol_model.pkl', 'wb') as f:
        pickle.dump(cholesterol_model, f)
    logger.info("Cholesterol model saved to models/cholesterol_model.pkl")
    
    # Generate Oxygen Saturation model
    logger.info("Generating oxygen saturation model...")
    oxygen = 97 + X_train[:, 1] * 0.5 - X_train[:, 2] * 0.3 + np.random.randn(n_samples) * 1
    oxygen = np.clip(oxygen, 90, 100)  # Keep in realistic range
    
    oxygen_model = RandomForestRegressor(n_estimators=100, random_state=42)
    oxygen_model.fit(X_train, oxygen)
    
    with open('models/oxygen_model.pkl', 'wb') as f:
        pickle.dump(oxygen_model, f)
    logger.info("Oxygen saturation model saved to models/oxygen_model.pkl")
    
    # Generate Stress Level model
    logger.info("Generating stress level model...")
    stress = 50 + X_train[:, 1] * 20 + X_train[:, 0] * 10 + np.random.randn(n_samples) * 10
    stress = np.clip(stress, 0, 100)  # Keep in 0-100 range
    
    stress_model = RandomForestRegressor(n_estimators=100, random_state=42)
    stress_model.fit(X_train, stress)
    
    with open('models/stress_model.pkl', 'wb') as f:
        pickle.dump(stress_model, f)
    logger.info("Stress level model saved to models/stress_model.pkl")
    
    # Save scaler for feature normalization
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("Feature scaler saved to models/scaler.pkl")
    
    logger.info("All models generated successfully!")
    return True

if __name__ == "__main__":
    # Check if models already exist
    models_exist = all([
        os.path.exists('models/bp_model.pkl'),
        os.path.exists('models/glucose_model.pkl'),
        os.path.exists('models/cholesterol_model.pkl'),
        os.path.exists('models/oxygen_model.pkl'),
        os.path.exists('models/stress_model.pkl'),
        os.path.exists('models/scaler.pkl')
    ])
    
    if models_exist:
        logger.info("All models already exist. Skipping generation.")
    else:
        generate_models()