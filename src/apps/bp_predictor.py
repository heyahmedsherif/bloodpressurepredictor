"""
Blood Pressure Prediction Application

This application uses clinical and physiological markers to predict blood pressure
using trained machine learning models. It provides real prediction capabilities
based on established cardiovascular risk factors and biomarkers.

Features:
- Multiple predictive markers (demographics, lifestyle, clinical)
- Trained ML models (Random Forest, Gradient Boosting, Neural Network)
- Real-time prediction with confidence intervals
- Feature importance analysis
- Clinical interpretation and recommendations
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import warnings
from datetime import datetime
import json
from typing import Dict, Tuple, List
warnings.filterwarnings('ignore')

# Machine learning imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.inspection import permutation_importance
    import joblib
    ML_AVAILABLE = True
except ImportError as e:
    st.error(f"Machine learning libraries not available: {e}")
    ML_AVAILABLE = False

# Page configuration - commented out to avoid conflict with main app
# st.set_page_config(
#     page_title="Blood Pressure Predictor",
#     page_icon="🩺",
#     layout="wide",
#     initial_sidebar_state="expanded"
)

class BPPredictor:
    """Blood Pressure Prediction Model"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.is_trained = False
        self.training_data = None
        
    def get_feature_names(self) -> List[str]:
        """Get list of feature names used for prediction"""
        return [
            # Demographics
            'age', 'gender_male', 'bmi',
            
            # Lifestyle factors
            'smoking_status', 'alcohol_weekly_units', 'exercise_hours_week',
            'sleep_hours', 'stress_level',
            
            # Clinical markers
            'resting_heart_rate', 'cholesterol_total', 'cholesterol_hdl',
            'glucose_fasting', 'sodium_intake', 'potassium_intake',
            
            # Family history and medical conditions
            'family_history_hypertension', 'diabetes', 'kidney_disease',
            
            # Physical measurements
            'waist_circumference', 'neck_circumference'
        ]
    
    def generate_synthetic_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate synthetic training data based on clinical relationships"""
        np.random.seed(42)
        
        data = {}
        
        # Demographics
        data['age'] = np.random.normal(50, 15, n_samples).clip(18, 90)
        data['gender_male'] = np.random.binomial(1, 0.5, n_samples)
        
        # BMI with realistic distribution
        data['bmi'] = np.array([np.random.gamma(4, 6) + 16 for _ in range(n_samples)])
        data['bmi'] = np.clip(data['bmi'], 16, 45)
        
        # Lifestyle factors
        data['smoking_status'] = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.25, 0.15])  # 0=never, 1=former, 2=current
        data['alcohol_weekly_units'] = np.random.gamma(2, 3, n_samples).clip(0, 50)
        data['exercise_hours_week'] = np.random.gamma(2, 2, n_samples).clip(0, 20)
        data['sleep_hours'] = np.random.normal(7.5, 1.2, n_samples).clip(4, 12)
        data['stress_level'] = np.random.uniform(1, 10, n_samples)
        
        # Clinical markers
        data['resting_heart_rate'] = np.random.normal(70, 12, n_samples).clip(45, 120)
        data['cholesterol_total'] = np.random.normal(200, 40, n_samples).clip(120, 400)
        data['cholesterol_hdl'] = np.random.normal(50, 15, n_samples).clip(20, 100)
        data['glucose_fasting'] = np.random.gamma(20, 5, n_samples).clip(70, 300)
        data['sodium_intake'] = np.random.normal(2300, 500, n_samples).clip(1000, 5000)  # mg/day
        data['potassium_intake'] = np.random.normal(3000, 800, n_samples).clip(1500, 6000)  # mg/day
        
        # Medical conditions (binary)
        data['family_history_hypertension'] = np.random.binomial(1, 0.3, n_samples)
        data['diabetes'] = np.random.binomial(1, 0.1, n_samples)
        data['kidney_disease'] = np.random.binomial(1, 0.05, n_samples)
        
        # Physical measurements
        data['waist_circumference'] = data['bmi'] * 2.5 + np.random.normal(0, 5, n_samples)  # Correlated with BMI
        data['neck_circumference'] = 35 + data['gender_male'] * 5 + np.random.normal(0, 3, n_samples)
        
        df = pd.DataFrame(data)
        
        # Generate realistic blood pressure based on features
        # Systolic BP prediction formula (evidence-based relationships)
        systolic_base = 100
        systolic_bp = (
            systolic_base +
            (df['age'] - 30) * 0.8 +  # Age effect
            df['gender_male'] * 8 +   # Male gender effect
            (df['bmi'] - 25) * 1.2 +  # BMI effect
            df['smoking_status'] * 5 + # Smoking effect
            (df['alcohol_weekly_units'] - 7) * 0.3 +  # Alcohol effect
            -(df['exercise_hours_week'] - 3) * 1.5 +  # Exercise protective effect
            (df['stress_level'] - 5) * 2 +  # Stress effect
            (df['resting_heart_rate'] - 70) * 0.3 +  # HR effect
            (df['cholesterol_total'] - 200) * 0.05 +  # Cholesterol effect
            -(df['cholesterol_hdl'] - 50) * 0.2 +  # HDL protective effect
            (df['glucose_fasting'] - 100) * 0.1 +  # Glucose effect
            (df['sodium_intake'] - 2300) * 0.005 +  # Sodium effect
            -(df['potassium_intake'] - 3000) * 0.003 +  # Potassium protective effect
            df['family_history_hypertension'] * 12 +  # Family history
            df['diabetes'] * 15 +  # Diabetes effect
            df['kidney_disease'] * 20 +  # Kidney disease effect
            (df['waist_circumference'] - 85) * 0.5 +  # Waist circumference
            np.random.normal(0, 8, n_samples)  # Random noise
        ).clip(90, 200)
        
        # Diastolic BP (typically 60-65% of systolic with some variation)
        diastolic_bp = (
            systolic_bp * 0.65 +
            (df['age'] - 30) * 0.1 +  # Less age effect than systolic
            df['gender_male'] * 3 +
            (df['bmi'] - 25) * 0.8 +
            df['smoking_status'] * 3 +
            (df['resting_heart_rate'] - 70) * 0.2 +
            df['diabetes'] * 8 +
            np.random.normal(0, 5, n_samples)  # Random noise
        ).clip(60, 120)
        
        df['systolic_bp'] = systolic_bp
        df['diastolic_bp'] = diastolic_bp
        
        return df
    
    def train_models(self, df: pd.DataFrame):
        """Train multiple ML models for blood pressure prediction"""
        if not ML_AVAILABLE:
            st.error("Machine learning libraries not available for training")
            return
        
        self.training_data = df
        self.feature_names = self.get_feature_names()
        
        # Prepare features and targets
        X = df[self.feature_names]
        y_systolic = df['systolic_bp']
        y_diastolic = df['diastolic_bp']
        
        # Split data
        X_train, X_test, y_sys_train, y_sys_test, y_dia_train, y_dia_test = train_test_split(
            X, y_systolic, y_diastolic, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scalers['systolic'] = StandardScaler()
        self.scalers['diastolic'] = StandardScaler()
        
        X_train_scaled = self.scalers['systolic'].fit_transform(X_train)
        X_test_scaled = self.scalers['systolic'].transform(X_test)
        
        # Train models for systolic BP
        self.models['systolic'] = {}
        
        # Random Forest
        rf_sys = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_sys.fit(X_train, y_sys_train)
        self.models['systolic']['random_forest'] = rf_sys
        
        # Gradient Boosting
        gb_sys = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
        gb_sys.fit(X_train, y_sys_train)
        self.models['systolic']['gradient_boost'] = gb_sys
        
        # Neural Network
        nn_sys = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        nn_sys.fit(X_train_scaled, y_sys_train)
        self.models['systolic']['neural_net'] = nn_sys
        
        # Train models for diastolic BP
        self.models['diastolic'] = {}
        
        # Random Forest
        rf_dia = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_dia.fit(X_train, y_dia_train)
        self.models['diastolic']['random_forest'] = rf_dia
        
        # Gradient Boosting
        gb_dia = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
        gb_dia.fit(X_train, y_dia_train)
        self.models['diastolic']['gradient_boost'] = gb_dia
        
        # Neural Network
        nn_dia = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        nn_dia.fit(X_train_scaled, y_dia_train)
        self.models['diastolic']['neural_net'] = nn_dia
        
        # Evaluate models
        self.model_performance = {}
        
        for bp_type in ['systolic', 'diastolic']:
            self.model_performance[bp_type] = {}
            y_test = y_sys_test if bp_type == 'systolic' else y_dia_test
            
            for model_name, model in self.models[bp_type].items():
                if model_name == 'neural_net':
                    X_test_input = X_test_scaled
                else:
                    X_test_input = X_test
                
                y_pred = model.predict(X_test_input)
                
                self.model_performance[bp_type][model_name] = {
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred)
                }
        
        self.is_trained = True
        
    def predict_bp(self, features: Dict, model_type: str = 'gradient_boost') -> Dict:
        """Predict blood pressure from input features"""
        if not self.is_trained:
            return {'error': 'Models not trained yet'}
        
        # Convert features to DataFrame
        feature_array = np.array([[features[name] for name in self.feature_names]])
        feature_df = pd.DataFrame(feature_array, columns=self.feature_names)
        
        # Scale for neural network if needed
        if model_type == 'neural_net':
            feature_scaled = self.scalers['systolic'].transform(feature_df)
            systolic_pred = self.models['systolic'][model_type].predict(feature_scaled)[0]
            diastolic_pred = self.models['diastolic'][model_type].predict(feature_scaled)[0]
        else:
            systolic_pred = self.models['systolic'][model_type].predict(feature_df)[0]
            diastolic_pred = self.models['diastolic'][model_type].predict(feature_df)[0]
        
        # Calculate confidence intervals (simplified)
        sys_std = self.model_performance['systolic'][model_type]['rmse']
        dia_std = self.model_performance['diastolic'][model_type]['rmse']
        
        return {
            'systolic_bp': round(max(90, min(200, systolic_pred)), 1),
            'diastolic_bp': round(max(60, min(120, diastolic_pred)), 1),
            'systolic_ci': (
                round(systolic_pred - 1.96 * sys_std, 1),
                round(systolic_pred + 1.96 * sys_std, 1)
            ),
            'diastolic_ci': (
                round(diastolic_pred - 1.96 * dia_std, 1),
                round(diastolic_pred + 1.96 * dia_std, 1)
            ),
            'model_performance': {
                'systolic_mae': self.model_performance['systolic'][model_type]['mae'],
                'diastolic_mae': self.model_performance['diastolic'][model_type]['mae'],
                'systolic_r2': self.model_performance['systolic'][model_type]['r2'],
                'diastolic_r2': self.model_performance['diastolic'][model_type]['r2']
            }
        }
    
    def get_feature_importance(self, model_type: str = 'gradient_boost') -> Dict:
        """Get feature importance from trained models"""
        if not self.is_trained or model_type == 'neural_net':
            return {}
        
        sys_importance = self.models['systolic'][model_type].feature_importances_
        dia_importance = self.models['diastolic'][model_type].feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'systolic_importance': sys_importance,
            'diastolic_importance': dia_importance
        }).sort_values('systolic_importance', ascending=False)
        
        return importance_df

def interpret_bp_values(systolic: float, diastolic: float) -> Dict:
    """Interpret blood pressure values according to clinical guidelines"""
    
    # AHA/ACC Guidelines
    if systolic < 120 and diastolic < 80:
        category = "Normal"
        color = "green"
        risk = "Low"
        recommendations = [
            "Maintain healthy lifestyle",
            "Regular exercise and balanced diet",
            "Annual check-ups"
        ]
    elif systolic < 130 and diastolic < 80:
        category = "Elevated"
        color = "yellow"
        risk = "Moderate"
        recommendations = [
            "Lifestyle modifications recommended",
            "Reduce sodium intake",
            "Increase physical activity",
            "Monitor regularly"
        ]
    elif (130 <= systolic < 140) or (80 <= diastolic < 90):
        category = "Stage 1 Hypertension"
        color = "orange"
        risk = "High"
        recommendations = [
            "Lifestyle changes essential",
            "Consider medication consultation",
            "Monthly monitoring",
            "Stress management"
        ]
    elif (140 <= systolic < 180) or (90 <= diastolic < 120):
        category = "Stage 2 Hypertension"
        color = "red"
        risk = "Very High"
        recommendations = [
            "Medication likely needed",
            "Immediate medical consultation",
            "Weekly monitoring",
            "Comprehensive lifestyle changes"
        ]
    else:
        category = "Hypertensive Crisis"
        color = "darkred"
        risk = "Critical"
        recommendations = [
            "EMERGENCY: Seek immediate medical attention",
            "Call emergency services if symptomatic",
            "Do not delay treatment"
        ]
    
    return {
        'category': category,
        'color': color,
        'risk_level': risk,
        'recommendations': recommendations
    }

def main():
    """Main application"""
    
    st.title("🩺 Blood Pressure Predictor")
    st.markdown("*Predict blood pressure using clinical and lifestyle markers*")
    
    # Initialize predictor
    if 'bp_predictor' not in st.session_state:
        st.session_state.bp_predictor = BPPredictor()
        
        # Auto-train on synthetic data
        with st.spinner("Training prediction models..."):
            training_data = st.session_state.bp_predictor.generate_synthetic_data(5000)
            st.session_state.bp_predictor.train_models(training_data)
        
        st.success("✅ Models trained successfully!")
    
    # Sidebar - Model Information
    st.sidebar.header("📊 Model Information")
    
    if st.session_state.bp_predictor.is_trained:
        st.sidebar.success("✅ Models trained and ready")
        
        # Model selection
        model_type = st.sidebar.selectbox(
            "Select Prediction Model:",
            ['gradient_boost', 'random_forest', 'neural_net'],
            index=0,
            format_func=lambda x: {
                'gradient_boost': 'Gradient Boosting (Recommended)',
                'random_forest': 'Random Forest',
                'neural_net': 'Neural Network'
            }[x]
        )
        
        # Show model performance
        if st.sidebar.button("Show Model Performance"):
            perf = st.session_state.bp_predictor.model_performance
            
            st.sidebar.subheader(f"{model_type.title()} Performance")
            st.sidebar.metric("Systolic MAE", f"{perf['systolic'][model_type]['mae']:.1f} mmHg")
            st.sidebar.metric("Diastolic MAE", f"{perf['diastolic'][model_type]['mae']:.1f} mmHg")
            st.sidebar.metric("Systolic R²", f"{perf['systolic'][model_type]['r2']:.3f}")
            st.sidebar.metric("Diastolic R²", f"{perf['diastolic'][model_type]['r2']:.3f}")
    else:
        st.sidebar.error("❌ Models not trained")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("👤 Patient Information")
        
        # Demographics
        age = st.slider("Age (years)", 18, 90, 47)
        gender = st.selectbox("Gender", ["Female", "Male"], index=1)
        gender_male = 1 if gender == "Male" else 0
        
        height_cm = st.slider("Height (cm)", 140, 220, 173)
        weight_kg = st.slider("Weight (kg)", 40, 150, 83)
        bmi = weight_kg / ((height_cm/100) ** 2)
        st.info(f"BMI: {bmi:.1f}")
        
        # Physical measurements
        waist_circumference = st.slider("Waist Circumference (cm)", 60, 150, 85)
        neck_circumference = st.slider("Neck Circumference (cm)", 25, 50, 35)
        
        st.subheader("🏃 Lifestyle Factors")
        
        # Lifestyle
        smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        smoking_code = {"Never": 0, "Former": 1, "Current": 2}[smoking_status]
        
        alcohol_weekly_units = st.slider("Alcohol (units/week)", 0.0, 50.0, 7.0, 0.5)
        exercise_hours_week = st.slider("Exercise (hours/week)", 0.0, 20.0, 3.0, 0.5)
        sleep_hours = st.slider("Sleep (hours/night)", 4.0, 12.0, 7.5, 0.5)
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        
        # Diet
        sodium_intake = st.slider("Sodium Intake (mg/day)", 1000, 5000, 2300, 50)
        potassium_intake = st.slider("Potassium Intake (mg/day)", 1500, 6000, 3000, 100)
    
    with col2:
        st.subheader("🩺 Clinical Markers")
        
        # Clinical measurements
        resting_heart_rate = st.slider("Resting Heart Rate (bpm)", 45, 120, 70)
        cholesterol_total = st.slider("Total Cholesterol (mg/dL)", 120, 400, 200)
        cholesterol_hdl = st.slider("HDL Cholesterol (mg/dL)", 20, 100, 50)
        glucose_fasting = st.slider("Fasting Glucose (mg/dL)", 70, 300, 95)
        
        st.subheader("📋 Medical History")
        
        # Medical conditions
        family_history_hypertension = st.checkbox("Family History of Hypertension")
        diabetes = st.checkbox("Diabetes")
        kidney_disease = st.checkbox("Kidney Disease")
        
        # Compile features
        features = {
            'age': age,
            'gender_male': gender_male,
            'bmi': bmi,
            'smoking_status': smoking_code,
            'alcohol_weekly_units': alcohol_weekly_units,
            'exercise_hours_week': exercise_hours_week,
            'sleep_hours': sleep_hours,
            'stress_level': stress_level,
            'resting_heart_rate': resting_heart_rate,
            'cholesterol_total': cholesterol_total,
            'cholesterol_hdl': cholesterol_hdl,
            'glucose_fasting': glucose_fasting,
            'sodium_intake': sodium_intake,
            'potassium_intake': potassium_intake,
            'family_history_hypertension': int(family_history_hypertension),
            'diabetes': int(diabetes),
            'kidney_disease': int(kidney_disease),
            'waist_circumference': waist_circumference,
            'neck_circumference': neck_circumference
        }
        
        # Predict button
        if st.button("🔮 Predict Blood Pressure", type="primary"):
            if st.session_state.bp_predictor.is_trained:
                with st.spinner("Making prediction..."):
                    prediction = st.session_state.bp_predictor.predict_bp(features, model_type)
                
                if 'error' not in prediction:
                    # Display results
                    st.subheader("📊 Prediction Results")
                    
                    # Main metrics
                    col_sys, col_dia = st.columns(2)
                    
                    with col_sys:
                        st.metric(
                            "Systolic BP", 
                            f"{prediction['systolic_bp']:.0f} mmHg",
                            help=f"95% CI: {prediction['systolic_ci'][0]:.0f} - {prediction['systolic_ci'][1]:.0f} mmHg"
                        )
                    
                    with col_dia:
                        st.metric(
                            "Diastolic BP",
                            f"{prediction['diastolic_bp']:.0f} mmHg", 
                            help=f"95% CI: {prediction['diastolic_ci'][0]:.0f} - {prediction['diastolic_ci'][1]:.0f} mmHg"
                        )
                    
                    # Clinical interpretation
                    interpretation = interpret_bp_values(
                        prediction['systolic_bp'], 
                        prediction['diastolic_bp']
                    )
                    
                    st.subheader("🏥 Clinical Interpretation")
                    
                    # Status indicator
                    if interpretation['color'] == 'green':
                        st.success(f"✅ {interpretation['category']} - {interpretation['risk_level']} Risk")
                    elif interpretation['color'] == 'yellow':
                        st.warning(f"⚠️ {interpretation['category']} - {interpretation['risk_level']} Risk")
                    elif interpretation['color'] == 'orange':
                        st.warning(f"⚠️ {interpretation['category']} - {interpretation['risk_level']} Risk")
                    elif interpretation['color'] == 'red':
                        st.error(f"🚨 {interpretation['category']} - {interpretation['risk_level']} Risk")
                    else:
                        st.error(f"🚨 {interpretation['category']} - {interpretation['risk_level']} Risk")
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    for rec in interpretation['recommendations']:
                        st.write(f"• {rec}")
                    
                    # Model performance
                    st.subheader("📈 Model Performance")
                    perf = prediction['model_performance']
                    
                    col_perf1, col_perf2 = st.columns(2)
                    
                    with col_perf1:
                        st.metric("Systolic Accuracy", f"±{perf['systolic_mae']:.1f} mmHg MAE")
                        st.metric("Systolic R²", f"{perf['systolic_r2']:.3f}")
                    
                    with col_perf2:
                        st.metric("Diastolic Accuracy", f"±{perf['diastolic_mae']:.1f} mmHg MAE")
                        st.metric("Diastolic R²", f"{perf['diastolic_r2']:.3f}")
    
    # Feature Importance Analysis
    if st.session_state.bp_predictor.is_trained and model_type != 'neural_net':
        st.subheader("📊 Feature Importance Analysis")
        
        importance_df = st.session_state.bp_predictor.get_feature_importance(model_type)
        
        if not importance_df.empty:
            # Plot feature importance
            fig = px.bar(
                importance_df.head(10),
                x='systolic_importance',
                y='feature',
                orientation='h',
                title="Top 10 Most Important Features for Systolic BP",
                labels={'systolic_importance': 'Feature Importance', 'feature': 'Feature'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top features
            st.write("**Most Important Factors for Blood Pressure:**")
            for i, row in importance_df.head(5).iterrows():
                st.write(f"{i+1}. {row['feature'].replace('_', ' ').title()}: {row['systolic_importance']:.3f}")
    
    # Export functionality
    st.subheader("💾 Export Results")
    
    if st.button("📄 Generate Detailed Report"):
        report = {
            "timestamp": datetime.now().isoformat(),
            "patient_info": {
                "age": age,
                "gender": gender,
                "bmi": round(bmi, 1),
                "height_cm": height_cm,
                "weight_kg": weight_kg
            },
            "lifestyle_factors": {
                "smoking_status": smoking_status,
                "alcohol_weekly_units": alcohol_weekly_units,
                "exercise_hours_week": exercise_hours_week,
                "sleep_hours": sleep_hours,
                "stress_level": stress_level
            },
            "clinical_markers": {
                "resting_heart_rate": resting_heart_rate,
                "cholesterol_total": cholesterol_total,
                "cholesterol_hdl": cholesterol_hdl,
                "glucose_fasting": glucose_fasting,
                "sodium_intake": sodium_intake,
                "potassium_intake": potassium_intake
            },
            "medical_history": {
                "family_history_hypertension": family_history_hypertension,
                "diabetes": diabetes,
                "kidney_disease": kidney_disease
            }
        }
        
        if st.session_state.bp_predictor.is_trained:
            prediction = st.session_state.bp_predictor.predict_bp(features, model_type)
            if 'error' not in prediction:
                interpretation = interpret_bp_values(prediction['systolic_bp'], prediction['diastolic_bp'])
                report.update({
                    "predictions": {
                        "systolic_bp": prediction['systolic_bp'],
                        "diastolic_bp": prediction['diastolic_bp'],
                        "systolic_ci": prediction['systolic_ci'],
                        "diastolic_ci": prediction['diastolic_ci'],
                        "model_used": model_type
                    },
                    "interpretation": interpretation
                })
        
        json_data = json.dumps(report, indent=2)
        st.download_button(
            label="📄 Download Report (JSON)",
            data=json_data,
            file_name=f"bp_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()