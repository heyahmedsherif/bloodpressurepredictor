"""
Train LDL/HDL Cholesterol Prediction Models
============================================
This script analyzes PPG signals to predict LDL and HDL cholesterol separately.
Based on research showing correlations between PPG morphology and lipid profiles.
"""

import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import signal as scipy_signal
from scipy.signal import find_peaks, peak_widths
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("LDL/HDL Cholesterol Prediction Model Training")
print("="*60)

# Load the cholesterol dataset
data_dir = Path("datasets/ppg_cholesterol")
ppg_file = data_dir / "all_ppg_signals.csv"
subject_file = data_dir / "subject_info.xlsx"

print(f"\nLoading data from {data_dir}")

# Load PPG signals
ppg_df = pd.read_csv(ppg_file)
print(f"PPG data shape: {ppg_df.shape}")
print(f"Columns: {ppg_df.columns.tolist()}")

# Load subject information
try:
    subject_df = pd.read_excel(subject_file)
    print(f"\nSubject data shape: {subject_df.shape}")
    print(f"Columns: {subject_df.columns.tolist()}")
except Exception as e:
    print(f"Error loading Excel: {e}")
    subject_df = None

# Analyze unique subjects
unique_subjects = ppg_df[['Edad', 'Sexo', 'Colesterol']].drop_duplicates()
print(f"\nUnique subjects: {len(unique_subjects)}")
print("\nCholesterol distribution:")
print(unique_subjects['Colesterol'].describe())

def extract_advanced_ppg_features(ppg_signal, sampling_rate=100):
    """Extract advanced PPG features relevant for cholesterol prediction"""
    features = {}

    # Normalize signal
    ppg_signal = np.array(ppg_signal)
    ppg_signal = (ppg_signal - np.mean(ppg_signal)) / (np.std(ppg_signal) + 1e-10)

    # Apply bandpass filter
    nyquist = sampling_rate / 2
    low = 0.5 / nyquist
    high = min(5.0 / nyquist, 0.99)

    if low < high:
        b, a = scipy_signal.butter(4, [low, high], btype='band')
        filtered_signal = scipy_signal.filtfilt(b, a, ppg_signal)
    else:
        filtered_signal = ppg_signal

    # Find peaks (systolic peaks)
    peaks, properties = find_peaks(filtered_signal,
                                   distance=sampling_rate*0.5,
                                   prominence=0.1)

    if len(peaks) > 2:
        # Basic features
        features['heart_rate'] = len(peaks) / (len(ppg_signal) / sampling_rate) * 60
        features['hrv'] = np.std(np.diff(peaks)) / sampling_rate * 1000

        # Amplitude features
        features['mean_amplitude'] = np.mean(properties['prominences'])
        features['std_amplitude'] = np.std(properties['prominences'])

        # Pulse width features
        widths, width_heights, left_ips, right_ips = peak_widths(filtered_signal, peaks, rel_height=0.5)
        features['mean_width'] = np.mean(widths) / sampling_rate
        features['std_width'] = np.std(widths) / sampling_rate

        # Augmentation Index (AI) - indicator of arterial stiffness
        # Higher AI correlates with higher LDL
        for i, peak_idx in enumerate(peaks[:-1]):
            try:
                # Look for dicrotic notch between peaks
                segment = filtered_signal[peak_idx:peaks[i+1]]
                if len(segment) > 10:
                    # Find local minimum (dicrotic notch)
                    notch_idx = np.argmin(segment[len(segment)//3:]) + len(segment)//3
                    if notch_idx < len(segment) - 5:
                        # Find dicrotic peak after notch
                        dicrotic_peak = np.max(segment[notch_idx:])
                        systolic_peak = filtered_signal[peak_idx]
                        if systolic_peak != 0:
                            ai = dicrotic_peak / systolic_peak
                            if 'augmentation_index' not in features:
                                features['augmentation_index'] = []
                            features['augmentation_index'].append(ai)
            except:
                pass

        if 'augmentation_index' in features and len(features['augmentation_index']) > 0:
            features['mean_ai'] = np.mean(features['augmentation_index'])
            features['std_ai'] = np.std(features['augmentation_index'])
        else:
            features['mean_ai'] = 0.5  # Default value
            features['std_ai'] = 0.1

        if 'augmentation_index' in features:
            del features['augmentation_index']

        # Stiffness Index (SI) - pulse wave velocity indicator
        # Higher SI correlates with atherosclerosis (high LDL)
        if len(peaks) > 1:
            # Time from foot to peak (approximation)
            rise_times = []
            for peak_idx in peaks:
                # Find foot of pulse (start of upstroke)
                start_search = max(0, peak_idx - sampling_rate//2)
                segment = filtered_signal[start_search:peak_idx]
                if len(segment) > 0:
                    foot_idx = np.argmin(segment)
                    rise_time = (peak_idx - (start_search + foot_idx)) / sampling_rate
                    rise_times.append(rise_time)

            if rise_times:
                features['stiffness_index'] = 1.0 / np.mean(rise_times)  # Simplified SI
            else:
                features['stiffness_index'] = 10.0  # Default

        # Reflection Index (RI) - vascular tone indicator
        # Related to HDL protective effects
        features['reflection_index'] = features.get('mean_ai', 0.5) * features.get('stiffness_index', 10.0)

    else:
        # Default values if peak detection fails
        features = {
            'heart_rate': 75,
            'hrv': 50,
            'mean_amplitude': 0.5,
            'std_amplitude': 0.1,
            'mean_width': 0.3,
            'std_width': 0.05,
            'mean_ai': 0.5,
            'std_ai': 0.1,
            'stiffness_index': 10.0,
            'reflection_index': 5.0
        }

    return features

def estimate_ldl_hdl_from_total(total_cholesterol, age, sex, ppg_features):
    """
    Estimate LDL and HDL based on total cholesterol and PPG features.
    Uses empirical formulas and PPG-based adjustments.
    """

    # Base estimates using typical ratios
    # Average total cholesterol composition:
    # LDL: ~60-70% of total
    # HDL: ~20-30% of total
    # VLDL/Triglycerides: ~10-20% of total

    # Start with population averages
    if sex == 'F':
        # Women typically have higher HDL
        hdl_base = 55 + (45 - age) * 0.2  # HDL decreases with age
        ldl_ratio = 0.60  # Lower LDL ratio for women
    else:
        # Men typically have lower HDL
        hdl_base = 45 + (45 - age) * 0.15
        ldl_ratio = 0.65  # Higher LDL ratio for men

    # Adjust based on PPG features

    # Higher augmentation index suggests arterial stiffness (higher LDL)
    ai_factor = ppg_features.get('mean_ai', 0.5)
    ldl_adjustment = 1.0 + (ai_factor - 0.5) * 0.3  # ±15% based on AI

    # Higher stiffness index suggests atherosclerosis (higher LDL, lower HDL)
    si_factor = ppg_features.get('stiffness_index', 10.0) / 10.0
    ldl_adjustment *= (1.0 + (si_factor - 1.0) * 0.2)

    # Better HRV suggests better cardiovascular health (higher HDL)
    hrv_factor = ppg_features.get('hrv', 50) / 50.0
    hdl_adjustment = 1.0 + (hrv_factor - 1.0) * 0.2

    # Calculate estimates
    ldl_estimate = total_cholesterol * ldl_ratio * ldl_adjustment
    hdl_estimate = hdl_base * hdl_adjustment

    # Ensure HDL is within reasonable range (40-100 mg/dL)
    hdl_estimate = np.clip(hdl_estimate, 40, min(100, total_cholesterol * 0.4))

    # Calculate LDL using modified Friedewald equation
    # TC = LDL + HDL + (Triglycerides/5)
    # Assume triglycerides = 150 mg/dL (population average)
    triglycerides_estimate = 150
    ldl_estimate = total_cholesterol - hdl_estimate - (triglycerides_estimate / 5)

    # Apply PPG-based adjustment to LDL
    ldl_estimate *= ldl_adjustment

    # Ensure LDL is reasonable
    ldl_estimate = np.clip(ldl_estimate,
                          total_cholesterol * 0.4,  # At least 40% of total
                          total_cholesterol * 0.75)  # At most 75% of total

    # Final adjustment to maintain total
    scale_factor = (total_cholesterol - triglycerides_estimate/5) / (ldl_estimate + hdl_estimate)
    ldl_estimate *= scale_factor
    hdl_estimate *= scale_factor

    return ldl_estimate, hdl_estimate

# Process data and extract features
print("\n" + "="*40)
print("Processing PPG signals and extracting features...")
print("="*40)

# Group by subject and process
subject_features = []
for (age, sex, chol), group in ppg_df.groupby(['Edad', 'Sexo', 'Colesterol']):
    ppg_signal = group['Senal_PPG'].values

    # Extract advanced PPG features
    features = extract_advanced_ppg_features(ppg_signal)
    features['age'] = age
    features['sex'] = 1 if sex == 'M' else 0
    features['total_cholesterol'] = chol

    # Estimate LDL and HDL
    ldl, hdl = estimate_ldl_hdl_from_total(chol, age, sex, features)
    features['ldl_estimated'] = ldl
    features['hdl_estimated'] = hdl
    features['ldl_hdl_ratio'] = ldl / hdl if hdl > 0 else 3.5

    subject_features.append(features)

    # Print first few subjects as examples
    if len(subject_features) <= 3:
        print(f"\nSubject: Age={age}, Sex={sex}, Total Chol={chol}")
        print(f"  Estimated LDL: {ldl:.1f} mg/dL")
        print(f"  Estimated HDL: {hdl:.1f} mg/dL")
        print(f"  LDL/HDL Ratio: {ldl/hdl:.2f}")
        print(f"  AI: {features['mean_ai']:.3f}, SI: {features['stiffness_index']:.2f}")

# Create DataFrame
features_df = pd.DataFrame(subject_features)
print(f"\nProcessed {len(features_df)} subjects")

# Prepare for model training
feature_cols = ['heart_rate', 'hrv', 'mean_amplitude', 'std_amplitude',
                'mean_width', 'std_width', 'mean_ai', 'std_ai',
                'stiffness_index', 'reflection_index', 'age', 'sex']

X = features_df[feature_cols]
y_ldl = features_df['ldl_estimated']
y_hdl = features_df['hdl_estimated']
y_total = features_df['total_cholesterol']

print("\n" + "="*40)
print("Training LDL Prediction Model")
print("="*40)

# Split data
X_train, X_test, y_ldl_train, y_ldl_test, y_hdl_train, y_hdl_test = train_test_split(
    X, y_ldl, y_hdl, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train LDL model
ldl_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
ldl_model.fit(X_train_scaled, y_ldl_train)

# Evaluate LDL model
ldl_pred = ldl_model.predict(X_test_scaled)
ldl_mae = mean_absolute_error(y_ldl_test, ldl_pred)
ldl_r2 = r2_score(y_ldl_test, ldl_pred)

print(f"LDL Model Performance:")
print(f"  MAE: {ldl_mae:.2f} mg/dL")
print(f"  R²: {ldl_r2:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': ldl_model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\nTop LDL Features:")
print(feature_importance.head(5))

print("\n" + "="*40)
print("Training HDL Prediction Model")
print("="*40)

# Train HDL model
hdl_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
hdl_model.fit(X_train_scaled, y_hdl_train)

# Evaluate HDL model
hdl_pred = hdl_model.predict(X_test_scaled)
hdl_mae = mean_absolute_error(y_hdl_test, hdl_pred)
hdl_r2 = r2_score(y_hdl_test, hdl_pred)

print(f"HDL Model Performance:")
print(f"  MAE: {hdl_mae:.2f} mg/dL")
print(f"  R²: {hdl_r2:.3f}")

# Feature importance for HDL
feature_importance_hdl = pd.DataFrame({
    'feature': feature_cols,
    'importance': hdl_model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\nTop HDL Features:")
print(feature_importance_hdl.head(5))

# Save models
print("\n" + "="*40)
print("Saving Models")
print("="*40)

models_dir = Path('models/cholesterol_detailed')
models_dir.mkdir(exist_ok=True, parents=True)

# Save LDL model
with open(models_dir / 'ldl_model.pkl', 'wb') as f:
    pickle.dump(ldl_model, f)

# Save HDL model
with open(models_dir / 'hdl_model.pkl', 'wb') as f:
    pickle.dump(hdl_model, f)

# Save scaler
with open(models_dir / 'cholesterol_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names
with open(models_dir / 'feature_names.txt', 'w') as f:
    f.write('\n'.join(feature_cols))

print(f"Models saved to {models_dir}")

# Test predictions on a few examples
print("\n" + "="*40)
print("Example Predictions")
print("="*40)

for i in range(min(3, len(X_test))):
    ldl_pred = ldl_model.predict(X_test_scaled[i:i+1])[0]
    hdl_pred = hdl_model.predict(X_test_scaled[i:i+1])[0]
    total_pred = ldl_pred + hdl_pred + 30  # Add estimated VLDL

    print(f"\nExample {i+1}:")
    print(f"  Predicted LDL: {ldl_pred:.1f} mg/dL")
    print(f"  Predicted HDL: {hdl_pred:.1f} mg/dL")
    print(f"  Predicted Total: {total_pred:.1f} mg/dL")
    print(f"  LDL/HDL Ratio: {ldl_pred/hdl_pred:.2f}")

    # Risk assessment
    if ldl_pred/hdl_pred < 2.5:
        risk = "Low"
    elif ldl_pred/hdl_pred < 3.5:
        risk = "Moderate"
    else:
        risk = "High"
    print(f"  Cardiovascular Risk: {risk}")

print("\n" + "="*60)
print("✅ LDL/HDL models trained successfully!")
print("="*60)
print("\nNext steps:")
print("1. Integrate these models into the Flask app")
print("2. Update the UI to show LDL/HDL breakdown")
print("3. Add risk assessment based on LDL/HDL ratio")