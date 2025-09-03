#!/usr/bin/env python3
"""
Blood Pressure Model Validation Script

This script validates the accuracy and performance of the blood pressure prediction models
using cross-validation, holdout testing, and clinical scenario validation.
"""

import sys
sys.path.append('apps')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from datetime import datetime

from bp_predictor import BPPredictor, interpret_bp_values

def validate_model_accuracy():
    """Comprehensive model validation"""
    
    print("🔍 Blood Pressure Model Validation")
    print("=" * 50)
    
    # Initialize predictor
    bp_predictor = BPPredictor()
    
    # Generate large validation dataset
    print("📊 Generating validation dataset...")
    df = bp_predictor.generate_synthetic_data(2000)
    print(f"✅ Generated {len(df)} samples")
    
    # Train models
    print("\n🔧 Training models...")
    bp_predictor.train_models(df)
    print("✅ Models trained successfully")
    
    # Cross-validation
    print("\n🎯 Performing cross-validation...")
    
    feature_names = bp_predictor.get_feature_names()
    X = df[feature_names]
    y_systolic = df['systolic_bp']
    y_diastolic = df['diastolic_bp']
    
    cv_results = {}
    
    for model_name in ['random_forest', 'gradient_boost']:
        print(f"\nValidating {model_name}...")
        
        # Systolic BP cross-validation
        sys_scores = cross_val_score(
            bp_predictor.models['systolic'][model_name],
            X, y_systolic, cv=5, scoring='neg_mean_absolute_error'
        )
        
        # Diastolic BP cross-validation  
        dia_scores = cross_val_score(
            bp_predictor.models['diastolic'][model_name],
            X, y_diastolic, cv=5, scoring='neg_mean_absolute_error'
        )
        
        cv_results[model_name] = {
            'systolic_mae': -sys_scores.mean(),
            'systolic_std': sys_scores.std(),
            'diastolic_mae': -dia_scores.mean(), 
            'diastolic_std': dia_scores.std()
        }
        
        print(f"  Systolic MAE: {-sys_scores.mean():.2f} ± {sys_scores.std():.2f} mmHg")
        print(f"  Diastolic MAE: {-dia_scores.mean():.2f} ± {dia_scores.std():.2f} mmHg")
    
    return cv_results, bp_predictor

def test_clinical_scenarios(bp_predictor):
    """Test model on specific clinical scenarios"""
    
    print("\n🏥 Testing Clinical Scenarios")
    print("=" * 40)
    
    scenarios = [
        {
            'name': 'Healthy Young Adult',
            'features': {
                'age': 25, 'gender_male': 0, 'bmi': 22,
                'smoking_status': 0, 'alcohol_weekly_units': 3, 'exercise_hours_week': 5,
                'sleep_hours': 8, 'stress_level': 3,
                'resting_heart_rate': 65, 'cholesterol_total': 180, 'cholesterol_hdl': 60,
                'glucose_fasting': 90, 'sodium_intake': 2000, 'potassium_intake': 3500,
                'family_history_hypertension': 0, 'diabetes': 0, 'kidney_disease': 0,
                'waist_circumference': 70, 'neck_circumference': 32
            },
            'expected_range': {'systolic': (100, 120), 'diastolic': (60, 80)}
        },
        {
            'name': 'High Risk Middle-Aged Male',
            'features': {
                'age': 55, 'gender_male': 1, 'bmi': 32,
                'smoking_status': 2, 'alcohol_weekly_units': 20, 'exercise_hours_week': 1,
                'sleep_hours': 6, 'stress_level': 8,
                'resting_heart_rate': 85, 'cholesterol_total': 260, 'cholesterol_hdl': 35,
                'glucose_fasting': 130, 'sodium_intake': 3500, 'potassium_intake': 2200,
                'family_history_hypertension': 1, 'diabetes': 1, 'kidney_disease': 0,
                'waist_circumference': 105, 'neck_circumference': 42
            },
            'expected_range': {'systolic': (140, 180), 'diastolic': (90, 110)}
        },
        {
            'name': 'Elderly with Multiple Conditions',
            'features': {
                'age': 75, 'gender_male': 0, 'bmi': 28,
                'smoking_status': 1, 'alcohol_weekly_units': 2, 'exercise_hours_week': 2,
                'sleep_hours': 6.5, 'stress_level': 6,
                'resting_heart_rate': 75, 'cholesterol_total': 240, 'cholesterol_hdl': 45,
                'glucose_fasting': 115, 'sodium_intake': 2800, 'potassium_intake': 2800,
                'family_history_hypertension': 1, 'diabetes': 0, 'kidney_disease': 1,
                'waist_circumference': 95, 'neck_circumference': 36
            },
            'expected_range': {'systolic': (130, 160), 'diastolic': (80, 100)}
        },
        {
            'name': 'Athletic Young Male',
            'features': {
                'age': 28, 'gender_male': 1, 'bmi': 21,
                'smoking_status': 0, 'alcohol_weekly_units': 5, 'exercise_hours_week': 12,
                'sleep_hours': 8.5, 'stress_level': 2,
                'resting_heart_rate': 50, 'cholesterol_total': 160, 'cholesterol_hdl': 70,
                'glucose_fasting': 85, 'sodium_intake': 1800, 'potassium_intake': 4000,
                'family_history_hypertension': 0, 'diabetes': 0, 'kidney_disease': 0,
                'waist_circumference': 75, 'neck_circumference': 38
            },
            'expected_range': {'systolic': (100, 125), 'diastolic': (65, 80)}
        }
    ]
    
    scenario_results = []
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['name']}:")
        
        prediction = bp_predictor.predict_bp(scenario['features'], 'gradient_boost')
        interpretation = interpret_bp_values(prediction['systolic_bp'], prediction['diastolic_bp'])
        
        # Check if prediction is in expected range
        sys_in_range = (scenario['expected_range']['systolic'][0] <= 
                       prediction['systolic_bp'] <= 
                       scenario['expected_range']['systolic'][1])
        
        dia_in_range = (scenario['expected_range']['diastolic'][0] <= 
                       prediction['diastolic_bp'] <= 
                       scenario['expected_range']['diastolic'][1])
        
        print(f"  Predicted: {prediction['systolic_bp']:.0f}/{prediction['diastolic_bp']:.0f} mmHg")
        print(f"  Expected: {scenario['expected_range']['systolic'][0]}-{scenario['expected_range']['systolic'][1]}/{scenario['expected_range']['diastolic'][0]}-{scenario['expected_range']['diastolic'][1]} mmHg")
        print(f"  Category: {interpretation['category']}")
        print(f"  In Range: Systolic {'✅' if sys_in_range else '❌'}, Diastolic {'✅' if dia_in_range else '❌'}")
        
        scenario_results.append({
            'name': scenario['name'],
            'predicted_systolic': prediction['systolic_bp'],
            'predicted_diastolic': prediction['diastolic_bp'],
            'systolic_in_range': sys_in_range,
            'diastolic_in_range': dia_in_range,
            'category': interpretation['category']
        })
    
    return scenario_results

def feature_correlation_analysis(bp_predictor):
    """Analyze feature correlations with blood pressure"""
    
    print("\n📈 Feature Correlation Analysis")
    print("=" * 35)
    
    df = bp_predictor.training_data
    feature_names = bp_predictor.get_feature_names()
    
    # Calculate correlations
    correlations = {}
    
    for feature in feature_names:
        sys_corr = df[feature].corr(df['systolic_bp'])
        dia_corr = df[feature].corr(df['diastolic_bp'])
        correlations[feature] = {
            'systolic': sys_corr,
            'diastolic': dia_corr
        }
    
    # Sort by systolic correlation
    sorted_features = sorted(correlations.items(), 
                           key=lambda x: abs(x[1]['systolic']), 
                           reverse=True)
    
    print("Top features correlated with systolic BP:")
    for feature, corrs in sorted_features[:10]:
        print(f"  {feature.replace('_', ' ').title()}: {corrs['systolic']:.3f}")
    
    return correlations

def generate_validation_plots(bp_predictor, cv_results):
    """Generate validation plots"""
    
    print("\n📊 Generating validation plots...")
    
    df = bp_predictor.training_data
    feature_names = bp_predictor.get_feature_names()
    X = df[feature_names]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Blood Pressure Model Validation Results', fontsize=16)
    
    # 1. Model Performance Comparison
    models = list(cv_results.keys())
    sys_maes = [cv_results[model]['systolic_mae'] for model in models]
    dia_maes = [cv_results[model]['diastolic_mae'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0,0].bar(x - width/2, sys_maes, width, label='Systolic', alpha=0.8)
    axes[0,0].bar(x + width/2, dia_maes, width, label='Diastolic', alpha=0.8)
    axes[0,0].set_xlabel('Model')
    axes[0,0].set_ylabel('Mean Absolute Error (mmHg)')
    axes[0,0].set_title('Model Performance Comparison')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels([m.replace('_', ' ').title() for m in models])
    axes[0,0].legend()
    
    # 2. Predicted vs Actual (Gradient Boosting)
    gb_sys_pred = bp_predictor.models['systolic']['gradient_boost'].predict(X)
    gb_dia_pred = bp_predictor.models['diastolic']['gradient_boost'].predict(X)
    
    axes[0,1].scatter(df['systolic_bp'], gb_sys_pred, alpha=0.5, s=20)
    axes[0,1].plot([90, 200], [90, 200], 'r--', lw=2)
    axes[0,1].set_xlabel('Actual Systolic BP (mmHg)')
    axes[0,1].set_ylabel('Predicted Systolic BP (mmHg)')
    axes[0,1].set_title('Predicted vs Actual - Systolic BP')
    
    # 3. Feature Importance
    importance = bp_predictor.get_feature_importance('gradient_boost')
    if not importance.empty:
        top_features = importance.head(8)
        axes[1,0].barh(range(len(top_features)), top_features['systolic_importance'])
        axes[1,0].set_yticks(range(len(top_features)))
        axes[1,0].set_yticklabels([f.replace('_', ' ').title() for f in top_features['feature']])
        axes[1,0].set_xlabel('Feature Importance')
        axes[1,0].set_title('Top Features for Systolic BP')
    
    # 4. BP Distribution
    axes[1,1].hist(df['systolic_bp'], bins=30, alpha=0.7, label='Systolic', density=True)
    axes[1,1].hist(df['diastolic_bp'], bins=30, alpha=0.7, label='Diastolic', density=True)
    axes[1,1].set_xlabel('Blood Pressure (mmHg)')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('BP Distribution in Training Data')
    axes[1,1].legend()
    
    plt.tight_layout()
    
    # Save plot
    output_file = f'bp_validation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Validation plots saved as {output_file}")
    
    return output_file

def main():
    """Main validation function"""
    
    print("🩺 BLOOD PRESSURE MODEL VALIDATION SUITE")
    print("=" * 60)
    
    # Model accuracy validation
    cv_results, bp_predictor = validate_model_accuracy()
    
    # Clinical scenario testing
    scenario_results = test_clinical_scenarios(bp_predictor)
    
    # Feature correlation analysis
    correlations = feature_correlation_analysis(bp_predictor)
    
    # Generate plots
    plot_file = generate_validation_plots(bp_predictor, cv_results)
    
    # Summary report
    print("\n📋 VALIDATION SUMMARY")
    print("=" * 25)
    
    print("Model Performance (Cross-Validation):")
    for model, results in cv_results.items():
        print(f"  {model.replace('_', ' ').title()}:")
        print(f"    Systolic MAE: {results['systolic_mae']:.2f} ± {results['systolic_std']:.2f} mmHg")
        print(f"    Diastolic MAE: {results['diastolic_mae']:.2f} ± {results['diastolic_std']:.2f} mmHg")
    
    print(f"\nClinical Scenario Accuracy:")
    total_scenarios = len(scenario_results)
    sys_correct = sum(1 for s in scenario_results if s['systolic_in_range'])
    dia_correct = sum(1 for s in scenario_results if s['diastolic_in_range'])
    
    print(f"  Systolic predictions in expected range: {sys_correct}/{total_scenarios} ({sys_correct/total_scenarios*100:.1f}%)")
    print(f"  Diastolic predictions in expected range: {dia_correct}/{total_scenarios} ({dia_correct/total_scenarios*100:.1f}%)")
    
    # Generate comprehensive report
    report = {
        'timestamp': datetime.now().isoformat(),
        'validation_type': 'blood_pressure_model_validation',
        'cross_validation_results': cv_results,
        'clinical_scenarios': scenario_results,
        'top_correlations': {
            feature: corrs for feature, corrs in 
            sorted(correlations.items(), key=lambda x: abs(x[1]['systolic']), reverse=True)[:10]
        },
        'summary': {
            'best_model': min(cv_results.items(), key=lambda x: x[1]['systolic_mae'])[0],
            'scenario_accuracy': {
                'systolic': f"{sys_correct}/{total_scenarios}",
                'diastolic': f"{dia_correct}/{total_scenarios}"
            },
            'plot_file': plot_file
        }
    }
    
    # Save report
    report_file = f'bp_validation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Validation complete!")
    print(f"📊 Plots saved: {plot_file}")
    print(f"📄 Report saved: {report_file}")
    print(f"\n🚀 Start the BP Predictor app:")
    print(f"streamlit run apps/bp_predictor.py")

if __name__ == "__main__":
    main()