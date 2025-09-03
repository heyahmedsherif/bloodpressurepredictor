#!/usr/bin/env python3
"""
Quick Demo Script for Cardiovascular Risk Predictor

This script provides a complete demonstration of the cardiovascular risk predictor
without requiring the Streamlit interface. Useful for automated testing and
command-line demonstrations.
"""

import sys
import os
sys.path.append('.')
sys.path.append('./apps')
sys.path.append('./examples')

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Import our modules
from sample_ppg_data import generate_normal_ppg, generate_hypertensive_ppg
from apps.streamlit_app_robust import PPGProcessor, CardiovascularPredictor

def run_demo(patient_age=55, patient_gender="Male", condition="normal"):
    """Run complete cardiovascular risk prediction demo"""
    
    print("=" * 60)
    print("🏥 CARDIOVASCULAR RISK PREDICTOR DEMO")
    print("=" * 60)
    
    # Generate sample PPG data
    print(f"\n1. Generating {condition} PPG signal...")
    if condition == "normal":
        t, ppg_signal = generate_normal_ppg(duration=30, fs=250, heart_rate=70)
        expected_risk = "Low"
    elif condition == "hypertensive":
        t, ppg_signal = generate_hypertensive_ppg(duration=30, fs=250, heart_rate=85)
        expected_risk = "High"
    else:
        t, ppg_signal = generate_normal_ppg(duration=30, fs=250, heart_rate=70)
        expected_risk = "Low"
    
    fs = 250
    print(f"✅ Generated {len(ppg_signal)} samples at {fs} Hz ({len(ppg_signal)/fs:.1f} seconds)")
    
    # Initialize processors
    print("\n2. Initializing PaPaGei processors...")
    try:
        ppg_processor = PPGProcessor()
        cv_predictor = CardiovascularPredictor()
        print("✅ Processors initialized successfully")
    except Exception as e:
        print(f"❌ Processor initialization failed: {e}")
        return None
    
    # Process PPG signal
    print("\n3. Processing PPG signal...")
    try:
        segments, processed_signal = ppg_processor.process_ppg_signal(ppg_signal, fs)
        if segments is not None:
            print(f"✅ Signal processed: {len(segments)} segments of length {segments.shape[1]}")
        else:
            print("❌ PPG processing failed")
            return None
    except Exception as e:
        print(f"❌ PPG processing error: {e}")
        return None
    
    # Extract embeddings
    print("\n4. Extracting PaPaGei embeddings...")
    try:
        embeddings = cv_predictor.extract_embeddings(segments)
        if embeddings is not None:
            print(f"✅ Extracted embeddings: shape {embeddings.shape}")
        else:
            print("❌ Embedding extraction failed")
            return None
    except Exception as e:
        print(f"❌ Embedding extraction error: {e}")
        return None
    
    # Predict blood pressure
    print("\n5. Predicting blood pressure...")
    try:
        bp_prediction = cv_predictor.predict_blood_pressure(embeddings)
        print(f"✅ Blood Pressure Prediction:")
        print(f"   Systolic: {bp_prediction['systolic']:.1f} mmHg (CI: {bp_prediction['systolic_ci'][0]:.1f}-{bp_prediction['systolic_ci'][1]:.1f})")
        print(f"   Diastolic: {bp_prediction['diastolic']:.1f} mmHg (CI: {bp_prediction['diastolic_ci'][0]:.1f}-{bp_prediction['diastolic_ci'][1]:.1f})")
        print(f"   Confidence: {bp_prediction['confidence']*100:.1f}%")
    except Exception as e:
        print(f"❌ BP prediction error: {e}")
        return None
    
    # Calculate cardiovascular risk
    print("\n6. Calculating cardiovascular risk...")
    try:
        cv_risk = cv_predictor.calculate_cv_risk(bp_prediction, patient_age, patient_gender)
        print(f"✅ Cardiovascular Risk Assessment:")
        print(f"   Risk Score: {cv_risk['risk_score']:.1f}%")
        print(f"   Risk Category: {cv_risk['category']}")
        print(f"   Early Warning: {'🚨 YES' if cv_risk['early_warning'] else '✅ NO'}")
        print(f"   Confidence: {cv_risk['confidence']*100:.1f}%")
    except Exception as e:
        print(f"❌ CV risk calculation error: {e}")
        return None
    
    # Generate visualization
    print("\n7. Generating visualization...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Cardiovascular Risk Analysis - {condition.title()} Patient', fontsize=16)
        
        # Raw PPG signal
        axes[0, 0].plot(t[:2500], ppg_signal[:2500], 'b-', linewidth=1)
        axes[0, 0].set_title('Raw PPG Signal (First 10s)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Processed PPG signal  
        time_processed = np.arange(len(processed_signal)) / fs
        axes[0, 1].plot(time_processed[:2500], processed_signal[:2500], 'r-', linewidth=1)
        axes[0, 1].set_title('Processed PPG Signal (First 10s)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Blood pressure visualization
        bp_data = [bp_prediction['systolic'], bp_prediction['diastolic']]
        bp_labels = ['Systolic', 'Diastolic']
        colors = ['red' if bp_prediction['systolic'] > 140 else 'orange' if bp_prediction['systolic'] > 130 else 'green',
                 'red' if bp_prediction['diastolic'] > 90 else 'orange' if bp_prediction['diastolic'] > 80 else 'green']
        bars = axes[1, 0].bar(bp_labels, bp_data, color=colors, alpha=0.7)
        axes[1, 0].set_title('Blood Pressure Prediction')
        axes[1, 0].set_ylabel('mmHg')
        axes[1, 0].set_ylim(0, 200)
        
        # Add value labels on bars
        for bar, value in zip(bars, bp_data):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                           f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Risk score gauge (simplified)
        risk_score = cv_risk['risk_score']
        theta = np.linspace(0, np.pi, 100)
        risk_color = cv_risk['color']
        
        # Create gauge background
        axes[1, 1].plot(np.cos(theta), np.sin(theta), 'k-', linewidth=3)
        axes[1, 1].fill_between(np.cos(theta), 0, np.sin(theta), alpha=0.1)
        
        # Risk score needle
        needle_angle = np.pi * (1 - risk_score/100)
        needle_x = [0, 0.8 * np.cos(needle_angle)]
        needle_y = [0, 0.8 * np.sin(needle_angle)]
        axes[1, 1].plot(needle_x, needle_y, color=risk_color, linewidth=4)
        axes[1, 1].scatter([0], [0], color='black', s=100, zorder=5)
        
        axes[1, 1].set_xlim(-1.2, 1.2)
        axes[1, 1].set_ylim(-0.2, 1.2)
        axes[1, 1].set_aspect('equal')
        axes[1, 1].set_title(f'CV Risk: {risk_score:.1f}% ({cv_risk["category"]})')
        axes[1, 1].text(0, -0.1, f'{risk_score:.1f}%', ha='center', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        output_file = f'demo_results_{condition}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved as {output_file}")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
    
    # Generate summary report
    print("\n8. Generating summary report...")
    try:
        report = {
            "timestamp": datetime.now().isoformat(),
            "patient_info": {
                "age": patient_age,
                "gender": patient_gender,
                "condition": condition
            },
            "signal_info": {
                "duration_seconds": len(ppg_signal) / fs,
                "sampling_rate": fs,
                "num_segments": len(segments),
                "embedding_dimension": embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings)
            },
            "predictions": {
                "systolic_bp": bp_prediction['systolic'],
                "diastolic_bp": bp_prediction['diastolic'],
                "bp_confidence": bp_prediction['confidence'],
                "cv_risk_score": cv_risk['risk_score'],
                "cv_risk_category": cv_risk['category'],
                "early_warning": cv_risk['early_warning']
            },
            "clinical_interpretation": {
                "bp_category": "Hypertensive" if bp_prediction['systolic'] > 140 or bp_prediction['diastolic'] > 90 
                             else "Elevated" if bp_prediction['systolic'] > 130 or bp_prediction['diastolic'] > 80
                             else "Normal",
                "risk_assessment": cv_risk['category'],
                "recommendations": [
                    "Immediate medical consultation" if cv_risk['early_warning'] else "Continue monitoring",
                    "Lifestyle modifications" if cv_risk['risk_score'] > 40 else "Maintain healthy habits"
                ]
            }
        }
        
        report_file = f'demo_report_{condition}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Report saved as {report_file}")
        
    except Exception as e:
        print(f"⚠️ Report generation error: {e}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 DEMO SUMMARY")
    print("=" * 60)
    print(f"Patient: {patient_age}yo {patient_gender} ({condition} condition)")
    print(f"Blood Pressure: {bp_prediction['systolic']:.1f}/{bp_prediction['diastolic']:.1f} mmHg")
    print(f"CV Risk: {cv_risk['risk_score']:.1f}% ({cv_risk['category']})")
    print(f"Early Warning: {'🚨 ALERT' if cv_risk['early_warning'] else '✅ Normal'}")
    print(f"Expected vs Actual Risk: {expected_risk} vs {cv_risk['category'].split()[0]}")
    
    if cv_risk['early_warning']:
        print("\n🚨 CLINICAL ALERT: High cardiovascular risk detected!")
        print("   Recommend immediate medical evaluation")
    else:
        print("\n✅ No immediate cardiovascular concerns detected")
    
    print("\n🎯 Demo completed successfully!")
    print("   Run Streamlit app for interactive analysis:")
    print("   streamlit run apps/streamlit_app_robust.py")
    
    return report

def main():
    """Main demo function with multiple scenarios"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cardiovascular Risk Predictor Demo")
    parser.add_argument('--condition', type=str, choices=['normal', 'hypertensive'], 
                       default='normal', help="Patient condition")
    parser.add_argument('--age', type=int, default=55, help="Patient age")
    parser.add_argument('--gender', type=str, choices=['Male', 'Female'], 
                       default='Male', help="Patient gender")
    parser.add_argument('--all-scenarios', action='store_true', 
                       help="Run all demo scenarios")
    
    args = parser.parse_args()
    
    if args.all_scenarios:
        scenarios = [
            (35, "Female", "normal"),
            (55, "Male", "hypertensive"),
            (65, "Male", "normal"),
            (45, "Female", "hypertensive")
        ]
        
        print("Running all demo scenarios...")
        for i, (age, gender, condition) in enumerate(scenarios):
            print(f"\n{'='*20} SCENARIO {i+1}/4 {'='*20}")
            run_demo(age, gender, condition)
            if i < len(scenarios) - 1:
                input("\nPress Enter to continue to next scenario...")
    else:
        run_demo(args.age, args.gender, args.condition)

if __name__ == "__main__":
    main()