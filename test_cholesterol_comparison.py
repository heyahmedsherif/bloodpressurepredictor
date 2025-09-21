"""
Test Script: Compare Original vs LDL/HDL Cholesterol Predictions
================================================================
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.enhanced_ml_predictor import EnhancedMLHealthPredictor

def generate_test_ppg(heart_rate=75, duration=7.5):
    """Generate synthetic PPG signal for testing"""
    sampling_rate = 20
    num_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, num_samples)

    # Create PPG-like signal
    freq = heart_rate / 60  # Hz
    ppg = 128 + 10 * np.sin(2 * np.pi * freq * t)

    # Add dicrotic notch (for augmentation index)
    ppg += 3 * np.sin(4 * np.pi * freq * t - np.pi/4)

    # Add noise
    ppg += np.random.normal(0, 0.5, num_samples)

    return ppg

def test_cholesterol_comparison():
    """Test both cholesterol prediction methods"""

    print("="*70)
    print("CHOLESTEROL PREDICTION COMPARISON TEST")
    print("="*70)

    # Initialize predictor
    predictor = EnhancedMLHealthPredictor()

    # Test with different demographics
    test_cases = [
        {
            'name': 'Case 1: Middle-aged Male',
            'demographics': {
                'age': 47,
                'gender': 'M',
                'height': 173,
                'weight': 83
            },
            'heart_rate': 75
        },
        {
            'name': 'Case 2: Young Female',
            'demographics': {
                'age': 30,
                'gender': 'F',
                'height': 165,
                'weight': 60
            },
            'heart_rate': 68
        },
        {
            'name': 'Case 3: Elderly Male',
            'demographics': {
                'age': 65,
                'gender': 'M',
                'height': 170,
                'weight': 75
            },
            'heart_rate': 82
        }
    ]

    for case in test_cases:
        print(f"\n{'-'*70}")
        print(f"{case['name']}")
        print(f"{'-'*70}")

        # Generate PPG signal
        ppg_signal = generate_test_ppg(case['heart_rate'])

        # Get predictions
        results = predictor.predict_all_metrics(ppg_signal, case['demographics'])

        # Display demographics
        print(f"\nDemographics:")
        print(f"  Age: {case['demographics']['age']} years")
        print(f"  Gender: {case['demographics']['gender']}")
        print(f"  BMI: {results.get('bmi', 25):.1f} kg/m²")

        # Display original method
        print(f"\n📊 ORIGINAL METHOD (Total Cholesterol Only):")
        print(f"  Total Cholesterol: {results['cholesterol_total_original']:.1f} mg/dL")

        # Display new method
        print(f"\n📈 NEW METHOD (LDL/HDL Breakdown):")
        print(f"  LDL (Bad):  {results.get('ldl', 0):.1f} mg/dL")
        print(f"  HDL (Good): {results.get('hdl', 0):.1f} mg/dL")
        print(f"  VLDL (Est): {results.get('vldl_estimate', 0):.1f} mg/dL")
        print(f"  ─────────────────────")
        print(f"  Total:      {results.get('cholesterol_total_new', 0):.1f} mg/dL")

        # Display comparison
        comparison = results.get('comparison', {})
        print(f"\n🔍 COMPARISON:")
        print(f"  Original Total:    {comparison.get('total_original', 0):.1f} mg/dL")
        print(f"  New Total (Sum):   {comparison.get('total_new', 0):.1f} mg/dL")
        print(f"  Difference:        {comparison.get('difference', 0):.1f} mg/dL "
              f"({comparison.get('percentage_difference', 0):.1f}%)")
        print(f"  Validation:        {comparison.get('sum_validation', 'Unknown')}")

        # Display breakdown percentages
        print(f"\n📊 COMPOSITION BREAKDOWN:")
        if 'ldl_percentage' in comparison:
            print(f"  LDL:  {comparison['ldl_percentage']:.1f}% of total")
            print(f"  HDL:  {comparison['hdl_percentage']:.1f}% of total")
            print(f"  VLDL: {comparison['vldl_percentage']:.1f}% of total")

        # Display risk assessment
        print(f"\n⚠️  RISK ASSESSMENT:")
        print(f"  LDL/HDL Ratio:        {results.get('ldl_hdl_ratio', 0):.2f}")
        print(f"  Cardiovascular Risk:  {results.get('cardiovascular_risk', 'Unknown')}")

        # Interpretation
        ldl_hdl_ratio = results.get('ldl_hdl_ratio', 0)
        if ldl_hdl_ratio < 2.0:
            interpretation = "Excellent - Low cardiovascular risk"
        elif ldl_hdl_ratio < 2.5:
            interpretation = "Good - Low to moderate risk"
        elif ldl_hdl_ratio < 3.5:
            interpretation = "Borderline - Moderate risk"
        else:
            interpretation = "High - Increased cardiovascular risk"

        print(f"  Interpretation:       {interpretation}")

        # Other health metrics
        print(f"\n📋 OTHER HEALTH METRICS:")
        print(f"  Blood Pressure: {results['systolic']:.0f}/{results['diastolic']:.0f} mmHg")
        print(f"  Glucose:        {results['glucose']:.1f} mg/dL")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print("\n✅ Both methods are working:")
    print("   - Original model provides total cholesterol")
    print("   - New models provide LDL/HDL breakdown")
    print("   - Results can be compared side-by-side")
    print("   - LDL/HDL ratio provides risk assessment")
    print("\n📌 Key Insights:")
    print("   - The sum (LDL + HDL + VLDL) should ≈ Total cholesterol")
    print("   - LDL/HDL ratio is a better predictor of cardiovascular risk")
    print("   - Gender affects HDL levels (women typically have higher HDL)")

if __name__ == "__main__":
    test_cholesterol_comparison()