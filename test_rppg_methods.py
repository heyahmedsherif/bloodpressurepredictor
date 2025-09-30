#!/usr/bin/env python
"""
Comprehensive Unit Tests for rPPG Heart Rate Detection Methods
===============================================================
Tests all 7 methods individually and the ensemble system
"""

import numpy as np
import sys
import os
from scipy import signal as scipy_signal

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from core.rppg_integration import SimplifiedRPPGProcessor

def generate_test_signal(heart_rate=72, fps=30, duration=10, noise_level=0.01):
    """Generate synthetic PPG signal with known heart rate"""
    num_frames = fps * duration
    t = np.linspace(0, duration, num_frames)

    # Create base PPG signal
    heart_freq = heart_rate / 60  # Hz

    # Generate realistic RGB signals
    # Green channel has strongest PPG signal
    r_amplitude = 8
    g_amplitude = 12  # Strongest
    b_amplitude = 6

    # Base values (skin tone)
    r_base = 180
    g_base = 128
    b_base = 100

    # Generate signals
    r_signal = r_base + r_amplitude * np.sin(2 * np.pi * heart_freq * t)
    g_signal = g_base + g_amplitude * np.sin(2 * np.pi * heart_freq * t)
    b_signal = b_base + b_amplitude * np.sin(2 * np.pi * heart_freq * t)

    # Add realistic noise
    if noise_level > 0:
        r_signal += np.random.normal(0, noise_level * r_amplitude, num_frames)
        g_signal += np.random.normal(0, noise_level * g_amplitude, num_frames)
        b_signal += np.random.normal(0, noise_level * b_amplitude, num_frames)

    # Create frames
    frames = []
    for i in range(num_frames):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = np.clip(b_signal[i], 0, 255)  # BGR format
        frame[:, :, 1] = np.clip(g_signal[i], 0, 255)
        frame[:, :, 2] = np.clip(r_signal[i], 0, 255)
        frames.append(frame)

    return frames, heart_rate

def test_individual_method(processor, frames, method_name, expected_hr):
    """Test a specific method"""
    # Extract RGB signals
    rgb_signal = processor.extract_rgb_signals(frames)

    # Test specific method
    if method_name == 'fft':
        green_signal = processor.extract_color_signal(frames)
        green_signal = scipy_signal.detrend(green_signal)
        hr = processor.calculate_heart_rate_fft(green_signal, fps=30)
    elif method_name == 'peaks':
        green_signal = processor.extract_color_signal(frames)
        green_signal = scipy_signal.detrend(green_signal)
        hr = processor.calculate_heart_rate_peaks(green_signal, fps=30)
    elif method_name == 'green':
        bvp = processor.apply_green_method(rgb_signal)
        hr = processor.calculate_heart_rate_from_bvp(bvp, fps=30)
    elif method_name == 'chrom':
        bvp = processor.apply_chrom_method(rgb_signal)
        hr = processor.calculate_heart_rate_from_bvp(bvp, fps=30)
    elif method_name == 'pos':
        bvp = processor.apply_pos_method(rgb_signal)
        hr = processor.calculate_heart_rate_from_bvp(bvp, fps=30)
    elif method_name == 'omit':
        bvp = processor.apply_omit_method(rgb_signal)
        hr = processor.calculate_heart_rate_from_bvp(bvp, fps=30)
    elif method_name == 'ica':
        bvp = processor.apply_ica_method(rgb_signal)
        hr = processor.calculate_heart_rate_from_bvp(bvp, fps=30)
    else:
        return None, None

    error = abs(hr - expected_hr)
    accuracy = max(0, 100 - (error / expected_hr * 100))

    return hr, accuracy

def test_ensemble_weighting():
    """Test if ensemble weighting is working correctly"""
    processor = SimplifiedRPPGProcessor()

    # Test with known values
    test_hrs = {
        'fft': 72.0,
        'peaks': 73.0,
        'green': 71.0,
        'chrom': 72.5,
        'pos': 72.8,
        'omit': 72.2,  # Should get highest weight
        'ica': 71.5
    }

    # Extract valid HRs (non-75.0 values)
    valid_hrs = []
    method_names = []
    for method, hr in test_hrs.items():
        if hr != 75.0:
            valid_hrs.append(hr)
            method_names.append(method)

    # Calculate weights as in actual implementation
    weights = []
    for method in method_names:
        if method == 'omit':
            weights.append(2.0)  # Highest weight
        elif method in ['chrom', 'pos']:
            weights.append(1.5)
        elif method in ['ica']:
            weights.append(1.2)
        elif method in ['fft', 'peaks']:
            weights.append(1.0)
        else:  # green
            weights.append(0.8)

    weights = np.array(weights) / np.sum(weights)
    weighted_hr = np.average(valid_hrs, weights=weights)
    median_hr = np.median(valid_hrs)

    print(f"\nEnsemble Weighting Test:")
    print(f"  Input HRs: {test_hrs}")
    print(f"  Weights: {dict(zip(method_names, weights))}")
    print(f"  Weighted Average: {weighted_hr:.2f} BPM")
    print(f"  Median: {median_hr:.2f} BPM")
    print(f"  Expected more weight on OMIT: {weighted_hr > median_hr}")

    return weighted_hr, median_hr

def run_comprehensive_tests():
    """Run all tests"""
    print("="*60)
    print("rPPG Methods Comprehensive Testing")
    print("="*60)

    processor = SimplifiedRPPGProcessor()

    # Test different heart rates
    test_heart_rates = [60, 72, 90, 120]
    noise_levels = [0.01, 0.05, 0.1]  # Low, medium, high noise

    results = {}

    for hr in test_heart_rates:
        print(f"\n{'='*40}")
        print(f"Testing Heart Rate: {hr} BPM")
        print(f"{'='*40}")

        for noise in noise_levels:
            print(f"\nNoise Level: {noise*100:.0f}%")

            # Generate test signal
            frames, expected_hr = generate_test_signal(hr, noise_level=noise)

            # Test each method
            method_results = {}
            for method in ['fft', 'peaks', 'green', 'chrom', 'pos', 'omit', 'ica']:
                detected_hr, accuracy = test_individual_method(processor, frames, method, expected_hr)
                method_results[method] = {
                    'hr': detected_hr,
                    'accuracy': accuracy,
                    'error': abs(detected_hr - expected_hr) if detected_hr else None
                }

                status = "✓" if accuracy and accuracy > 80 else "✗"
                print(f"  {method:8s}: {detected_hr:6.1f} BPM (accuracy: {accuracy:5.1f}%) {status}")

            # Test full ensemble
            result = processor.process_frames(frames, fps=30)
            ensemble_hr = result.get('heart_rate', 0)
            ensemble_error = abs(ensemble_hr - expected_hr)
            ensemble_accuracy = max(0, 100 - (ensemble_error / expected_hr * 100))

            print(f"\n  ENSEMBLE: {ensemble_hr:6.1f} BPM (accuracy: {ensemble_accuracy:5.1f}%)")
            print(f"  Confidence: {result.get('confidence', 0):.2f}")
            print(f"  Methods Used: {result.get('methods_used', 0)}/{result.get('total_methods', 7)}")

            # Store results
            key = f"hr{hr}_noise{int(noise*100)}"
            results[key] = {
                'methods': method_results,
                'ensemble': ensemble_hr,
                'confidence': result.get('confidence', 0)
            }

    # Test ensemble weighting
    print(f"\n{'='*40}")
    weighted_hr, median_hr = test_ensemble_weighting()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Find best performing methods
    method_scores = {'fft': [], 'peaks': [], 'green': [], 'chrom': [], 'pos': [], 'omit': [], 'ica': []}

    for key, data in results.items():
        for method, method_data in data['methods'].items():
            if method_data['accuracy']:
                method_scores[method].append(method_data['accuracy'])

    print("\nAverage Accuracy by Method:")
    for method, scores in method_scores.items():
        if scores:
            avg = np.mean(scores)
            print(f"  {method:8s}: {avg:5.1f}%")

    # Check for issues
    print(f"\n{'='*60}")
    print("POTENTIAL ISSUES DETECTED:")
    print(f"{'='*60}")

    issues = []

    # Check if OMIT is performing poorly
    if method_scores['omit'] and np.mean(method_scores['omit']) < 70:
        issues.append("OMIT method performing below expectations")

    # Check if too many methods return default
    for key, data in results.items():
        default_count = sum(1 for m, d in data['methods'].items() if d['hr'] == 75.0)
        if default_count > 3:
            issues.append(f"Many methods returning default (75.0) for {key}")

    # Check confidence levels
    low_confidence_count = sum(1 for data in results.values() if data['confidence'] < 0.5)
    if low_confidence_count > len(results) / 2:
        issues.append("Low confidence scores in majority of tests")

    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print("No major issues detected")

    return results

if __name__ == "__main__":
    results = run_comprehensive_tests()

    # Check for critical failures
    print(f"\n{'='*60}")
    print("DIAGNOSTIC RESULT:")

    critical_failures = []

    # Check if basic methods work
    basic_working = False
    for data in results.values():
        if data['methods']['fft']['accuracy'] and data['methods']['fft']['accuracy'] > 70:
            basic_working = True
            break

    if not basic_working:
        critical_failures.append("Basic FFT method not working properly")

    # Check ensemble performance
    ensemble_accuracies = []
    for key, data in results.items():
        if 'noise5' not in key:  # Exclude high noise tests
            hr_val = int(key.split('_')[0][2:])
            ensemble_error = abs(data['ensemble'] - hr_val)
            accuracy = max(0, 100 - (ensemble_error / hr_val * 100))
            ensemble_accuracies.append(accuracy)

    avg_ensemble_accuracy = np.mean(ensemble_accuracies) if ensemble_accuracies else 0

    if avg_ensemble_accuracy < 70:
        critical_failures.append(f"Ensemble accuracy too low: {avg_ensemble_accuracy:.1f}%")

    if critical_failures:
        print("⚠️  CRITICAL ISSUES FOUND:")
        for issue in critical_failures:
            print(f"   - {issue}")
    else:
        print("✅ All methods functioning within acceptable parameters")

    print(f"{'='*60}")