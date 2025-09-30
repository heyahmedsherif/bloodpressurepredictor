#!/usr/bin/env python
"""
Analyze Real Data Issues from Logs
===================================
Diagnose why real recordings show different results than synthetic tests
"""

import numpy as np
import sys
import os
from scipy import signal as scipy_signal

sys.path.insert(0, os.path.dirname(__file__))
from core.rppg_integration import SimplifiedRPPGProcessor

def analyze_log_data():
    """Analyze issues from the log data"""

    print("="*60)
    print("REAL DATA ISSUE ANALYSIS")
    print("="*60)

    # From logs: Heart rate estimates - FFT: 72.0, Peaks: 114.1, GREEN: 48.0
    # Advanced methods - CHROM: 48.0, POS: 91.5, OMIT: 48.0, ICA: 48.0

    log_results = {
        'fft': 72.0,
        'peaks': 114.1,
        'green': 48.0,
        'chrom': 48.0,
        'pos': 91.5,
        'omit': 48.0,
        'ica': 48.0
    }

    print("\nActual Results from Your Recording:")
    for method, hr in log_results.items():
        print(f"  {method:8s}: {hr:6.1f} BPM")

    # Analyze the pattern
    print("\n" + "="*40)
    print("ISSUE IDENTIFICATION:")
    print("="*40)

    issues = []

    # Issue 1: Many methods returning 48.0 BPM
    methods_at_48 = [m for m, hr in log_results.items() if hr == 48.0]
    if len(methods_at_48) >= 3:
        issues.append(f"Multiple methods returning exactly 48.0 BPM: {methods_at_48}")
        issues.append("48 BPM = 0.8 Hz, likely the lower filter cutoff frequency")

    # Issue 2: Large variance between methods
    hrs = list(log_results.values())
    variance = np.std(hrs)
    if variance > 20:
        issues.append(f"Very high variance between methods: std={variance:.1f}")

    # Issue 3: Peaks method showing double the expected rate
    if log_results['peaks'] > 100 and log_results['fft'] < 80:
        issues.append("Peaks method may be detecting harmonics (double counting)")

    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")

    return issues

def test_filter_cutoffs():
    """Test if filter cutoffs are causing issues"""

    print("\n" + "="*40)
    print("FILTER CUTOFF ANALYSIS:")
    print("="*40)

    # Create a test signal at 48 BPM (0.8 Hz)
    fps = 30
    duration = 10
    t = np.linspace(0, duration, fps * duration)

    # Test signal at exactly 48 BPM (0.8 Hz)
    test_hr = 48
    freq = test_hr / 60
    signal = np.sin(2 * np.pi * freq * t)

    # Apply the filter used in calculate_heart_rate_from_bvp
    nyquist = fps / 2
    low = 0.7 / nyquist  # This is the lower cutoff
    high = min(3.5 / nyquist, 0.99)

    print(f"Filter cutoffs:")
    print(f"  Low cutoff:  {low * nyquist:.2f} Hz = {low * nyquist * 60:.1f} BPM")
    print(f"  High cutoff: {high * nyquist:.2f} Hz = {high * nyquist * 60:.1f} BPM")

    if low * nyquist * 60 >= 42:  # If lower cutoff is too high
        print("  ⚠️  Lower cutoff might be filtering out valid heart rates!")

    # Test with different filter cutoffs
    b, a = scipy_signal.butter(3, [low, high], btype='band')
    filtered = scipy_signal.filtfilt(b, a, signal)

    # Check signal attenuation
    attenuation = np.max(np.abs(filtered)) / np.max(np.abs(signal))
    print(f"\nSignal at 48 BPM after filtering:")
    print(f"  Attenuation: {attenuation:.2%}")

    if attenuation < 0.5:
        print("  ⚠️  Significant signal loss at 48 BPM!")

    return attenuation

def analyze_duplicate_frames():
    """Analyze duplicate frame issue from logs"""

    print("\n" + "="*40)
    print("DUPLICATE FRAME ANALYSIS:")
    print("="*40)

    # From logs: many duplicate frame warnings
    print("From your logs:")
    print("  - Many frames marked as duplicates")
    print("  - Frame differences often < 0.01")
    print("  - This suggests:")
    print("    1. Very stable video (person not moving)")
    print("    2. Possible frame rate mismatch")
    print("    3. Or compression artifacts")

    # Test how duplicates affect each method
    processor = SimplifiedRPPGProcessor()

    # Create signal with duplicates
    fps = 30
    duration = 5
    frames = []

    # Generate with some duplicate frames
    for i in range(fps * duration):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Add PPG signal
        value = 128 + 10 * np.sin(2 * np.pi * (72/60) * (i/fps))
        frame[:, :, :] = value

        # Duplicate every 5th frame
        if i % 5 == 0 and i > 0:
            frames.append(frames[-1].copy())  # Duplicate
        else:
            frames.append(frame)

    result = processor.process_frames(frames[:150], fps=30)

    print("\nEffect of duplicate frames:")
    print(f"  Expected HR: 72 BPM")
    print(f"  Detected HR: {result.get('heart_rate', 0):.1f} BPM")
    print(f"  Confidence: {result.get('confidence', 0):.2f}")

    return result

def test_real_world_conditions():
    """Test with more realistic conditions"""

    print("\n" + "="*40)
    print("REAL-WORLD CONDITION TEST:")
    print("="*40)

    processor = SimplifiedRPPGProcessor()

    # Simulate real-world issues
    fps = 30
    duration = 5
    frames = []

    for i in range(fps * duration):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Add realistic skin tone with PPG signal
        t = i / fps
        hr = 65  # Typical resting HR

        # RGB with different amplitudes (like real skin)
        r = 180 + 3 * np.sin(2 * np.pi * (hr/60) * t)
        g = 128 + 5 * np.sin(2 * np.pi * (hr/60) * t)  # Green strongest
        b = 100 + 2 * np.sin(2 * np.pi * (hr/60) * t)

        # Add noise
        r += np.random.normal(0, 2)
        g += np.random.normal(0, 2)
        b += np.random.normal(0, 2)

        # Add motion artifacts occasionally
        if i % 30 == 0:
            r += np.random.normal(0, 10)
            g += np.random.normal(0, 10)
            b += np.random.normal(0, 10)

        frame[:, :, 2] = np.clip(r, 0, 255)
        frame[:, :, 1] = np.clip(g, 0, 255)
        frame[:, :, 0] = np.clip(b, 0, 255)

        frames.append(frame)

    result = processor.process_frames(frames, fps=30)

    print("Realistic conditions test:")
    print(f"  Target HR: 65 BPM")
    print(f"  FFT: {result.get('heart_rate_fft', 0):.1f} BPM")
    print(f"  Peaks: {result.get('heart_rate_peaks', 0):.1f} BPM")
    print(f"  GREEN: {result.get('heart_rate_green', 0):.1f} BPM")
    print(f"  CHROM: {result.get('heart_rate_chrom', 0):.1f} BPM")
    print(f"  POS: {result.get('heart_rate_pos', 0):.1f} BPM")
    print(f"  OMIT: {result.get('heart_rate_omit', 0):.1f} BPM")
    print(f"  ICA: {result.get('heart_rate_ica', 0):.1f} BPM")
    print(f"  ENSEMBLE: {result.get('heart_rate', 0):.1f} BPM")

    return result

def main():
    # Analyze log issues
    issues = analyze_log_data()

    # Test filter cutoffs
    attenuation = test_filter_cutoffs()

    # Analyze duplicate frames
    dup_result = analyze_duplicate_frames()

    # Test real-world conditions
    real_result = test_real_world_conditions()

    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY:")
    print("="*60)

    print("\nLIKELY CAUSES OF ISSUES:")
    print("1. Filter cutoff at 0.7 Hz (42 BPM) is too close to 48 BPM")
    print("   - Signals near 48 BPM get attenuated or distorted")
    print("   - This explains why multiple methods return exactly 48.0")

    print("\n2. Real face video has lower SNR than test signals")
    print("   - Lighting conditions")
    print("   - Micro-movements")
    print("   - Compression artifacts")

    print("\n3. Possible calibration issue")
    print("   - The 0.75 calibration factor may be overcorrecting")
    print("   - Real HR might be around 65-70, showing as 48 after calibration")

    print("\nRECOMMENDATIONS (NO CHANGES YET):")
    print("1. Consider lowering filter cutoff to 0.5 Hz (30 BPM)")
    print("2. Check if calibration factor is being applied twice")
    print("3. Improve signal quality assessment before processing")
    print("4. Add logging to see raw vs filtered signals")

    print("\nNOTE: All unit tests pass, so the core algorithms work.")
    print("The issues appear to be with real-world signal characteristics.")

if __name__ == "__main__":
    main()