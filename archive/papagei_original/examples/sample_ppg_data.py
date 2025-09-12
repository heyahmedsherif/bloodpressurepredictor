#!/usr/bin/env python3
"""
Sample PPG Data Generator

This script generates realistic PPG signals for testing the cardiovascular risk predictor.
Includes various physiological conditions and signal qualities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import os

def generate_normal_ppg(duration=60, fs=250, heart_rate=70, noise_level=0.1):
    """Generate normal PPG signal"""
    t = np.linspace(0, duration, int(duration * fs))
    hr_hz = heart_rate / 60
    
    # Base PPG pulse
    ppg = np.sin(2 * np.pi * hr_hz * t)
    
    # Add dicrotic notch
    ppg += 0.3 * np.sin(2 * np.pi * hr_hz * t * 2 + np.pi/4)
    
    # Add respiratory modulation
    ppg += 0.2 * np.sin(2 * np.pi * 0.25 * t)  # 15 breaths/min
    
    # Add noise
    ppg += noise_level * np.random.normal(0, 1, len(t))
    
    return t, ppg

def generate_hypertensive_ppg(duration=60, fs=250, heart_rate=85, noise_level=0.15):
    """Generate PPG signal characteristic of hypertension"""
    t = np.linspace(0, duration, int(duration * fs))
    hr_hz = heart_rate / 60
    
    # Base PPG with altered morphology
    ppg = np.sin(2 * np.pi * hr_hz * t)
    
    # Reduced dicrotic notch (characteristic of stiff arteries)
    ppg += 0.15 * np.sin(2 * np.pi * hr_hz * t * 2 + np.pi/3)
    
    # Increased baseline variation
    ppg += 0.3 * np.sin(2 * np.pi * 0.3 * t)
    
    # Higher noise (poorer signal quality)
    ppg += noise_level * np.random.normal(0, 1, len(t))
    
    # Add occasional artifacts
    artifacts = np.random.rand(len(t)) < 0.001
    ppg[artifacts] += 2 * np.random.randn(np.sum(artifacts))
    
    return t, ppg

def generate_elderly_ppg(duration=60, fs=250, heart_rate=65, noise_level=0.2):
    """Generate PPG signal characteristic of elderly patients"""
    t = np.linspace(0, duration, int(duration * fs))
    hr_hz = heart_rate / 60
    
    # Base PPG with irregular rhythm
    heart_rate_variation = 0.1 * np.sin(2 * np.pi * 0.1 * t)  # HRV
    instantaneous_hr = hr_hz * (1 + heart_rate_variation)
    
    phase = np.cumsum(instantaneous_hr) * 2 * np.pi / fs
    ppg = np.sin(phase)
    
    # Altered pulse wave morphology
    ppg += 0.1 * np.sin(phase * 2 + np.pi/6)
    
    # Reduced amplitude variation
    ppg += 0.15 * np.sin(2 * np.pi * 0.2 * t)
    
    # Higher noise
    ppg += noise_level * np.random.normal(0, 1, len(t))
    
    return t, ppg

def generate_young_athlete_ppg(duration=60, fs=250, heart_rate=50, noise_level=0.05):
    """Generate PPG signal characteristic of young athletes"""
    t = np.linspace(0, duration, int(duration * fs))
    hr_hz = heart_rate / 60
    
    # Base PPG with strong pulse
    ppg = 1.5 * np.sin(2 * np.pi * hr_hz * t)
    
    # Prominent dicrotic notch
    ppg += 0.4 * np.sin(2 * np.pi * hr_hz * t * 2 + np.pi/4)
    
    # Clear respiratory variation
    ppg += 0.3 * np.sin(2 * np.pi * 0.2 * t)  # 12 breaths/min
    
    # Low noise (good signal quality)
    ppg += noise_level * np.random.normal(0, 1, len(t))
    
    return t, ppg

def save_ppg_data(t, ppg, filename, metadata=None):
    """Save PPG data to CSV file"""
    df = pd.DataFrame({
        'time': t,
        'ppg': ppg
    })
    
    # Add metadata as comments in the file
    with open(filename, 'w') as f:
        f.write(f"# PPG Data Generated: {datetime.now().isoformat()}\n")
        if metadata:
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
        f.write("#\n")
    
    # Append the CSV data
    df.to_csv(filename, mode='a', index=False)
    print(f"Saved PPG data to {filename}")

def plot_ppg_comparison(datasets):
    """Plot comparison of different PPG signals"""
    plt.figure(figsize=(15, 10))
    
    for i, (name, t, ppg) in enumerate(datasets):
        plt.subplot(len(datasets), 1, i+1)
        plt.plot(t[:2500], ppg[:2500])  # Plot first 10 seconds
        plt.title(f"{name} PPG Signal (First 10 seconds)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("examples/ppg_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved comparison plot to examples/ppg_comparison.png")

def main():
    parser = argparse.ArgumentParser(description="Generate sample PPG data")
    parser.add_argument('--duration', type=int, default=60, help="Duration in seconds")
    parser.add_argument('--fs', type=int, default=250, help="Sampling frequency")
    parser.add_argument('--output_dir', type=str, default="examples", help="Output directory")
    parser.add_argument('--plot', action='store_true', help="Generate comparison plot")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    datasets = []
    
    # Generate normal PPG
    t, ppg_normal = generate_normal_ppg(args.duration, args.fs, heart_rate=70)
    filename = os.path.join(args.output_dir, "normal_ppg.csv")
    metadata = {"condition": "normal", "heart_rate": 70, "age_group": "adult", "risk_level": "low"}
    save_ppg_data(t, ppg_normal, filename, metadata)
    datasets.append(("Normal Adult", t, ppg_normal))
    
    # Generate hypertensive PPG
    t, ppg_hyper = generate_hypertensive_ppg(args.duration, args.fs, heart_rate=85)
    filename = os.path.join(args.output_dir, "hypertensive_ppg.csv")
    metadata = {"condition": "hypertensive", "heart_rate": 85, "age_group": "middle_aged", "risk_level": "high"}
    save_ppg_data(t, ppg_hyper, filename, metadata)
    datasets.append(("Hypertensive", t, ppg_hyper))
    
    # Generate elderly PPG
    t, ppg_elderly = generate_elderly_ppg(args.duration, args.fs, heart_rate=65)
    filename = os.path.join(args.output_dir, "elderly_ppg.csv")
    metadata = {"condition": "elderly", "heart_rate": 65, "age_group": "elderly", "risk_level": "moderate"}
    save_ppg_data(t, ppg_elderly, filename, metadata)
    datasets.append(("Elderly", t, ppg_elderly))
    
    # Generate young athlete PPG
    t, ppg_athlete = generate_young_athlete_ppg(args.duration, args.fs, heart_rate=50)
    filename = os.path.join(args.output_dir, "athlete_ppg.csv")
    metadata = {"condition": "athlete", "heart_rate": 50, "age_group": "young", "risk_level": "very_low"}
    save_ppg_data(t, ppg_athlete, filename, metadata)
    datasets.append(("Young Athlete", t, ppg_athlete))
    
    if args.plot:
        plot_ppg_comparison(datasets)
    
    print(f"\nGenerated {len(datasets)} PPG datasets in {args.output_dir}/")
    print("These files can be uploaded to the Streamlit app for testing.")

if __name__ == "__main__":
    main()