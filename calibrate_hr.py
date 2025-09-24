#!/usr/bin/env python3
"""
Heart Rate Calibration Tool
============================
This tool helps calibrate the HR detection to match your reference device (e.g., Apple Watch).
"""

import json
import requests
import time
import numpy as np
from datetime import datetime
import sys


class HRCalibrationTool:
    """Tool for calibrating heart rate detection"""

    def __init__(self, api_url="http://localhost:5001"):
        self.api_url = api_url
        self.measurements = []
        self.current_calibration_factor = 0.92  # Current default

    def take_measurement(self):
        """Prompt user to take a measurement"""
        print("\n" + "="*60)
        print("📊 Heart Rate Calibration Measurement")
        print("="*60)

        # Get reference HR from user
        while True:
            try:
                reference_hr = input("\n1️⃣  What's your Apple Watch HR reading right now? ")
                reference_hr = float(reference_hr)
                if 40 <= reference_hr <= 200:
                    break
                print("❌ HR should be between 40-200 BPM")
            except ValueError:
                print("❌ Please enter a valid number")

        print("\n2️⃣  Now click 'Start Recording' in the web app...")
        print("    Press Enter here AFTER the recording completes...")
        input()

        # Get app HR from user
        while True:
            try:
                app_hr = input("\n3️⃣  What HR did the app show? ")
                app_hr = float(app_hr)
                if 40 <= app_hr <= 200:
                    break
                print("❌ HR should be between 40-200 BPM")
            except ValueError:
                print("❌ Please enter a valid number")

        # Store measurement
        measurement = {
            'reference_hr': reference_hr,
            'app_hr': app_hr,
            'raw_hr': app_hr / self.current_calibration_factor,  # Reverse current calibration
            'difference': app_hr - reference_hr,
            'timestamp': datetime.now().isoformat()
        }
        self.measurements.append(measurement)

        print(f"\n✅ Measurement recorded:")
        print(f"   Watch: {reference_hr:.0f} BPM")
        print(f"   App: {app_hr:.0f} BPM")
        print(f"   Difference: {measurement['difference']:+.0f} BPM")

        return measurement

    def calculate_optimal_calibration(self):
        """Calculate optimal calibration factor from measurements"""
        if not self.measurements:
            print("❌ No measurements available")
            return None

        print("\n" + "="*60)
        print("📈 Calibration Analysis")
        print("="*60)

        # Extract data
        reference_hrs = [m['reference_hr'] for m in self.measurements]
        raw_hrs = [m['raw_hr'] for m in self.measurements]

        # Calculate optimal factor
        # We want: raw_hr * factor = reference_hr
        # So: factor = reference_hr / raw_hr
        factors = []
        for ref, raw in zip(reference_hrs, raw_hrs):
            if raw > 0:
                factors.append(ref / raw)

        if not factors:
            print("❌ Cannot calculate calibration")
            return None

        # Use median for robustness
        optimal_factor = np.median(factors)

        print(f"\nMeasurements taken: {len(self.measurements)}")
        print(f"Current calibration factor: {self.current_calibration_factor:.3f}")
        print(f"Recommended calibration factor: {optimal_factor:.3f}")

        # Show predicted accuracy
        print("\n📊 Predicted accuracy with new calibration:")
        for m in self.measurements:
            predicted_hr = m['raw_hr'] * optimal_factor
            new_diff = predicted_hr - m['reference_hr']
            old_diff = m['app_hr'] - m['reference_hr']
            print(f"   Watch: {m['reference_hr']:.0f} → "
                  f"Old diff: {old_diff:+.0f}, "
                  f"New diff: {new_diff:+.1f}")

        # Calculate average improvement
        old_errors = [abs(m['difference']) for m in self.measurements]
        new_errors = [abs(m['raw_hr'] * optimal_factor - m['reference_hr'])
                      for m in self.measurements]

        avg_old_error = np.mean(old_errors)
        avg_new_error = np.mean(new_errors)

        print(f"\n📉 Average error:")
        print(f"   Current: {avg_old_error:.1f} BPM")
        print(f"   With new calibration: {avg_new_error:.1f} BPM")

        if avg_new_error < avg_old_error:
            print(f"   ✅ Improvement: {(avg_old_error - avg_new_error):.1f} BPM better!")

        return optimal_factor

    def save_calibration(self, factor):
        """Save calibration factor to config file"""
        config = {
            'calibration_factor': factor,
            'calibrated_at': datetime.now().isoformat(),
            'measurements_used': len(self.measurements),
            'measurement_data': self.measurements
        }

        with open('hr_calibration.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n💾 Calibration saved to hr_calibration.json")
        print(f"   Factor: {factor:.3f}")

    def update_source_code(self, factor):
        """Show how to update the source code"""
        print("\n" + "="*60)
        print("🔧 To apply this calibration, update:")
        print("="*60)
        print("\nFile: core/ppg_signal_enhancer.py")
        print(f"Line ~363: CALIBRATION_FACTOR = {factor:.3f}")
        print("\nThen restart the app for changes to take effect.")

    def run_interactive_calibration(self):
        """Run interactive calibration session"""
        print("\n" + "🎯 "*20)
        print("HEART RATE CALIBRATION TOOL")
        print("🎯 "*20)
        print("\nThis tool will help calibrate the app's HR detection")
        print("to match your Apple Watch readings.")

        print("\n📋 Instructions:")
        print("1. Have your Apple Watch ready")
        print("2. Open the web app at http://localhost:5001")
        print("3. Take at least 3 measurements at different times")
        print("4. For best results, vary your heart rate (rest, after walking, etc.)")

        while True:
            print("\n" + "-"*60)
            print("Options:")
            print("1. Take a measurement")
            print("2. Calculate optimal calibration")
            print("3. Save and apply calibration")
            print("4. Clear all measurements")
            print("5. Exit")

            choice = input("\nChoice (1-5): ").strip()

            if choice == '1':
                self.take_measurement()

            elif choice == '2':
                factor = self.calculate_optimal_calibration()
                if factor and len(self.measurements) >= 3:
                    print("\n✅ Calibration ready! Choose option 3 to apply.")
                elif factor:
                    print(f"\n⚠️  Consider taking {3 - len(self.measurements)} more measurements for better accuracy.")

            elif choice == '3':
                if len(self.measurements) < 2:
                    print("❌ Need at least 2 measurements to calibrate")
                else:
                    factor = self.calculate_optimal_calibration()
                    if factor:
                        self.save_calibration(factor)
                        self.update_source_code(factor)

            elif choice == '4':
                self.measurements = []
                print("✅ All measurements cleared")

            elif choice == '5':
                print("\n👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice")


def main():
    """Main entry point"""
    tool = HRCalibrationTool()

    # Check if app is running
    try:
        response = requests.get("http://localhost:5001")
        if response.status_code != 200:
            print("⚠️  Warning: App might not be running properly")
    except:
        print("❌ Cannot connect to app at http://localhost:5001")
        print("   Please ensure the app is running first.")
        print("   Run: PORT=5001 python app.py")
        return

    tool.run_interactive_calibration()


if __name__ == "__main__":
    main()