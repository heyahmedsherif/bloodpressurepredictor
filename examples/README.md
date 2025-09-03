# Examples Directory

This directory contains example data, scripts, and usage demonstrations for the cardiovascular risk predictor.

## Files

### `sample_ppg_data.py`
Python script to generate realistic PPG signals for testing the application.

**Features:**
- Multiple physiological conditions (normal, hypertensive, elderly, athlete)
- Configurable duration and sampling rate
- Realistic PPG morphology with artifacts and noise
- CSV output format compatible with Streamlit app
- Optional visualization of generated signals

**Usage:**
```bash
# Generate default sample data
python examples/sample_ppg_data.py

# Generate 2-minute signals with plot
python examples/sample_ppg_data.py --duration 120 --plot

# Custom sampling rate and output directory
python examples/sample_ppg_data.py --fs 500 --output_dir data/samples
```

**Generated Files:**
- `normal_ppg.csv`: Healthy adult PPG signal (HR: 70 bpm, low risk)
- `hypertensive_ppg.csv`: Hypertensive patient (HR: 85 bpm, high risk)
- `elderly_ppg.csv`: Elderly patient (HR: 65 bpm, moderate risk)
- `athlete_ppg.csv`: Young athlete (HR: 50 bpm, very low risk)

### Usage with Streamlit App

1. Generate sample data:
   ```bash
   python examples/sample_ppg_data.py --plot
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run apps/streamlit_app_robust.py
   ```

3. Upload generated CSV files using the "Upload PPG File" option

4. Select the `ppg` column and set appropriate sampling rate (default: 250 Hz)

## PPG Signal Characteristics

### Normal Adult
- Heart rate: 70 bpm
- Clear dicrotic notch
- Regular rhythm
- Low noise level
- Expected BP: ~120/80 mmHg

### Hypertensive Patient
- Heart rate: 85 bpm
- Reduced dicrotic notch (arterial stiffness)
- Higher baseline variation
- Occasional artifacts
- Expected BP: >140/90 mmHg

### Elderly Patient
- Heart rate: 65 bpm
- Irregular rhythm with HRV
- Altered pulse wave morphology
- Reduced amplitude variation
- Expected BP: Elevated systolic

### Young Athlete
- Heart rate: 50 bpm (bradycardia)
- Strong pulse amplitude
- Prominent dicrotic notch
- Clear respiratory variation
- Expected BP: Optimal range

## Testing Scenarios

Use these datasets to validate:
- PPG preprocessing pipeline
- Blood pressure prediction accuracy
- Cardiovascular risk scoring
- Early warning system functionality
- System robustness with various signal qualities