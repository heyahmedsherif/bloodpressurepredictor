# PPG-BP Database Setup Guide

## Overview
This project uses the PPG-BP Database v1.0.0 for training machine learning models to predict blood pressure and other health metrics from PPG (photoplethysmography) signals. The database contains 657 PPG recordings from 219 subjects with corresponding blood pressure measurements.

## Dataset Information
- **Name**: PPG-BP Database v1.0.0
- **Source**: Figshare (https://figshare.com/articles/dataset/PPG-BP_Database_zip/5459299)
- **DOI**: 10.6084/m9.figshare.5459299
- **Size**: ~40 MB compressed
- **Records**: 657 PPG recordings from 219 subjects
- **Contents**:
  - PPG signals
  - Systolic and diastolic blood pressure measurements
  - Subject metadata (age, gender, etc.)

## Citation
If you use this dataset, please cite:
```
Liang, Yongbo; Chen, Zhencheng; Ward, Rabab; Elgendi, Mohamed (2018):
PPG-BP Database. figshare. Dataset.
https://doi.org/10.6084/m9.figshare.5459299.v1
```

## Setup Instructions

### 1. Download the Dataset

```bash
# Create datasets directory
mkdir -p datasets
cd datasets

# Download from Figshare
wget https://figshare.com/ndownloader/files/24916291 -O PPG-BP_Database.zip
```

### 2. Extract the Dataset

```bash
# Extract the archive
unzip PPG-BP_Database.zip

# This creates the following structure:
# datasets/
#   └── PPG_BP_Database-1.0.0/
#       ├── 0_subject/
#       ├── 1_subject/
#       ├── ...
#       └── 218_subject/
```

### 3. Verify the Structure

Each subject folder contains:
- PPG signal files (`.txt` or `.mat` format)
- Blood pressure readings
- Subject information

```bash
# Check the structure
ls PPG_BP_Database-1.0.0/
# Should show folders from 0_subject to 218_subject
```

## Data Format

### PPG Signal Files
- Sampling rate: 1000 Hz
- Duration: Variable (typically 2-3 minutes)
- Format: Text files with single column of PPG values

### Blood Pressure Data
- Systolic BP (SBP): mmHg
- Diastolic BP (DBP): mmHg
- Mean Arterial Pressure (MAP): Calculated as DBP + (SBP - DBP)/3

### Subject Information
- Age: Years
- Gender: M/F
- Height: cm
- Weight: kg
- BMI: Calculated from height and weight

## Integration with Training Scripts

The training scripts expect the data in the following structure:
```
datasets/
├── PPG_BP_Database-1.0.0/
│   ├── 0_subject/
│   ├── 1_subject/
│   └── ...
└── processed/  # Created by preprocessing scripts
    ├── features.csv
    └── labels.csv
```

## Preprocessing Pipeline

1. **Signal Quality Assessment**: Remove recordings with poor signal quality
2. **Feature Extraction**: Extract PPG features (amplitude, heart rate, variability)
3. **Data Normalization**: Standardize features for ML models
4. **Train/Test Split**: 80/20 split with stratification

## Training Models

Once the dataset is set up, train the models using:

```bash
# Train all models with real data
python train_with_real_data.py

# Train specific model
python train_with_real_data.py --model bp  # Blood pressure only
python train_with_real_data.py --model glucose  # Glucose only
python train_with_real_data.py --model cholesterol  # Cholesterol only
```

## Important Notes

1. **Storage**: The dataset is NOT included in the Git repository due to size and licensing
2. **Privacy**: Subject data is anonymized but should be handled responsibly
3. **License**: Check Figshare page for specific usage terms
4. **Gitignore**: The `datasets/` folder is excluded from version control

## Troubleshooting

### Dataset Not Found
If scripts can't find the dataset:
```bash
# Check if dataset exists
ls datasets/PPG_BP_Database-1.0.0/

# If missing, follow download instructions above
```

### Memory Issues
For large batch processing:
```bash
# Process in smaller batches
python train_with_real_data.py --batch-size 32
```

### Signal Quality Issues
Some recordings may have poor quality. The preprocessing scripts automatically filter these out based on:
- Signal-to-noise ratio
- Presence of motion artifacts
- Completeness of recording

## Additional Resources

- Original Paper: [Link to paper describing the dataset]
- PPG Processing Guide: See `core/ppg_feature_extractor.py`
- Model Architecture: See `models/README.md`

---

*Last updated: September 2025*