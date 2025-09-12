# PPG Health Prediction Suite

A Flask-based web application for non-invasive health monitoring using remote photoplethysmography (rPPG) from webcam video.

## Features

- Real-time heart rate detection using webcam
- ML-based health predictions (glucose, cholesterol, blood pressure)
- PPG signal visualization
- User-friendly web interface

## Quick Start

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate ML models:
```bash
python generate_ml_models.py
```

3. Run the application:
```bash
python app.py
```

4. Open browser to http://localhost:5001

### Railway Deployment

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

The app is configured for Railway deployment with:
- `Dockerfile` for containerization
- `railway.json` for Railway configuration
- Port binding to `$PORT` environment variable

## Architecture

- **Core Modules** (`core/`): rPPG processing, ML predictions, feature extraction
- **ML Models** (`models/`): Pre-trained models for health metrics
- **External** (`external/`): Forked research repositories
- **Static/Templates**: Web interface assets

## Research Foundation

Based on research from:
- webcam-pulse-detector (Tristan Hearn)
- rPPG signal processing algorithms
- ML models for health prediction from PPG signals

## Note

This is a research tool for demonstration purposes only. Not intended for medical diagnosis or treatment.