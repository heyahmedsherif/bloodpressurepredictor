# PPG Health Prediction Suite

A Flask-based web application for non-invasive health monitoring using remote photoplethysmography (rPPG) from webcam video. This application uses **real PPG data** from clinical databases to make accurate health predictions.

## 🚀 Recent Updates (September 2025)

### Major Improvements
- ✅ **Real Data Integration**: Models now trained on 657 real PPG recordings from PPG-BP Database
- ✅ **Enhanced Signal Processing**: Implemented Hybrid Smart Processing with quality assessment
- ✅ **Improved Accuracy**: Blood pressure MAE reduced to 9.18/5.66 mmHg (systolic/diastolic)
- ✅ **Heart Rate Calibration**: Added Apple Watch calibration tool for accurate HR detection
- ✅ **Measurement Stabilization**: Averaging system to reduce variability between readings
- ✅ **LDL/HDL Breakdown**: Separate cholesterol component predictions
- ✅ **Docker Support**: Full containerization with optimized settings

## 📊 Model Performance

### Blood Pressure (Real Data)
- **Training Data**: 657 PPG recordings from 219 subjects
- **Systolic BP**: MAE = 9.18 mmHg, R² = 0.640
- **Diastolic BP**: MAE = 5.66 mmHg, R² = 0.521
- **Model**: GradientBoostingRegressor

### Other Metrics
- **Heart Rate**: Direct extraction from PPG peaks (calibrated)
- **Glucose**: MAE = 4.88 mg/dL (estimated labels)
- **Cholesterol**: MAE = 8.69 mg/dL with LDL/HDL breakdown
- **Vascular Age**: Based on arterial stiffness indicators

## 🎯 Features

- **Real-time PPG Extraction**: Camera-based pulse detection
- **Comprehensive Health Metrics**:
  - Blood Pressure (Systolic/Diastolic)
  - Heart Rate with variability analysis
  - Blood Glucose estimation
  - Total Cholesterol with LDL/HDL/VLDL breakdown
  - Vascular Age assessment
  - Cardiovascular risk scoring
- **Signal Quality Assessment**: Real-time feedback on measurement quality
- **Enhanced Mode**: Advanced signal processing for better accuracy
- **Measurement History**: Track and average multiple readings

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Webcam/camera access
- Good lighting conditions

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/heyahmedsherif/bloodpressurepredictor.git
cd papagei-foundation-model
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download and prepare real data (optional - models already included):**
```bash
# PPG-BP Database is already in datasets/Data File/
# To retrain models with real data:
python train_ppg_bp_real_data.py
```

4. **Run the application:**
```bash
# Development mode
python app.py

# Or with specific port
PORT=5001 python app.py
```

5. **Open browser:** http://localhost:5001

## 🐳 Docker Deployment

### Build and run with Docker:
```bash
# Build the container
docker build -t ppg-health-app:latest -f Dockerfile.flask .

# Run the container
docker run -d --name ppg-health-container -p 8888:5000 ppg-health-app:latest

# Access at http://localhost:8888
```

## 📁 Project Structure

```
papagei-foundation-model/
├── app.py                          # Main Flask application
├── core/                           # Core processing modules
│   ├── ppg_signal_enhancer.py     # Signal enhancement with quality assessment
│   ├── enhanced_ml_predictor.py   # ML prediction with real models
│   ├── measurement_stabilizer.py  # Averaging and outlier detection
│   ├── rppg_integration.py        # Camera PPG extraction
│   └── ppg_feature_extractor.py   # Feature engineering
├── models/                         # Trained ML models
│   ├── bp_model.pkl               # Blood pressure (real data)
│   ├── glucose_model.pkl          # Glucose prediction
│   ├── cholesterol_model.pkl      # Total cholesterol
│   └── cholesterol_detailed/      # LDL/HDL models
├── datasets/                       # Real PPG databases
│   ├── Data File/                 # PPG-BP Database (657 recordings)
│   └── ppg_cholesterol/           # Cholesterol dataset
├── static/                         # Frontend assets
│   ├── js/app.js                  # Video capture and processing
│   └── css/style.css             # UI styling
└── templates/                      # HTML templates
```

## 🔧 Configuration

### Heart Rate Calibration
To calibrate heart rate against your reference device (e.g., Apple Watch):
```bash
python calibrate_hr.py
```

### Signal Processing Modes
- **Standard Mode**: Basic PPG extraction
- **Enhanced Mode** (Default): Advanced signal processing with quality assessment

### Recording Settings
- **Duration**: 10 seconds (configurable)
- **Frame Rate**: 30 FPS
- **Buffer Size**: 450 frames

## 📈 Training with Real Data

The models are trained on real PPG data from clinical databases:

### PPG-BP Database
- **Source**: Figshare (DOI: 10.6084/m9.figshare.5459299)
- **Subjects**: 219 individuals
- **Recordings**: 657 PPG signals
- **Includes**: Blood pressure, heart rate, demographics

### To retrain models:
```bash
# Train with PPG-BP database
python train_ppg_bp_real_data.py

# Train LDL/HDL models
python train_ldl_hdl_models.py
```

## 🎯 Usage Tips

### For Best Results:
1. **Lighting**: Ensure good, consistent lighting on your face
2. **Position**: Keep face centered and still during recording
3. **Distance**: Maintain 30-50cm from camera
4. **Stability**: Rest arms on a surface to minimize movement
5. **Multiple Readings**: Take 3-5 measurements and use the average

### Troubleshooting:
- **Poor Signal Quality**: Check lighting, reduce movement
- **Inconsistent Readings**: Use measurement stabilizer (takes 3+ readings)
- **High HR Readings**: Run calibration tool against reference device

## 🔬 Technical Details

### Signal Processing Pipeline
1. **Video Capture**: 30 FPS RGB video from webcam
2. **Face Detection**: MediaPipe or Haar Cascade
3. **ROI Extraction**: Forehead region for PPG signal
4. **Signal Enhancement**:
   - Bandpass filtering (0.5-4 Hz)
   - Adaptive filtering based on SNR
   - Motion artifact removal
5. **Feature Extraction**: Time and frequency domain features
6. **ML Prediction**: Ensemble models trained on real data

### Quality Metrics
- Signal-to-Noise Ratio (SNR)
- Peak consistency score
- Amplitude stability
- Spectral purity
- Motion artifact detection

## 🚢 Deployment Options

### Recommended Platforms:
- **Local Development**: Best performance and privacy
- **Docker**: Containerized deployment
- **DigitalOcean**: VPS with full control
- **Fly.io**: Good WebRTC support
- **Render**: Auto-SSL, WebSocket support

### Not Recommended:
- **Railway**: Connection timeout issues with video streaming
- **Heroku**: Limited WebRTC support

## 📚 References

### Datasets Used:
1. **PPG-BP Database**: Liang et al. (2018) - [Figshare](https://doi.org/10.6084/m9.figshare.5459299)
2. **PPG Cholesterol Dataset**: Internal dataset with 10 subjects

### Research Papers:
- Remote PPG techniques for vital sign monitoring
- Correlation between PPG morphology and cardiovascular parameters
- Machine learning approaches for health prediction from PPG

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This application is for educational and research purposes only. It is not intended as a medical device or for diagnostic purposes. Always consult with healthcare professionals for medical advice.

## 📞 Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/heyahmedsherif/bloodpressurepredictor/issues)
- Documentation: See `CLAUDE.md` for detailed development notes

---

*Last Updated: September 2025*
*Version: 2.0.0 - Real Data Integration*