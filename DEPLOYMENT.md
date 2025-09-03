# 🚀 Streamlit Cloud Deployment Guide

## 📁 **New Organized Folder Structure**

```
bloodpressurepredictor/
├── streamlit_app.py           # 🎯 MAIN ENTRY POINT for Streamlit Cloud
├── requirements.txt           # 📦 Dependencies (required by Streamlit Cloud)
├── README.md                  # 📖 Main documentation
├── DEPLOYMENT.md             # 🚀 This deployment guide
│
├── src/                      # 📂 Source code (organized)
│   ├── apps/                 # 🖥️ Streamlit applications
│   │   ├── realistic_bp_predictor.py    # Main BP predictor (recommended)
│   │   ├── bp_predictor.py              # Extended BP predictor  
│   │   └── streamlit_app_robust.py      # PPG signal processor
│   │
│   ├── core/                 # 🧠 PaPaGei core components
│   │   ├── models/           # AI model architectures
│   │   ├── preprocessing/    # Signal processing
│   │   ├── linearprobing/    # Feature extraction
│   │   └── utilities.py      # Core utilities
│   │
│   └── utils/                # 🛠️ Helper utilities
│
├── weights/                  # 🧬 Model weights
│   └── papagei_s.pt         # PaPaGei foundation model (23.3MB)
│
├── docs/                     # 📚 Documentation
├── examples/                 # 📋 Usage examples
├── scripts/                  # 🔧 Validation scripts
└── config/                   # ⚙️ Configuration files
```

## 🎯 **Streamlit Cloud Deployment**

### **1. Quick Deploy**
1. **Fork/Clone** this repository
2. **Connect to Streamlit Cloud**: https://share.streamlit.io
3. **Select Repository**: `your-username/bloodpressurepredictor`
4. **Main File Path**: `streamlit_app.py` (auto-detected)
5. **Deploy** ✅

### **2. Requirements**
- **Automatic Detection**: Streamlit Cloud automatically finds `requirements.txt` in root
- **Dependencies**: All ML, signal processing, and visualization libraries included
- **Model Weights**: PaPaGei weights (23.3MB) included in repository

### **3. App Selection**
The main app (`streamlit_app.py`) provides three options:
- **Realistic BP Predictor** ⭐ (Recommended for production)
- **Extended BP Predictor** (Research/testing)  
- **PPG Signal Processor** (Signal analysis focus)

## 📊 **App Features**

### **🩺 Realistic BP Predictor**
- **Clinically Deployable**: Uses only available features
- **Real PaPaGei Integration**: Actual foundation model weights
- **Visual PPG Generation**: Shows expected PPG patterns
- **Accuracy**: ±10.0 mmHg systolic, ±6.4 mmHg diastolic

### **🔬 Extended BP Predictor**  
- **Research Mode**: 16+ lifestyle and clinical features
- **Higher Accuracy**: ±8.5 mmHg (but less practical)
- **Comprehensive Analysis**: Full feature importance

### **📈 PPG Signal Processor**
- **Signal Focus**: PPG analysis and visualization
- **Real-time Processing**: Upload and analyze PPG files
- **Educational**: Learn about PPG signal characteristics

## 🛠️ **Local Development**

### **Setup**
```bash
git clone git@github.com:your-username/bloodpressurepredictor.git
cd bloodpressurepredictor
pip install -r requirements.txt
```

### **Run Main App**
```bash
streamlit run streamlit_app.py
```

### **Run Individual Apps**
```bash
# Realistic BP Predictor (recommended)
streamlit run src/apps/realistic_bp_predictor.py --server.port=8503

# Extended BP Predictor
streamlit run src/apps/bp_predictor.py --server.port=8502

# PPG Signal Processor  
streamlit run src/apps/streamlit_app_robust.py --server.port=8501
```

## ⚠️ **Troubleshooting**

### **Import Issues**
The new structure uses relative imports. If you encounter import errors:
```python
# All apps automatically handle path resolution for deployment
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
```

### **Model Loading**
- **PaPaGei weights**: Automatically loaded from `weights/papagei_s.pt`
- **CPU Deployment**: Optimized for CPU-only environments (Streamlit Cloud)
- **Fallback Mode**: Apps gracefully degrade if weights unavailable

## 📱 **Mobile Optimization**
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Touch-Friendly**: Optimized for touch interfaces
- **Fast Loading**: Efficient model loading and caching

## 🔒 **Security & Privacy**
- **No Data Storage**: All processing happens in user session
- **Session Isolation**: Each user gets independent session state
- **Privacy-First**: No personal health data retention

## 📈 **Performance**
- **Model Caching**: PaPaGei model cached after first load
- **Efficient Processing**: Optimized for Streamlit Cloud resources
- **Fast Predictions**: <2 seconds for BP prediction

---

**🎯 Ready for deployment!** The main entry point `streamlit_app.py` is optimized for Streamlit Cloud's requirements and folder structure.