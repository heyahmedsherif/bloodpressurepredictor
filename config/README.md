# Configuration Directory

This directory contains configuration files and requirements for the cardiovascular risk predictor project.

## Files

### `requirements_streamlit.txt`
Required packages for running the Streamlit cardiovascular risk predictor application.

**Installation:**
```bash
pip install -r config/requirements_streamlit.txt
```

### Core Dependencies
- `streamlit`: Web application framework
- `plotly`: Interactive visualizations
- `torch`: PyTorch deep learning framework
- `pyPPG`: PPG signal processing library
- `torch-ecg`: ECG/PPG preprocessing utilities

### Model Configuration
The PaPaGei-S model configuration is embedded in the application:
- Base filters: 32
- Kernel size: 3
- Stride: 2
- N blocks: 18
- N classes: 512 (embedding dimension)
- N experts: 3 (MoE architecture)

## Environment Setup

1. Create conda environment:
```bash
conda create -n papagei_cv python=3.10
conda activate papagei_cv
```

2. Install requirements:
```bash
pip install -r config/requirements_streamlit.txt
```

3. Download model weights (optional):
Place `papagei_s.pt` in the `weights/` directory.
Download from: https://zenodo.org/records/13983110