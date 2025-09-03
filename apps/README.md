# Applications Directory

This directory contains the main application implementations for the cardiovascular risk predictor.

## Files

### `streamlit_app_robust.py`
The main Streamlit application with comprehensive error handling and fallback mechanisms.

**Features:**
- Robust error tracking and reporting
- Fallback implementations for missing dependencies
- Safe import mechanisms with graceful degradation
- System status monitoring
- Complete cardiovascular risk prediction pipeline

**Usage:**
```bash
streamlit run apps/streamlit_app_robust.py
```

### `streamlit_app.py`
Original Streamlit application (basic version without comprehensive error handling).

### `run_app.py`
Test script to validate app components and dependencies before deployment.

**Usage:**
```bash
python apps/run_app.py
```

## Dependencies

See `config/requirements_streamlit.txt` for required packages.

## Error Handling

The robust version includes:
- Import validation with fallbacks
- Comprehensive error tracking
- System status monitoring
- Safe plotting and processing
- Graceful degradation when components fail