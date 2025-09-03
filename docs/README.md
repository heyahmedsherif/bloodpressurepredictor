# Documentation Directory

This directory contains project documentation, requirements specifications, and design documents.

## Files

### `prd-cardiovascular-risk-predictor.md`
**Product Requirements Document** for the cardiovascular risk predictor application.

Contains:
- Comprehensive feature specifications
- Technical requirements
- User stories and use cases
- Success metrics and validation criteria
- Integration with PaPaGei foundation model

### `context_log.md`
**Rolling Context Log** maintaining project history and decision tracking.

Contains:
- Session-by-session development progress
- Technical decisions and implementations
- Status updates and milestone tracking
- Future development roadmap

## Key Documents Summary

### Project Scope
- **Target Users**: Researchers conducting cardiovascular studies
- **Primary Use Case**: Early warning system for cardiovascular events
- **Core Technology**: PaPaGei foundation model for PPG signal processing
- **Prediction Targets**: Blood pressure estimation and cardiovascular risk scoring

### Technical Architecture
- **Frontend**: Streamlit web dashboard
- **Backend**: PaPaGei-S ResNet1D-MoE model
- **Processing Pipeline**: PPG preprocessing → Segmentation → Feature extraction → Prediction
- **Performance Target**: <2 second processing for 10-second PPG segments

### Implementation Status
- ✅ Complete Streamlit application with error handling
- ✅ PPG signal processing pipeline integrated
- ✅ Mock prediction models (ready for trained models)
- ✅ Interactive dashboard with risk assessment
- ✅ Research-grade data export functionality

### Next Steps
- Train actual BP and CV risk prediction models
- Validate on clinical datasets
- Optimize performance for large-scale studies
- Add real-time monitoring capabilities