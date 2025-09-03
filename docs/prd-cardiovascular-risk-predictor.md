# Product Requirements Document: Cardiovascular Risk Predictor

## 1. Introduction/Overview

The Cardiovascular Risk Predictor is an early warning system that leverages the PaPaGei foundation model to predict blood pressure and cardiovascular disease risk from PPG signals. This system aims to provide researchers and clinicians with a powerful tool for early detection of cardiovascular events, potentially saving lives through timely intervention.

The product addresses the critical need for non-invasive, continuous cardiovascular monitoring that can identify at-risk individuals before clinical symptoms manifest. By building on PaPaGei's robust PPG signal processing capabilities, we can transform consumer-grade wearable data into clinically meaningful cardiovascular risk assessments.

**Goal:** Create a research-grade web platform that provides early cardiovascular event warnings through PPG-based blood pressure monitoring and cardiovascular disease risk scoring.

## 2. Goals

1. **Primary Goal**: Develop a reliable early warning system that can predict cardiovascular events 24-48 hours before they occur with variable accuracy depending on risk level and signal quality
2. **Secondary Goal**: Enable large-scale cardiovascular research by providing researchers with advanced PPG analysis tools and standardized risk metrics
3. **Tertiary Goal**: Create a foundation for future clinical deployment by establishing robust data pipelines and validation frameworks

## 3. User Stories

### Primary User Stories
- **As a cardiovascular researcher**, I want to upload PPG datasets and receive comprehensive cardiovascular risk analysis so that I can identify patterns in population-level cardiovascular health
- **As a clinical researcher**, I want to process real-time PPG streams from study participants so that I can monitor cardiovascular changes during interventions
- **As a data scientist**, I want to access blood pressure predictions with confidence intervals so that I can validate the model's performance against ground truth measurements

### Secondary User Stories
- **As a research team lead**, I want to manage multiple studies and datasets through a centralized dashboard so that I can oversee various cardiovascular research projects
- **As a biomedical engineer**, I want to integrate the system with existing research infrastructure so that I can incorporate cardiovascular risk prediction into larger health monitoring studies

## 4. Functional Requirements

### Core PPG Processing Requirements
1. The system must accept PPG input data in multiple formats: real-time streams, batch uploads of recorded files, API integration with wearable devices, and standard research file formats (CSV, HDF5, etc.)
2. The system must preprocess PPG signals using PaPaGei's existing preprocessing pipeline (bandpass filtering, segmentation, resampling to 125Hz)
3. The system must extract embeddings using the pre-trained PaPaGei-S model with 512-dimensional feature vectors
4. The system must handle various PPG sampling rates (125Hz to 1000Hz) and automatically resample to PaPaGei's expected 125Hz

### Blood Pressure Prediction Requirements
5. The system must predict both systolic and diastolic blood pressure values from PPG embeddings
6. The system must provide confidence intervals (95%) for all blood pressure predictions using bootstrapping methods
7. The system must flag predictions with low confidence scores (below configurable threshold) as unreliable
8. The system must support both single-point predictions and trend analysis over time

### Cardiovascular Risk Assessment Requirements
9. The system must calculate cardiovascular disease risk scores based on PPG-derived features and demographic inputs
10. The system must integrate multiple risk factors: predicted blood pressure, heart rate variability, signal quality metrics, and morphological features (sVRI, IPA, SQI)
11. The system must provide risk stratification (low, moderate, high, critical) with accompanying explanations
12. The system must generate early warning alerts when cardiovascular risk exceeds configurable thresholds

### Web Dashboard Requirements
13. The system must provide a responsive web interface accessible from desktop browsers
14. The system must support user authentication and role-based access control for research team management
15. The system must display real-time cardiovascular metrics through interactive charts and visualizations
16. The system must allow users to configure alert thresholds and notification preferences
17. The system must provide data export functionality (CSV, JSON, PDF reports) for research publication
18. The system must maintain a history of all predictions and risk assessments for longitudinal analysis

### Data Management Requirements
19. The system must store all PPG data, predictions, and metadata in a secure, HIPAA-compliant database
20. The system must support data versioning and audit trails for research reproducibility
21. The system must implement data retention policies with automated archiving capabilities
22. The system must provide APIs for programmatic access to predictions and historical data

## 5. Non-Goals (Out of Scope)

1. **Clinical Decision Support**: This system is not intended for direct clinical decision-making without physician oversight
2. **Consumer Mobile App**: No mobile application development - web dashboard only for this version
3. **Real-time Treatment Recommendations**: The system will not provide specific treatment or medication recommendations
4. **Integration with Electronic Health Records**: EHR integration is out of scope for the initial release
5. **FDA Approval Process**: Regulatory approval activities are not included in this development scope
6. **Multi-language Support**: English-only interface for the research version
7. **Patient-facing Features**: No direct patient access or patient-friendly interfaces

## 6. Design Considerations

### Architecture
- **Backend**: Build on PaPaGei's existing Python codebase, extending `feature_extraction.py` and `models/resnet.py`
- **Frontend**: Modern web dashboard using React/Vue.js with real-time data visualization libraries (D3.js, Chart.js)
- **Database**: Time-series database (InfluxDB) for PPG data storage with PostgreSQL for metadata and user management
- **APIs**: RESTful APIs for data ingestion and GraphQL for complex data queries

### User Interface Guidelines
- Clean, research-focused interface with emphasis on data visualization
- Dark mode support for extended research sessions
- Accessible design following WCAG 2.1 guidelines
- Responsive design optimized for large monitors and research workstations

### Data Visualization Requirements
- Real-time PPG signal plots with overlaid predictions
- Cardiovascular risk trend charts with confidence intervals
- Comparative analysis tools for multi-subject studies
- Customizable dashboards for different research workflows

## 7. Technical Considerations

### PaPaGei Integration
- Extend existing `feature_extraction.py` to include cardiovascular-specific feature computation
- Leverage pre-trained PaPaGei-S model weights (papagei_s.pt) for embedding extraction
- Utilize existing morphological feature computation (`morphology.py`) for enhanced risk assessment
- Build upon the segmentation pipeline (`segmentations.py`) for consistent data processing

### Performance Requirements
- Process PPG segments in real-time (< 2 seconds for 10-second segments)
- Support concurrent analysis of up to 100 research participants
- Handle datasets up to 1TB with efficient querying and visualization
- Maintain 99.9% uptime for critical research studies

### Security and Privacy
- End-to-end encryption for all PPG data transmission
- Role-based access control with researcher authentication
- Audit logging for all data access and predictions
- Data anonymization tools for research publication

### Scalability
- Containerized deployment using Docker for easy scaling
- Cloud-ready architecture supporting AWS/Azure deployment
- Horizontal scaling capabilities for large research studies
- Efficient caching mechanisms for frequently accessed predictions

## 8. Success Metrics

### Performance Metrics
- **Blood Pressure Accuracy**: Mean Absolute Error (MAE) < 5mmHg for systolic, < 3mmHg for diastolic blood pressure predictions
- **Risk Prediction Sensitivity**: Correctly identify 85% of high cardiovascular risk cases
- **System Performance**: 95% of predictions completed within 2-second latency requirement
- **Model Reliability**: Achieve 90% prediction confidence on high-quality PPG signals

### Research Impact Metrics
- **Adoption**: 50+ researchers actively using the platform within 6 months
- **Data Processing**: Successfully process 10,000+ hours of PPG data in the first year
- **Publication Support**: Enable 5+ peer-reviewed publications using the platform
- **Early Warning Effectiveness**: Demonstrate 70% accuracy in predicting cardiovascular events 24-48 hours in advance

### Technical Metrics
- **System Reliability**: 99.9% uptime with < 1-hour recovery time for critical failures
- **Data Integrity**: Zero data loss incidents with complete audit trail coverage
- **User Satisfaction**: 85% user satisfaction score from researcher feedback surveys
- **API Performance**: 99% of API requests completed within 500ms response time

## 9. Open Questions

### Technical Questions
1. **Model Calibration**: How should we calibrate the blood pressure prediction models for different populations and PPG device types?
2. **Signal Quality Thresholds**: What minimum signal quality scores should trigger low-confidence warnings?
3. **Temporal Windows**: What time windows (5min, 10min, 1hr) provide optimal prediction accuracy for cardiovascular events?

### Research Questions
4. **Validation Dataset**: Which external datasets should we use for independent validation of cardiovascular risk predictions?
5. **Demographic Factors**: How should age, gender, and ethnicity be incorporated into risk scoring algorithms?
6. **Device Compatibility**: What PPG devices and sampling rates should be prioritized for initial testing?

### Product Questions
7. **Alert Thresholds**: What cardiovascular risk score thresholds should trigger different alert levels?
8. **Data Retention**: What data retention policies align with research ethics and storage constraints?
9. **User Training**: What documentation and training materials are needed for researcher onboarding?

### Future Development
10. **Clinical Integration**: What additional features would be needed for eventual clinical deployment?
11. **Multi-modal Integration**: How could this system integrate with other physiological signals (ECG, activity data) in future versions?
12. **Real-time Processing**: What infrastructure changes would support true real-time processing for hundreds of concurrent users?