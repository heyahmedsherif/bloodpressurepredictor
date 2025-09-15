# Claude AI Assistant - Project Context and History

## Project Overview
This is a Flask-based health prediction application that uses camera-based PPG (photoplethysmography) to predict health metrics including blood pressure, glucose, and cholesterol levels. The app captures video from the user's camera, extracts PPG signals from facial video, and uses ML models to predict health metrics.

## Current Branch Status
- **Current Branch**: `camera-testing` 
- **Main Branch**: `main`
- **Repository**: https://github.com/heyahmedsherif/bloodpressurepredictor

## Key Components

### 1. Core Application Structure
- `app.py` - Main Flask application
- `static/js/app.js` - Frontend JavaScript for video capture and processing
- `core/ml_health_predictor.py` - ML model integration for health predictions
- `core/ppg_feature_extractor.py` - PPG signal feature extraction
- `core/rppg_integration.py` - Integration with webcam-pulse-detector

### 2. ML Models (in `models/` directory)
- **Glucose Prediction**: Uses PPG amplitude, heart rate, systolic, diastolic, age, BMI
- **Cholesterol Prediction**: Uses age, heart rate, systolic, diastolic, BMI, PPG variability  
- **Blood Pressure**: Uses pulse transit time, PPG amplitude, heart rate, PPG width, age, BMI

All models are real scikit-learn models (LinearRegression with polynomial features and scaling) trained on PPG data.

## Major Issues Fixed

### 1. Video Streaming Issue (Snapshot Problem)
**Problem**: Video was showing single snapshots instead of continuous streaming
**Root Cause**: JavaScript was hiding the live video element and showing static canvas during recording
**Fix Applied**: Modified `static/js/app.js` to keep video element visible during recording
```javascript
// Changed from hiding video to keeping it visible
this.video.style.display = 'block';  // Was 'none'
this.canvas.style.display = 'none';   // Was 'block'
```

### 2. PPG Signal Flat Line Issue  
**Problem**: PPG signal was showing as flat line instead of periodic waves
**Root Cause**: Duplicate frames being sent due to static video capture
**Fix Applied**: 
- Ensured continuous video streaming
- Added frame timestamps to prevent duplicates
- Simplified frame capture logic in `processFrame()`

### 3. Docker Container Frame Reset Issue
**Problem**: Frame counter kept resetting (0/75, 0/75...) instead of counting up
**Root Cause**: Multiple gunicorn workers causing stateless request handling
**Fix Applied**: Modified `Dockerfile.flask` to use single worker
```dockerfile
# Changed from 2 workers to 1 worker with threads
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "180", "app:app"]
```

### 4. Railway Deployment Issues
**Problems Encountered**:
- Aggressive connection timeouts
- WebRTC/WebSocket limitations  
- Proxy layer interference with video streams
**Status**: Railway deployment proved unsuitable for real-time video streaming apps

## Docker Configuration

### Building the Container
```bash
docker build -t ppg-health-app:latest -f Dockerfile.flask .
```

### Running the Container
```bash
docker run -d --name ppg-health-container -p 8888:5000 ppg-health-app:latest
```

### Key Docker Settings
- Single worker for stateful video processing
- Port 5000 internal, mapped to 8888 external
- 180 second timeout for processing
- 4 threads for concurrent connections

## Deployment Options Evaluated

### Recommended for Camera/WebRTC Apps:
1. **Fly.io** - Excellent WebRTC support, free tier available
2. **Render** - Good WebSocket support, auto-SSL
3. **DigitalOcean/Linode** - Full control VPS options

### ARM Architecture Support:
- Oracle Cloud (Free tier with 4 ARM cores)
- AWS Graviton instances
- Fly.io (auto-detects architecture)

## Running Locally

### Development Mode
```bash
python app.py
```
Access at: http://localhost:5000

### Production Mode (Gunicorn)
```bash
gunicorn --bind 0.0.0.0:5000 --workers 1 --threads 4 --timeout 180 app:app
```

## Important Technical Details

### Camera Requirements
- HTTPS required for browser camera access
- WebSocket/WebRTC support needed for streaming
- Persistent connections required (no aggressive timeouts)

### PPG Processing Flow
1. Capture 75 frames from camera (3 seconds at 25 FPS)
2. Extract green channel from facial region
3. Process with webcam-pulse-detector for PPG signal
4. Extract features (amplitude, heart rate, variability, etc.)
5. Feed features to ML models for predictions

### State Management
- Video frame collection requires stateful processing
- Must use single worker in production to maintain state
- Frame counter and video data stored in memory during capture

## Testing Notes

### Verified Functionality
- ✅ Continuous video streaming (not snapshots)
- ✅ PPG signal shows proper waves (60-100 BPM)
- ✅ ML models using real algorithms (not hard-coded)
- ✅ Docker container maintains state properly
- ✅ Frame counter increments correctly (0-75)

### Known Limitations
- Requires good lighting for PPG extraction
- User must remain still during recording
- Processing can take 10-20 seconds after recording

## File Modifications History

### Key Files Modified
1. `static/js/app.js` - Fixed video streaming and frame capture
2. `Dockerfile.flask` - Changed to single worker configuration
3. `app.py` - Various fixes for PPG processing integration

### Git Commands for Recovery
```bash
# Access working version before Railway changes
git checkout 2fd09b1

# Current camera-testing branch
git checkout camera-testing
```

## Environment Setup

### Python Dependencies
- Flask with Werkzeug
- OpenCV (headless version for Docker)
- NumPy, SciPy, Pandas
- Scikit-learn for ML models
- Pillow for image processing
- Gunicorn for production serving

### System Dependencies (for Docker)
- libglib2.0-0
- libgomp1
- Other OpenCV dependencies

## Debugging Commands

### Check Docker Container
```bash
docker ps -a
docker logs ppg-health-container
docker exec -it ppg-health-container bash
```

### Monitor Background Processes
```bash
ps aux | grep python
lsof -i :5000
```

## DigitalOcean Deployment (NEW - September 15, 2025)

### Server Setup Completed
- Created DigitalOcean droplet with Ubuntu 22.04 LTS x64
- Configured VS Code Remote SSH for development
- Installed Miniconda and Docker on server
- Set up development environment in `/opt/apps/`

### Deployment Files Created
1. **digitalocean-deployment.md** - Complete deployment guide
2. **deploy-to-digitalocean.sh** - Automated deployment script
3. **server-setup-commands.txt** - Quick reference commands

### VS Code Remote SSH Configuration
```
Host do-ppg
    HostName YOUR_DROPLET_IP
    User root
    Port 22
    PasswordAuthentication yes
```

### Miniconda Installation on Server
```bash
# Installed at /root/miniconda3
conda create -n ppg-app python=3.10 -y
conda activate ppg-app
```

### Claude Code Installation on Server
```bash
# Installed at ~/.local/bin/claude
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

## Additional Context from Today's Session

### Questions Answered About Frame Processing
- **Frame Processing**: Processes frames incrementally as they arrive, not all at once
- **Minimum Frames**: 75 frames required (5 seconds at 15 FPS)
- **Expected FPS**: ~15 FPS (captures every ~67ms)
- **Processing Window**: 5-second recording captures 4-5 heartbeats

### Unused Imports Identified
- `PPGProcessor` and `SimplePPGProcessor` - Can be removed (dead code)
- `MLHealthPredictor` - Must keep (actively used for predictions)

### Cloud Provider Analysis
- **x64/AMD64 Support**: All major providers (DigitalOcean, Vultr, Hetzner, etc.)
- **ARM Support**: Oracle Cloud (free tier), AWS Graviton, Hetzner CAX
- **Best Value**: Hetzner Cloud (€3.79/month)
- **Easiest Setup**: DigitalOcean ($6/month)
- **Free Option**: Oracle Cloud (always free tier)

### Current Deployment Status
- Repository pushed to `camera-testing` branch
- Docker container tested and working locally
- DigitalOcean server provisioned and configured
- Ready for production deployment

## Next Steps and TODOs

1. Complete DigitalOcean deployment with Docker container
2. Set up Nginx reverse proxy on server
3. Configure SSL certificate for HTTPS
4. Consider implementing proper session management for multi-user support
5. Add WebSocket for real-time frame upload progress
6. Implement proper error handling for camera access denial
7. Consider adding Redis for state management in multi-worker scenarios
8. Optimize PPG processing performance

## Contact and Repository Info

- GitHub: https://github.com/heyahmedsherif/bloodpressurepredictor
- Branch for testing: `camera-testing`
- Docker image: `ppg-health-app:latest`
- Deployment Scripts: `deploy-to-digitalocean.sh`, `digitalocean-deployment.md`

---

*Last Updated: September 15, 2025*
*This file should be reviewed at the start of each Claude session for context*