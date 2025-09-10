# Railway Deployment Guide

## PaPaGei Blood Pressure Predictor - Camera-Based Health App

### Quick Deploy to Railway

1. **Fork/Clone this repository**

2. **Go to [Railway.app](https://railway.app)**
   - Sign in with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select this repository

3. **Railway will automatically:**
   - Detect the Dockerfile
   - Build the container with all dependencies
   - Deploy the app with camera support

4. **Access your app:**
   - Railway will provide a public URL
   - App runs on port 8080 internally
   - Full camera functionality available

### Features

✅ **Simplified Interface**
- Single recording button with live preview
- Real-time camera feedback during recording
- Automatic PPG signal extraction

✅ **Health Predictions**
- Blood Pressure estimation
- Heart rate detection  
- Blood glucose prediction
- Cholesterol level prediction

✅ **Railway Optimized**
- Full OpenCV camera support
- System-level dependencies included
- Optimized Docker container
- Auto-scaling ready

### Local Testing

```bash
# Test locally before deployment
conda activate bloodpressure
streamlit run streamlit_app.py
```

### Environment Variables

No additional environment variables required - everything is configured in the Dockerfile.

### Camera Requirements

- **Local**: Works with any USB camera
- **Production**: Railway supports camera access through browser
- **Browser permissions**: Users need to grant camera access

### Troubleshooting

- **Camera not working**: Check browser permissions
- **Slow startup**: Railway cold starts take ~30 seconds
- **Memory issues**: Railway free tier has memory limits

### Cost Estimate

Railway pricing (as of 2024):
- **Free tier**: $5/month credit (sufficient for testing)
- **Usage-based**: ~$0.10/hour when active
- **Estimated monthly**: $10-30 for moderate usage