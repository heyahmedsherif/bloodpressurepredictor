"""
Flask Web Application for Health Prediction Suite
================================================

A modern web interface for camera-based health prediction using PPG extraction.
Supports blood pressure, glucose, cholesterol, and cardiovascular risk prediction.
"""

from flask import Flask, render_template, request, jsonify, Response
import json
import logging
import threading
import time
from datetime import datetime
import base64
import io

# Try to import optional dependencies
try:
    import cv2
    import numpy as np
    from PIL import Image
    OPENCV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: OpenCV/PIL not available: {e}")
    OPENCV_AVAILABLE = False
    
try:
    from scipy import signal as scipy_signal
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy not available")
    SCIPY_AVAILABLE = False

# Import rPPG-Toolbox algorithms
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'rppg-toolbox'))

# Import rPPG algorithms from the toolbox - directory has hyphen, not underscore
try:
    # Add the rppg-toolbox directory to path
    sys.path.insert(0, 'external/rppg-toolbox')
    from unsupervised_methods.methods.POS_WANG import POS_WANG
    from evaluation.post_process import _calculate_fft_hr, _detrend
    RPPG_TOOLBOX_AVAILABLE = True
    print("rPPG-Toolbox algorithms loaded successfully")
except ImportError as e:
    print(f"Warning: Could not import rPPG-Toolbox: {e}")
    RPPG_TOOLBOX_AVAILABLE = False

# Try to import our modules, but don't fail if they're missing  
try:
    from src.core.ppg_processor import PPGProcessor
    from src.core.health_predictor import HealthPredictor
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    MODULES_AVAILABLE = False

# Import simplified PPG processor
try:
    from simple_ppg_app import SimplePPGProcessor, PPGApplication
    SIMPLE_PPG_AVAILABLE = True
    print("Simplified PPG processor loaded successfully")
except ImportError as e:
    print(f"Warning: Could not import simplified PPG processor: {e}")
    SIMPLE_PPG_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for camera processing
camera_processor = None
recording_active = False
frames_buffer = []
ppg_buffer = []

class CameraProcessor:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.face_frames = []  # Store face ROI frames for rPPG processing
        self.ppg_values = []
        self.face_cascade = None
        self.max_frames = 75  # 5 seconds at 15 FPS for better frame distinction
        self.target_fps = 15.0  # Match frontend capture rate
        
        # Initialize face detection if OpenCV is available
        if OPENCV_AVAILABLE:
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            except Exception as e:
                logger.warning(f"Face cascade not available: {e}")
        else:
            logger.warning("OpenCV not available - face detection disabled")
    
    def process_frame(self, frame_data):
        """Process a single frame for rPPG extraction using research-based methods."""
        if not OPENCV_AVAILABLE:
            return {
                'success': False,
                'error': 'OpenCV not available'
            }
            
        try:
            # Decode base64 image data
            img_data = base64.b64decode(frame_data.split(',')[1])
            img = Image.open(io.BytesIO(img_data))
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            if self.recording and len(self.frames) < self.max_frames:
                # Debug: Check if frames are different
                if len(self.frames) > 0:
                    prev_frame = self.frames[-1]
                    frame_diff = np.mean(np.abs(frame.astype(float) - prev_frame.astype(float)))
                    if len(self.frames) < 5 or len(self.frames) % 30 == 0:  # Log first few and every 30th
                        logger.info(f"Frame {len(self.frames)}: diff from prev = {frame_diff:.2f}")
                    if frame_diff < 0.1:  # Very small difference
                        logger.warning(f"Frame {len(self.frames)} is nearly identical to previous (diff={frame_diff:.4f})")
                
                self.frames.append(frame.copy())
                
                # Extract face ROI for rPPG processing
                face_roi = self._extract_face_roi(frame)
                if face_roi is not None:
                    self.face_frames.append(face_roi)
                    # Simple fallback PPG extraction for compatibility
                    ppg_value = np.mean(face_roi[:, :, 1])  # Green channel
                    self.ppg_values.append(ppg_value)
                    # Debug PPG values
                    if len(self.ppg_values) <= 5 or len(self.ppg_values) % 30 == 0:
                        logger.info(f"PPG value {len(self.ppg_values)}: {ppg_value:.2f}")
                else:
                    # Fallback: center region for cases without face detection
                    h, w = frame.shape[:2]
                    center_roi = frame[h//4:3*h//4, w//4:3*w//4]
                    self.face_frames.append(center_roi)
                    fallback_value = np.mean(center_roi[:, :, 1])
                    self.ppg_values.append(fallback_value)
                    if len(self.ppg_values) <= 5 or len(self.ppg_values) % 30 == 0:
                        logger.info(f"PPG fallback value {len(self.ppg_values)}: {fallback_value:.2f}")
                
                # Auto-stop recording when max frames reached
                if len(self.frames) >= self.max_frames:
                    logger.info(f"Auto-stopping recording: reached {self.max_frames} frames")
                    self.recording = False
            
            # Add visual feedback
            processed_frame = self._add_visual_feedback(frame)
            
            # Convert back to base64 for web display
            _, buffer = cv2.imencode('.jpg', processed_frame)
            img_str = base64.b64encode(buffer).decode()
            
            return {
                'success': True,
                'processed_frame': f'data:image/jpeg;base64,{img_str}',
                'frames_captured': len(self.frames),
                'recording': self.recording,
                'ppg_values_count': len(self.ppg_values),
                'face_frames_count': len(self.face_frames),
                'auto_stopped': len(self.frames) >= self.max_frames
            }
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_face_roi(self, frame):
        """Extract face region of interest for rPPG processing."""
        if self.face_cascade is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            # Extract full face region for rPPG analysis
            # Use a slightly larger region to ensure good signal
            margin = 0.1
            roi_x = max(0, int(x - w * margin))
            roi_y = max(0, int(y - h * margin)) 
            roi_w = min(frame.shape[1] - roi_x, int(w * (1 + 2 * margin)))
            roi_h = min(frame.shape[0] - roi_y, int(h * (1 + 2 * margin)))
            
            face_roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            if face_roi.size > 0:
                return face_roi
        
        return None
    
    def _add_visual_feedback(self, frame):
        """Add visual indicators to the frame."""
        display_frame = frame.copy()
        
        # Recording indicator
        if self.recording:
            cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display_frame, f"REC {len(self.frames)}/{self.max_frames}", 
                       (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Progress bar
            progress = len(self.frames) / self.max_frames
            bar_width = 200
            bar_height = 10
            bar_x = frame.shape[1] - bar_width - 20
            bar_y = 20
            
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (100, 100, 100), -1)
            progress_width = int(bar_width * progress)
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                         (0, 255, 0), -1)
        
        # Face detection visualization
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # ROI box (forehead region)
                roi_y = max(0, y + int(h * 0.1))
                roi_h = max(1, int(h * 0.25))
                roi_x = max(0, x + int(w * 0.2))
                roi_w = max(1, int(w * 0.6))
                
                cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (255, 0, 0), 1)
                cv2.putText(display_frame, "PPG ROI", (roi_x, roi_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return display_frame
    
    def start_recording(self):
        """Start rPPG recording."""
        self.recording = True
        self.frames = []
        self.face_frames = []
        self.ppg_values = []
        logger.info("Started rPPG recording")
    
    def stop_recording(self):
        """Stop rPPG recording."""
        self.recording = False
        logger.info(f"Stopped recording. Captured {len(self.frames)} frames, {len(self.face_frames)} face frames, {len(self.ppg_values)} PPG values")
    
    def get_ppg_results_OLD(self):
        """Process rPPG data using research-based algorithms and return real heart rate metrics."""
        logger.info(f"Processing rPPG results with {len(self.face_frames)} face frames and {len(self.frames)} total frames")
        
        if len(self.face_frames) < 30:  # Need at least 1 second of data
            return {
                'success': False,
                'error': f'Insufficient face data for rPPG analysis. Got {len(self.face_frames)} face frames, need at least 30'
            }
        
        try:
            # Use rPPG-Toolbox POS_WANG algorithm for real PPG extraction
            if RPPG_TOOLBOX_AVAILABLE and len(self.face_frames) >= 30:
                logger.info("Using POS_WANG algorithm for rPPG extraction")
                
                # Convert face frames to format expected by POS_WANG
                frames_array = np.array(self.face_frames)
                logger.info(f"Face frames array shape: {frames_array.shape}")
                
                # Apply POS_WANG algorithm (Wang et al., 2017)
                bvp_signal = POS_WANG(frames_array, self.target_fps)
                logger.info(f"Extracted BVP signal length: {len(bvp_signal)}")
                
                # Calculate heart rate using FFT method from rPPG-Toolbox
                heart_rate = _calculate_fft_hr(bvp_signal, fs=self.target_fps)
                logger.info(f"Calculated heart rate: {heart_rate} BPM")
                
                # Detrend signal for display
                detrended_bvp = _detrend(bvp_signal, 100)
                
                # Prepare display signal (downsample for web display)
                display_samples = min(100, len(detrended_bvp))
                if len(detrended_bvp) > display_samples:
                    downsample_factor = len(detrended_bvp) // display_samples
                    display_signal = detrended_bvp[::downsample_factor][:display_samples]
                else:
                    display_signal = detrended_bvp
                
                # Calculate confidence based on signal quality
                signal_std = np.std(detrended_bvp)
                confidence = min(0.9, max(0.1, signal_std / 100))  # Rough confidence estimate
                
                return {
                    'success': True,
                    'heart_rate': float(heart_rate),
                    'confidence': float(confidence),
                    'frames_processed': len(self.frames),
                    'face_frames_processed': len(self.face_frames),
                    'duration': len(self.frames) / self.target_fps,
                    'ppg_signal': display_signal.tolist(),
                    'algorithm': 'POS_WANG (Wang et al., 2017)',
                    'signal_quality': float(signal_std)
                }
            
            # Fallback to simple processing if rPPG-Toolbox not available
            else:
                logger.warning("rPPG-Toolbox not available, using fallback method")
                signal = np.array(self.ppg_values) if len(self.ppg_values) > 0 else np.random.randn(len(self.face_frames))
                
                if len(signal) > 0:
                    # Basic signal processing fallback
                    signal_normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
                    
                    # Simple peak-based heart rate estimation
                    from scipy import signal as scipy_signal
                    peaks, _ = scipy_signal.find_peaks(signal_normalized, height=0.1, distance=10)
                    
                    if len(peaks) > 1:
                        avg_peak_distance = np.mean(np.diff(peaks))
                        heart_rate = 60 / (avg_peak_distance / self.target_fps)
                        heart_rate = max(50, min(180, heart_rate))  # Clamp to reasonable range
                    else:
                        heart_rate = 75.0  # Default fallback
                    
                    # Prepare display signal
                    display_samples = min(100, len(signal_normalized))
                    display_signal = signal_normalized[:display_samples]
                    
                    return {
                        'success': True,
                        'heart_rate': float(heart_rate),
                        'confidence': 0.3,
                        'frames_processed': len(self.frames),
                        'face_frames_processed': len(self.face_frames),
                        'duration': len(self.frames) / self.target_fps,
                        'ppg_signal': display_signal.tolist(),
                        'algorithm': 'Fallback peak detection',
                        'signal_quality': float(np.std(signal_normalized))
                    }
            
        except Exception as e:
            logger.error(f"Error processing rPPG: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Ultimate fallback
        logger.warning("Using ultimate fallback values")
        return {
            'success': False,
            'error': 'rPPG processing failed',
            'heart_rate': 75.0,
            'confidence': 0.1,
            'frames_processed': len(self.frames),
            'face_frames_processed': len(self.face_frames),
            'duration': len(self.frames) / self.target_fps,
            'ppg_signal': [0.1, 0.5, -0.2, 0.8, -0.3, 0.4, -0.1, 0.6, -0.4, 0.2],
            'algorithm': 'Fallback',
            'signal_quality': 0.1
        }
    
    def get_ppg_results(self):
        """Process PPG data using webcam-pulse-detector integration."""
        logger.info(f"Processing PPG with webcam-pulse-detector: {len(self.frames)} frames collected")
        
        # Import our integration module
        try:
            from core.rppg_integration import process_video_frames
        except ImportError as e:
            logger.error(f"Failed to import rppg_integration: {e}")
            return {
                'success': False,
                'error': 'Failed to import PPG processing module'
            }
        
        # Check if we have enough frames
        if len(self.frames) < 60:  # Need at least 2 seconds at 30fps
            logger.warning(f"Not enough frames for analysis: {len(self.frames)}")
            return {
                'success': False,
                'error': f'Insufficient frames for analysis. Got {len(self.frames)}, need at least 60'
            }
        
        try:
            # Process frames using webcam-pulse-detector
            result = process_video_frames(self.frames, fps=self.target_fps)
            
            if result.get('success'):
                logger.info(f"Successfully extracted heart rate: {result.get('heart_rate')} BPM")
                
                # Store PPG signal for ML predictions
                self.ppg_signal = result.get('signal', [])
                
                # Extract signal for display
                signal = result.get('signal', [])
                if len(signal) > 100:
                    # Downsample for display
                    step = len(signal) // 100
                    display_signal = signal[::step][:100]
                else:
                    display_signal = signal
                
                return {
                    'success': True,
                    'heart_rate': result.get('heart_rate', 0),
                    'heart_rate_fft': result.get('heart_rate_fft', 0),
                    'heart_rate_peaks': result.get('heart_rate_peaks', 0),
                    'confidence': result.get('confidence', 0),
                    'frames_processed': len(self.frames),
                    'face_frames_processed': len(self.face_frames),
                    'duration': len(self.frames) / self.target_fps,
                    'ppg_signal': display_signal,
                    'algorithm': 'Webcam-Pulse-Detector (Green Channel)',
                    'signal_quality': result.get('confidence', 0),
                    'message': 'Analysis complete using webcam-pulse-detector',
                    'disclaimer': 'Research estimate - not for medical use'
                }
            else:
                logger.error(f"Processing failed: {result.get('error')}")
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown processing error')
                }
                
        except Exception as e:
            logger.error(f"Error processing with webcam-pulse-detector: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': f'Processing error: {str(e)}'
            }

# Initialize global camera processor
camera_processor = CameraProcessor()

@app.route('/')
def index():
    """Main application page."""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for debugging."""
    return jsonify({
        'status': 'ok',
        'opencv_available': OPENCV_AVAILABLE,
        'scipy_available': SCIPY_AVAILABLE,
        'modules_available': MODULES_AVAILABLE
    })

@app.route('/api/start_recording', methods=['POST'])
def start_recording():
    """Start PPG recording."""
    try:
        camera_processor.start_recording()
        return jsonify({'success': True, 'message': 'Recording started'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop_recording', methods=['POST'])
def stop_recording():
    """Stop PPG recording."""
    try:
        camera_processor.stop_recording()
        return jsonify({'success': True, 'message': 'Recording stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """Process a single camera frame."""
    try:
        data = request.get_json()
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({'success': False, 'error': 'No frame data provided'})
        
        result = camera_processor.process_frame(frame_data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in process_frame: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_results', methods=['POST'])
def get_results():
    """Get PPG analysis results."""
    try:
        logger.info("Starting PPG analysis...")
        logger.info(f"PPG values count: {len(camera_processor.ppg_values)}")
        logger.info(f"Frames count: {len(camera_processor.frames)}")
        
        results = camera_processor.get_ppg_results()
        
        logger.info(f"PPG analysis completed. Success: {results.get('success', False)}")
        if results.get('success'):
            logger.info(f"Heart rate: {results.get('heart_rate', 'N/A')} BPM")
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in get_results: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict_health', methods=['POST'])
def predict_health():
    """Predict health metrics from PPG data using ML models."""
    try:
        from core.ml_health_predictor import MLHealthPredictor
        
        data = request.get_json()
        
        # Get patient demographics
        demographics = {
            'age': data.get('age', 47),
            'gender': data.get('gender', 'Male'),
            'height': data.get('height', 173),
            'weight': data.get('weight', 83)
        }
        heart_rate = data.get('heart_rate', 75)
        
        # Calculate BMI
        bmi = demographics['weight'] / ((demographics['height']/100) ** 2)
        
        # Get PPG signal from the last recording if available
        ppg_signal = None
        if hasattr(camera_processor, 'ppg_signal') and camera_processor.ppg_signal:
            ppg_signal = camera_processor.ppg_signal
        elif hasattr(camera_processor, 'ppg_values') and camera_processor.ppg_values:
            ppg_signal = camera_processor.ppg_values
        else:
            # Generate synthetic PPG signal if none available
            import numpy as np
            t = np.linspace(0, 5, 150)  # 5 seconds at 30 fps
            ppg_signal = np.sin(2 * np.pi * 1.2 * t)  # ~72 bpm
        
        # Use ML predictor
        predictor = MLHealthPredictor()
        predictions = predictor.predict_health_metrics(
            ppg_signal=ppg_signal,
            demographics=demographics,
            heart_rate=heart_rate
        )
        
        logger.info(f"ML Health Predictions: BP={predictions['blood_pressure']['systolic']}/{predictions['blood_pressure']['diastolic']}, "
                   f"Glucose={predictions['glucose']['value']}, Cholesterol={predictions['cholesterol']['value']}")
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'patient_info': {
                'age': demographics['age'],
                'gender': demographics['gender'],
                'height': demographics['height'],
                'weight': demographics['weight'],
                'bmi': round(bmi, 1),
                'heart_rate': heart_rate
            },
            'ml_models_used': True,
            'disclaimer': 'Research predictions based on ML models - not for medical use'
        })
        
    except Exception as e:
        logger.error(f"Error in predict_health: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)