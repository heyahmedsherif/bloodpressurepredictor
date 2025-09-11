// Health Prediction Suite - JavaScript Application
class HealthPredictionApp {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.stream = null;
        this.recording = false;
        this.processing = false;
        this.frameCount = 0;
        this.maxFrames = 900; // 30 seconds at 30 FPS
        this.frameInterval = null;
        this.ppgChart = null;
        
        this.initializeElements();
        this.bindEvents();
        this.initializeChart();
    }

    initializeElements() {
        // Video elements
        this.video = document.getElementById('cameraVideo');
        this.canvas = document.getElementById('processedCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Control buttons
        this.startCameraBtn = document.getElementById('startCameraBtn');
        this.stopCameraBtn = document.getElementById('stopCameraBtn');
        this.startRecordingBtn = document.getElementById('startRecordingBtn');
        this.stopRecordingBtn = document.getElementById('stopRecordingBtn');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.predictHealthBtn = document.getElementById('predictHealthBtn');
        this.newSessionBtn = document.getElementById('newSessionBtn');

        // UI elements
        this.cameraStatus = document.getElementById('cameraStatus');
        this.recordingControls = document.getElementById('recordingControls');
        this.frameCounter = document.getElementById('frameCounter');
        this.recordingProgress = document.getElementById('recordingProgress');
        
        // Results elements
        this.ppgResults = document.getElementById('ppgResults');
        this.healthPredictions = document.getElementById('healthPredictions');
        
        // Toast elements
        this.errorToast = new bootstrap.Toast(document.getElementById('errorToast'));
        this.successToast = new bootstrap.Toast(document.getElementById('successToast'));
        this.loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
    }

    bindEvents() {
        this.startCameraBtn.addEventListener('click', () => this.startCamera());
        this.stopCameraBtn.addEventListener('click', () => this.stopCamera());
        this.startRecordingBtn.addEventListener('click', () => this.startRecording());
        this.stopRecordingBtn.addEventListener('click', () => this.stopRecording());
        this.analyzeBtn.addEventListener('click', () => this.analyzeResults());
        this.predictHealthBtn.addEventListener('click', () => this.predictHealth());
        this.newSessionBtn.addEventListener('click', () => this.startNewSession());
    }

    async startCamera() {
        try {
            this.showLoading('Initializing Camera', 'Please allow camera access when prompted');
            
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 30 }
                },
                audio: false
            });

            this.video.srcObject = this.stream;
            this.cameraStatus.style.display = 'none';
            this.video.style.display = 'block';

            // Update UI
            this.startCameraBtn.disabled = true;
            this.stopCameraBtn.disabled = false;
            this.startRecordingBtn.disabled = false;

            this.hideLoading();
            this.showSuccess('Camera started successfully!');

        } catch (error) {
            this.hideLoading();
            this.showError('Failed to access camera: ' + error.message);
            console.error('Camera error:', error);
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        this.video.style.display = 'none';
        this.canvas.style.display = 'none';
        this.cameraStatus.style.display = 'flex';

        // Reset UI
        this.startCameraBtn.disabled = false;
        this.stopCameraBtn.disabled = true;
        this.startRecordingBtn.disabled = true;
        
        this.resetRecording();
    }

    async startRecording() {
        try {
            this.showLoading('Starting Recording', 'Initializing PPG analysis...');

            // Call backend to start recording
            const response = await fetch('/api/start_recording', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error);
            }

            this.recording = true;
            this.frameCount = 0;

            // Update UI
            this.startRecordingBtn.style.display = 'none';
            this.recordingControls.style.display = 'block';
            this.stopRecordingBtn.disabled = false;
            this.analyzeBtn.disabled = true;

            // Show processed video
            this.video.style.display = 'none';
            this.canvas.style.display = 'block';

            // Start frame processing
            this.frameInterval = setInterval(() => this.processFrame(), 1000/30); // 30 FPS

            this.hideLoading();
            this.showSuccess('Recording started! Keep your face in view.');

        } catch (error) {
            this.hideLoading();
            this.showError('Failed to start recording: ' + error.message);
            console.error('Recording error:', error);
        }
    }

    async stopRecording() {
        try {
            this.showLoading('Stopping Recording', 'Finalizing data collection...');

            // Stop frame processing
            if (this.frameInterval) {
                clearInterval(this.frameInterval);
                this.frameInterval = null;
            }

            // Call backend to stop recording
            const response = await fetch('/api/stop_recording', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error);
            }

            this.recording = false;
            
            // Update UI
            this.stopRecordingBtn.disabled = true;
            this.analyzeBtn.disabled = false;

            this.hideLoading();
            this.showSuccess('Recording completed! Click "Analyze Results" to process PPG data.');

        } catch (error) {
            this.hideLoading();
            this.showError('Failed to stop recording: ' + error.message);
            console.error('Stop recording error:', error);
        }
    }

    async processFrame() {
        if (!this.recording || !this.video || this.video.readyState !== 4) {
            return;
        }

        try {
            // Capture frame from video
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

            // Convert to base64
            const frameData = this.canvas.toDataURL('image/jpeg', 0.8);

            // Send to backend for processing
            const response = await fetch('/api/process_frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ frame: frameData })
            });

            const result = await response.json();
            if (result.success) {
                // Update frame counter and progress
                this.frameCount = result.frames_captured;
                this.updateProgress();

                // Display processed frame if available
                if (result.processed_frame) {
                    const img = new Image();
                    img.onload = () => {
                        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                        this.ctx.drawImage(img, 0, 0);
                    };
                    img.src = result.processed_frame;
                }

                // Auto-stop when max frames reached
                if (this.frameCount >= this.maxFrames) {
                    await this.stopRecording();
                }
            }

        } catch (error) {
            console.error('Frame processing error:', error);
        }
    }

    updateProgress() {
        const progress = (this.frameCount / this.maxFrames) * 100;
        this.frameCounter.textContent = `${this.frameCount} / ${this.maxFrames} frames`;
        this.recordingProgress.style.width = `${progress}%`;
        this.recordingProgress.setAttribute('aria-valuenow', progress);
    }

    async analyzeResults() {
        try {
            this.showLoading('Analyzing PPG Data', 'Processing heart rate and signal quality...');

            const response = await fetch('/api/get_results', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error);
            }

            // Display results
            this.displayPPGResults(result);
            this.ppgResults.style.display = 'block';
            this.ppgResults.scrollIntoView({ behavior: 'smooth' });

            this.hideLoading();
            this.showSuccess('PPG analysis completed successfully!');

        } catch (error) {
            this.hideLoading();
            this.showError('Failed to analyze results: ' + error.message);
            console.error('Analysis error:', error);
        }
    }

    displayPPGResults(result) {
        document.getElementById('heartRate').textContent = `${result.heart_rate.toFixed(1)} BPM`;
        document.getElementById('framesProcessed').textContent = result.frames_processed;
        document.getElementById('recordingDuration').textContent = `${result.duration.toFixed(1)} s`;

        // Update PPG chart
        if (result.ppg_signal && result.ppg_signal.length > 0) {
            this.updatePPGChart(result.ppg_signal);
        }

        // Store heart rate for health predictions
        this.lastHeartRate = result.heart_rate;
    }

    async predictHealth() {
        try {
            this.showLoading('Predicting Health Metrics', 'Analyzing demographics and PPG data...');

            // Get patient demographics
            const demographics = {
                age: parseInt(document.getElementById('patientAge').value),
                gender: document.getElementById('patientGender').value,
                height: parseInt(document.getElementById('patientHeight').value),
                weight: parseInt(document.getElementById('patientWeight').value),
                heart_rate: this.lastHeartRate || 75
            };

            const response = await fetch('/api/predict_health', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(demographics)
            });

            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error);
            }

            // Display health predictions
            this.displayHealthPredictions(result.predictions);
            this.healthPredictions.style.display = 'block';
            this.healthPredictions.scrollIntoView({ behavior: 'smooth' });

            this.hideLoading();
            this.showSuccess('Health predictions completed!');

        } catch (error) {
            this.hideLoading();
            this.showError('Failed to predict health metrics: ' + error.message);
            console.error('Health prediction error:', error);
        }
    }

    displayHealthPredictions(predictions) {
        // Blood pressure
        const bp = predictions.blood_pressure;
        document.getElementById('bloodPressureValue').textContent = `${bp.systolic} / ${bp.diastolic}`;
        const bpBadge = document.getElementById('bloodPressureCategory');
        bpBadge.textContent = bp.category;
        bpBadge.className = `badge bg-${bp.category.toLowerCase()}`;

        // Glucose
        const glucose = predictions.glucose;
        document.getElementById('glucoseValue').textContent = glucose.value;
        const glucoseBadge = document.getElementById('glucoseCategory');
        glucoseBadge.textContent = glucose.category;
        glucoseBadge.className = `badge bg-${glucose.category.toLowerCase()}`;

        // Cholesterol
        const cholesterol = predictions.cholesterol;
        document.getElementById('cholesterolValue').textContent = cholesterol.value;
        const cholesterolBadge = document.getElementById('cholesterolCategory');
        cholesterolBadge.textContent = cholesterol.category;
        cholesterolBadge.className = `badge bg-${cholesterol.category.toLowerCase()}`;

        // Cardiovascular Risk
        const cvRisk = predictions.cardiovascular_risk;
        document.getElementById('cvRiskValue').textContent = cvRisk.score;
        const cvRiskBadge = document.getElementById('cvRiskCategory');
        cvRiskBadge.textContent = cvRisk.category;
        cvRiskBadge.className = `badge bg-${cvRisk.category.toLowerCase()}`;
    }

    initializeChart() {
        const ctx = document.getElementById('ppgChart').getContext('2d');
        this.ppgChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'PPG Signal',
                    data: [],
                    borderColor: 'rgb(220, 53, 69)',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    borderWidth: 2,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        beginAtZero: false
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    updatePPGChart(ppgData) {
        const labels = Array.from({length: ppgData.length}, (_, i) => i);
        this.ppgChart.data.labels = labels;
        this.ppgChart.data.datasets[0].data = ppgData;
        this.ppgChart.update();
    }

    startNewSession() {
        // Reset all states
        this.resetRecording();
        this.ppgResults.style.display = 'none';
        this.healthPredictions.style.display = 'none';
        this.lastHeartRate = null;

        // Clear chart
        this.ppgChart.data.labels = [];
        this.ppgChart.data.datasets[0].data = [];
        this.ppgChart.update();

        this.showSuccess('New session started. You can begin recording again.');
    }

    resetRecording() {
        this.recording = false;
        this.frameCount = 0;

        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }

        // Reset UI
        this.startRecordingBtn.style.display = 'block';
        this.recordingControls.style.display = 'none';
        this.recordingProgress.style.width = '0%';
        this.frameCounter.textContent = '0 / 900 frames';
    }

    showLoading(title, subtitle) {
        document.getElementById('loadingText').textContent = title;
        document.getElementById('loadingSubtext').textContent = subtitle;
        this.loadingModal.show();
    }

    hideLoading() {
        this.loadingModal.hide();
    }

    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        this.errorToast.show();
    }

    showSuccess(message) {
        document.getElementById('successMessage').textContent = message;
        this.successToast.show();
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new HealthPredictionApp();
});