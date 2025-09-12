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
        this.maxFrames = 75; // 5 seconds at 15 FPS
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

            // IMPORTANT: Keep video visible for live feed, canvas will be updated with frames
            // Don't hide the video or we'll get static frames!
            this.video.style.display = 'block';
            this.canvas.style.display = 'none';

            // Start frame processing
            // Capture at 15 FPS to ensure distinct frames and reduce duplicate captures
            // PPG extraction works well at 15-30 FPS
            this.frameInterval = setInterval(() => this.processFrame(), 1000/15); // 15 FPS

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
            // Directly capture and send frame
            // The setInterval already handles timing, no need for additional callbacks
            await this.captureAndSendFrame();
        } catch (error) {
            console.error('Frame processing error:', error);
        }
    }
    
    async captureAndSendFrame() {
        if (!this.recording) return;
        
        try {
            // Capture frame from video
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            
            // Add visual recording indicators as overlay
            this.drawRecordingIndicators();

            // Convert to base64
            const frameData = this.canvas.toDataURL('image/jpeg', 0.8);

            // Send to backend for processing with timestamp to ensure uniqueness
            const response = await fetch('/api/process_frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    frame: frameData,
                    timestamp: Date.now(),
                    frameNumber: this.frameCount
                })
            });

            const result = await response.json();
            if (result.success) {
                // Update frame counter and progress
                this.frameCount = result.frames_captured;
                this.updateProgress();

                // Don't display processed frame - keep showing live video
                // The processed frame contains face detection boxes which make it appear frozen
                // We'll add visual indicators separately without replacing the live feed

                // Auto-stop when max frames reached or server indicates auto-stop
                // Check this.recording to prevent multiple stop calls
                if (this.recording && (this.frameCount >= this.maxFrames || result.auto_stopped)) {
                    console.log('Auto-stopping recording: max frames reached');
                    // Clear interval immediately to prevent more frames
                    if (this.frameInterval) {
                        clearInterval(this.frameInterval);
                        this.frameInterval = null;
                    }
                    this.recording = false; // Prevent multiple calls
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
    
    drawRecordingIndicators() {
        // Add recording indicator (red circle)
        this.ctx.fillStyle = 'red';
        this.ctx.beginPath();
        this.ctx.arc(30, 30, 10, 0, 2 * Math.PI);
        this.ctx.fill();
        
        // Add "REC" text
        this.ctx.fillStyle = 'red';
        this.ctx.font = 'bold 16px Arial';
        this.ctx.fillText(`REC ${this.frameCount}/${this.maxFrames}`, 50, 35);
        
        // Add timestamp to prove continuous recording
        const timestamp = new Date().toLocaleTimeString();
        this.ctx.fillStyle = 'white';
        this.ctx.strokeStyle = 'black';
        this.ctx.lineWidth = 3;
        this.ctx.strokeText(timestamp, 10, this.canvas.height - 10);
        this.ctx.fillText(timestamp, 10, this.canvas.height - 10);
        
        // Add progress bar
        const barWidth = 200;
        const barHeight = 10;
        const barX = this.canvas.width - barWidth - 20;
        const barY = 20;
        const progress = this.frameCount / this.maxFrames;
        
        // Background
        this.ctx.fillStyle = 'rgba(100, 100, 100, 0.5)';
        this.ctx.fillRect(barX, barY, barWidth, barHeight);
        
        // Progress
        this.ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
        this.ctx.fillRect(barX, barY, barWidth * progress, barHeight);
    }

    async analyzeResults() {
        console.log('Starting analysis...');
        try {
            this.showLoading('Analyzing PPG Data', 'Processing heart rate and signal quality...');

            // Add timeout controller
            const controller = new AbortController();
            const timeoutId = setTimeout(() => {
                console.log('Request timed out');
                controller.abort();
            }, 10000); // 10 second timeout

            console.log('Sending request to /api/get_results');
            const response = await fetch('/api/get_results', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                signal: controller.signal
            });

            clearTimeout(timeoutId);
            console.log('Response received:', response.status);

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const result = await response.json();
            console.log('Result parsed:', result);
            
            // Force hide loading first
            this.hideLoading();
            
            if (!result.success) {
                throw new Error(result.error || 'Analysis failed');
            }

            // Display results
            this.displayPPGResults(result);
            this.ppgResults.style.display = 'block';
            this.ppgResults.scrollIntoView({ behavior: 'smooth' });

            this.showSuccess('PPG analysis completed successfully!');

        } catch (error) {
            console.error('Analysis error:', error);
            this.hideLoading();
            if (error.name === 'AbortError') {
                this.showError('Analysis timed out. Please try again with shorter recording.');
            } else {
                this.showError('Failed to analyze results: ' + error.message);
            }
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
            console.log('Starting health prediction...');
            this.showLoading('Predicting Health Metrics', 'Analyzing demographics and PPG data...');

            // Get patient demographics with defaults if elements don't exist
            const demographics = {
                age: parseInt(document.getElementById('patientAge')?.value || '47'),
                gender: document.getElementById('patientGender')?.value || 'Male',
                height: parseInt(document.getElementById('patientHeight')?.value || '173'),
                weight: parseInt(document.getElementById('patientWeight')?.value || '83'),
                heart_rate: this.lastHeartRate || 75
            };

            console.log('Sending demographics:', demographics);

            const response = await fetch('/api/predict_health', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(demographics)
            });

            console.log('Response received:', response.status);
            const result = await response.json();
            console.log('Result:', result);
            
            if (!result.success) {
                throw new Error(result.error || 'Unknown error');
            }

            // Display health predictions
            console.log('Displaying predictions...');
            this.displayHealthPredictions(result.predictions);
            this.healthPredictions.style.display = 'block';
            this.healthPredictions.scrollIntoView({ behavior: 'smooth' });

            console.log('Hiding loading modal...');
            this.hideLoading();
            this.showSuccess('Health predictions completed!');
            console.log('Health prediction completed successfully');

        } catch (error) {
            console.error('Health prediction error:', error);
            console.log('Attempting to hide loading modal after error...');
            
            // Force hide the modal as fallback
            try {
                this.hideLoading();
            } catch (hideError) {
                console.error('Error hiding loading modal:', hideError);
                // Force hide using DOM manipulation
                const modal = document.getElementById('loadingModal');
                if (modal) {
                    modal.style.display = 'none';
                    modal.classList.remove('show');
                    document.body.classList.remove('modal-open');
                    const backdrop = document.querySelector('.modal-backdrop');
                    if (backdrop) backdrop.remove();
                }
            }
            
            this.showError('Failed to predict health metrics: ' + error.message);
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
        // Stop the camera first to ensure clean reset
        this.stopCamera();
        
        // Reset all states
        this.resetRecording();
        this.ppgResults.style.display = 'none';
        this.healthPredictions.style.display = 'none';
        this.lastHeartRate = null;

        // Clear chart
        this.ppgChart.data.labels = [];
        this.ppgChart.data.datasets[0].data = [];
        this.ppgChart.update();

        // Show message with instruction to restart camera
        this.showSuccess('New session started. Please click "Start Camera" to begin.');
        
        // Enable the start camera button
        this.startCameraBtn.disabled = false;
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
        this.frameCounter.textContent = '0 / 150 frames';
    }

    showLoading(title, subtitle) {
        document.getElementById('loadingText').textContent = title;
        document.getElementById('loadingSubtext').textContent = subtitle;
        this.loadingModal.show();
    }

    hideLoading() {
        try {
            // Remove focus from the modal first
            const modal = document.getElementById('loadingModal');
            if (modal && modal.contains(document.activeElement)) {
                document.activeElement.blur();
            }
            
            // Hide using Bootstrap modal method
            this.loadingModal.hide();
            
            // Force cleanup after a short delay
            setTimeout(() => {
                if (modal) {
                    modal.style.display = 'none';
                    modal.classList.remove('show');
                    modal.removeAttribute('aria-hidden');
                    modal.removeAttribute('aria-modal');
                    modal.removeAttribute('role');
                }
                document.body.classList.remove('modal-open');
                document.body.style.overflow = '';
                document.body.style.paddingRight = '';
                
                // Remove all backdrops
                const backdrops = document.querySelectorAll('.modal-backdrop');
                backdrops.forEach(backdrop => backdrop.remove());
            }, 300);
        } catch (error) {
            console.error('Error hiding loading modal:', error);
            // Force hide as fallback
            const modal = document.getElementById('loadingModal');
            if (modal) {
                modal.style.display = 'none';
                modal.classList.remove('show');
                modal.removeAttribute('aria-hidden');
                modal.removeAttribute('aria-modal');
            }
            document.body.classList.remove('modal-open');
            document.body.style.overflow = '';
            const backdrops = document.querySelectorAll('.modal-backdrop');
            backdrops.forEach(backdrop => backdrop.remove());
        }
    }

    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        this.errorToast.show();
    }

    showSuccess(message) {
        document.getElementById('successMessage').textContent = message;
        this.successToast.show();
    }
    
    forceCloseModal() {
        console.log('Force closing modal...');
        try {
            // Hide using Bootstrap method first
            this.loadingModal.hide();
        } catch (e) {
            console.error('Bootstrap hide failed:', e);
        }
        
        // Force hide with DOM manipulation
        const modal = document.getElementById('loadingModal');
        if (modal) {
            modal.style.display = 'none';
            modal.classList.remove('show');
            modal.removeAttribute('aria-hidden');
            modal.removeAttribute('aria-modal');
            modal.removeAttribute('role');
        }
        
        // Clean up body classes and styles
        document.body.classList.remove('modal-open');
        document.body.style.overflow = '';
        document.body.style.paddingRight = '';
        
        // Remove all backdrops
        const backdrops = document.querySelectorAll('.modal-backdrop');
        backdrops.forEach(backdrop => backdrop.remove());
        
        console.log('Modal force closed');
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new HealthPredictionApp();
});