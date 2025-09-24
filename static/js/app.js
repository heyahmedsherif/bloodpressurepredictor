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
        this.maxFrames = 150; // 5 seconds at 30 FPS for better signal quality
        this.frameInterval = null;
        this.ppgChart = null;

        // Session tracking for measurement stabilization
        this.sessionId = localStorage.getItem('ppg_session_id') || null;
        this.measurementCount = 0;

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
            // Capture at 30 FPS to match camera capabilities
            // Apple Studio Display typically runs at 30 FPS
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
                heart_rate: this.lastHeartRate || 75,
                session_id: this.sessionId  // Include session ID for measurement stabilization
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

            // Store session ID if provided for measurement stabilization
            if (result.stabilization_info && result.stabilization_info.session_id) {
                this.sessionId = result.stabilization_info.session_id;
                localStorage.setItem('ppg_session_id', this.sessionId);
                this.measurementCount = result.stabilization_info.measurement_count || 0;
                console.log(`Session ${this.sessionId}: ${this.measurementCount} measurements`);
            }

            // Display health predictions
            console.log('Displaying predictions...');
            this.displayHealthPredictions(result.predictions);
            this.healthPredictions.style.display = 'block';
            this.healthPredictions.scrollIntoView({ behavior: 'smooth' });

            // Show stabilization message if provided
            if (result.stabilization_message) {
                const stabMessage = document.createElement('div');
                stabMessage.className = 'alert alert-info mt-2';
                stabMessage.textContent = result.stabilization_message;
                this.healthPredictions.insertBefore(stabMessage, this.healthPredictions.firstChild);
            }

            // Check if retry is suggested due to poor signal quality
            if (result.retry_suggested) {
                setTimeout(() => {
                    if (confirm(result.retry_message + '\n\nWould you like to retry the measurement?')) {
                        // Reset and restart measurement
                        this.reset();
                        this.startRecording();
                    }
                }, 1000);
            }

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
        // Display signal quality if available
        if (predictions.signal_quality) {
            const quality = predictions.signal_quality;
            const qualityHtml = `
                <div class="alert alert-${quality.level === 'excellent' ? 'success' : quality.level === 'good' ? 'info' : quality.level === 'acceptable' ? 'warning' : 'danger'} mt-2">
                    <h6>Signal Quality: ${quality.level.toUpperCase()}</h6>
                    <div class="d-flex justify-content-between">
                        <span>Quality Score: ${(quality.score * 100).toFixed(0)}%</span>
                        <span>Confidence: ${quality.confidence}%</span>
                    </div>
                    ${quality.is_acceptable ?
                        '<small class="text-success">✓ Signal quality acceptable for analysis</small>' :
                        '<small class="text-danger">⚠ Poor signal quality - results may be less accurate</small>'}
                </div>
            `;

            // Add quality info at the top of predictions
            const predictionsContainer = document.getElementById('healthPredictions');
            const existingQuality = predictionsContainer.querySelector('.signal-quality-alert');
            if (existingQuality) {
                existingQuality.remove();
            }
            const qualityDiv = document.createElement('div');
            qualityDiv.className = 'signal-quality-alert';
            qualityDiv.innerHTML = qualityHtml;
            predictionsContainer.insertBefore(qualityDiv, predictionsContainer.firstChild);
        }

        // Blood pressure with confidence
        const bp = predictions.blood_pressure;
        let bpText = `${bp.systolic} / ${bp.diastolic}`;
        if (bp.confidence) {
            bpText += ` <small class="text-muted">(${bp.confidence}% conf.)</small>`;
        }
        document.getElementById('bloodPressureValue').innerHTML = bpText;
        const bpBadge = document.getElementById('bloodPressureCategory');
        const bpStatus = bp.status || bp.category || 'Normal';
        bpBadge.textContent = bpStatus;

        // Safe badge class for blood pressure
        let bpBadgeClass = 'success';
        if (bpStatus.toLowerCase().includes('elevated') || bpStatus.toLowerCase().includes('high')) {
            bpBadgeClass = 'warning';
        }
        bpBadge.className = `badge bg-${bpBadgeClass}`;

        // Glucose
        const glucose = predictions.glucose;
        document.getElementById('glucoseValue').textContent = glucose.value;
        const glucoseBadge = document.getElementById('glucoseCategory');
        const glucoseStatus = glucose.status || glucose.category || 'Normal';
        glucoseBadge.textContent = glucoseStatus;

        // Safe badge class for glucose
        let glucoseBadgeClass = 'success';
        if (glucoseStatus.toLowerCase().includes('elevated') || glucoseStatus.toLowerCase().includes('high')) {
            glucoseBadgeClass = 'warning';
        }
        glucoseBadge.className = `badge bg-${glucoseBadgeClass}`;

        // Cholesterol (with LDL/HDL if available)
        const cholesterol = predictions.cholesterol;
        document.getElementById('cholesterolValue').textContent = cholesterol.value;
        const cholesterolBadge = document.getElementById('cholesterolCategory');
        const cholesterolStatus = cholesterol.status || cholesterol.category || 'Normal';
        cholesterolBadge.textContent = cholesterolStatus;

        // Safe badge class assignment
        let badgeClass = 'secondary';
        if (cholesterolStatus) {
            const statusLower = cholesterolStatus.toLowerCase().replace(/\s+/g, '-');
            if (statusLower.includes('normal')) badgeClass = 'success';
            else if (statusLower.includes('borderline')) badgeClass = 'warning';
            else if (statusLower.includes('high')) badgeClass = 'danger';
            else if (statusLower.includes('elevated')) badgeClass = 'warning';
        }
        cholesterolBadge.className = `badge bg-${badgeClass}`;

        // Check if we have LDL/HDL data (enhanced models)
        if (cholesterol.ldl && cholesterol.hdl) {
            document.getElementById('cholesterolDetails').style.display = 'block';
            document.getElementById('ldlValue').textContent = cholesterol.ldl;
            document.getElementById('hdlValue').textContent = cholesterol.hdl;
            document.getElementById('ldlHdlRatio').textContent = cholesterol.ldl_hdl_ratio;

            // Set risk badge color
            const cvRiskLdl = document.getElementById('cvRiskLdl');
            cvRiskLdl.textContent = cholesterol.cardiovascular_risk;
            const riskColors = {
                'Optimal': 'success',
                'Low': 'info',
                'Moderate': 'warning',
                'High': 'danger'
            };
            cvRiskLdl.className = `badge bg-${riskColors[cholesterol.cardiovascular_risk] || 'secondary'}`;

            // Show comparison if available
            if (cholesterol.total_new) {
                document.getElementById('totalNew').textContent = cholesterol.total_new;

                // Show validation status
                if (cholesterol.comparison && cholesterol.comparison.sum_validation) {
                    const status = cholesterol.comparison.sum_validation;
                    const statusSpan = document.getElementById('comparisonStatus');
                    if (status.includes('Match')) {
                        statusSpan.innerHTML = '<i class="fas fa-check-circle text-success"></i>';
                        statusSpan.title = 'LDL + HDL + VLDL ≈ Total';
                    } else {
                        statusSpan.innerHTML = '<i class="fas fa-exclamation-triangle text-warning"></i>';
                        statusSpan.title = `Difference: ${Math.round(cholesterol.comparison.percentage_difference)}%`;
                    }
                }
            }
        } else {
            document.getElementById('cholesterolDetails').style.display = 'none';
        }

        // Cardiovascular Risk - now calculated from cholesterol ratio
        if (predictions.cardiovascular_risk) {
            // Legacy support for old response format
            const cvRisk = predictions.cardiovascular_risk;
            document.getElementById('cvRiskValue').textContent = cvRisk.score || '--';
            const cvRiskBadge = document.getElementById('cvRiskCategory');
            cvRiskBadge.textContent = cvRisk.category || 'Unknown';

            let riskBadgeClass = 'secondary';
            if (cvRisk.category) {
                const riskLower = cvRisk.category.toLowerCase();
                if (riskLower.includes('low')) riskBadgeClass = 'success';
                else if (riskLower.includes('moderate')) riskBadgeClass = 'warning';
                else if (riskLower.includes('high')) riskBadgeClass = 'danger';
            }
            cvRiskBadge.className = `badge bg-${riskBadgeClass}`;
        } else if (predictions.cholesterol && predictions.cholesterol.ldl_hdl_ratio) {
            // Calculate CV risk from LDL/HDL ratio
            const ratio = predictions.cholesterol.ldl_hdl_ratio;
            let riskScore, riskCategory, riskBadgeClass;

            if (ratio < 2.0) {
                riskScore = 20;
                riskCategory = 'Low Risk';
                riskBadgeClass = 'success';
            } else if (ratio < 2.5) {
                riskScore = 35;
                riskCategory = 'Moderate Risk';
                riskBadgeClass = 'info';
            } else if (ratio < 3.5) {
                riskScore = 55;
                riskCategory = 'Elevated Risk';
                riskBadgeClass = 'warning';
            } else {
                riskScore = 75;
                riskCategory = 'High Risk';
                riskBadgeClass = 'danger';
            }

            document.getElementById('cvRiskValue').textContent = riskScore;
            const cvRiskBadge = document.getElementById('cvRiskCategory');
            cvRiskBadge.textContent = riskCategory;
            cvRiskBadge.className = `badge bg-${riskBadgeClass}`;
        } else {
            // No risk data available
            document.getElementById('cvRiskValue').textContent = '--';
            document.getElementById('cvRiskCategory').textContent = 'Unknown';
            document.getElementById('cvRiskCategory').className = 'badge bg-secondary';
        }

        // Vascular Age Display (if available)
        if (predictions.vascular_age) {
            document.getElementById('vascularAgeResult').style.display = 'block';
            const vAge = predictions.vascular_age;

            // Main vascular age display - prefer formula-based (more accurate)
            // ML is experimental without proper training data
            const displayAge = vAge.vascular_age || vAge.vascular_age_ml;
            document.getElementById('vascularAge').textContent = displayAge;
            document.getElementById('chronoAge').textContent = vAge.chronological_age + ' years';

            // Age difference with color coding
            const ageDiff = vAge.age_difference;
            const ageDiffElement = document.getElementById('ageDifference');
            if (ageDiff > 0) {
                ageDiffElement.textContent = `+${ageDiff} years`;
                ageDiffElement.className = 'fw-bold text-danger';
            } else if (ageDiff < 0) {
                ageDiffElement.textContent = `${ageDiff} years`;
                ageDiffElement.className = 'fw-bold text-success';
            } else {
                ageDiffElement.textContent = 'Same as chronological';
                ageDiffElement.className = 'fw-bold text-info';
            }

            // Status badge - show ML status if available with confidence
            const statusBadge = document.getElementById('vascularStatus');
            if (vAge.vascular_age_ml && vAge.confidence) {
                const confidencePct = Math.round(vAge.confidence * 100);
                statusBadge.textContent = `${vAge.status_ml || vAge.status} (${confidencePct}% confidence)`;
            } else {
                statusBadge.textContent = vAge.status;
            }
            let statusClass = 'bg-secondary';
            if (vAge.status === 'Excellent' || vAge.status === 'Good') {
                statusClass = 'bg-success';
            } else if (vAge.status === 'Normal') {
                statusClass = 'bg-info';
            } else if (vAge.status === 'Accelerated Aging') {
                statusClass = 'bg-warning';
            } else if (vAge.status === 'Significant Aging') {
                statusClass = 'bg-danger';
            }
            statusBadge.className = `badge fs-6 ${statusClass}`;

            // Vascular health risk level
            document.getElementById('vascularHealth').textContent = vAge.risk_level;
            document.getElementById('vascularHealth').className =
                vAge.risk_level === 'Low' ? 'fw-bold text-success' :
                vAge.risk_level === 'Moderate' ? 'fw-bold text-warning' :
                'fw-bold text-danger';

            // Health score progress bar
            const healthScore = vAge.health_score;
            const healthBar = document.getElementById('healthScoreBar');
            healthBar.style.width = healthScore + '%';
            document.getElementById('healthScoreText').textContent = `${healthScore}/100`;

            // Progress bar color based on score
            if (healthScore >= 70) {
                healthBar.className = 'progress-bar bg-success';
            } else if (healthScore >= 50) {
                healthBar.className = 'progress-bar bg-warning';
            } else {
                healthBar.className = 'progress-bar bg-danger';
            }
        }
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

        // Hide all results including vascular age
        this.ppgResults.style.display = 'none';
        this.healthPredictions.style.display = 'none';
        if (document.getElementById('vascularAgeResult')) {
            document.getElementById('vascularAgeResult').style.display = 'none';
        }
        this.lastHeartRate = null;

        // Clear all displayed values
        document.getElementById('heartRate').textContent = '-- BPM';
        document.getElementById('framesProcessed').textContent = '--';
        document.getElementById('recordingDuration').textContent = '-- s';

        // Clear health prediction values
        document.getElementById('bloodPressureValue').textContent = '-- / --';
        document.getElementById('glucoseValue').textContent = '--';
        document.getElementById('cholesterolValue').textContent = '--';
        document.getElementById('cvRiskValue').textContent = '--';

        // Clear vascular age values
        if (document.getElementById('vascularAge')) {
            document.getElementById('vascularAge').textContent = '--';
            document.getElementById('chronoAge').textContent = '-- years';
            document.getElementById('ageDifference').textContent = '--';
            document.getElementById('vascularHealth').textContent = '--';
            document.getElementById('healthScoreBar').style.width = '0%';
            document.getElementById('healthScoreText').textContent = '0/100';
            document.getElementById('vascularStatus').textContent = '--';
        }

        // Hide cholesterol details
        if (document.getElementById('cholesterolDetails')) {
            document.getElementById('cholesterolDetails').style.display = 'none';
        }

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
        this.frameCounter.textContent = `0 / ${this.maxFrames} frames`;
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