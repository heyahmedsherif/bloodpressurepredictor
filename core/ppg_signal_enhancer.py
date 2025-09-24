"""
PPG Signal Enhancement and Quality Assessment
==============================================
Implements Hybrid Smart Processing for improved HR and BP accuracy.
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks, welch
import logging
import json
import os

logger = logging.getLogger(__name__)


class PPGSignalEnhancer:
    """Enhanced PPG signal processing with quality assessment"""

    def __init__(self, sampling_rate=30):
        """Initialize signal enhancer"""
        self.fs = sampling_rate
        self.quality_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'acceptable': 0.4,
            'poor': 0.2
        }

        # Load calibration if available
        self.calibration_factor = self._load_calibration()

    def enhance_signal(self, ppg_signal):
        """
        Apply enhanced signal processing pipeline

        Args:
            ppg_signal: Raw PPG signal array

        Returns:
            Dict with enhanced signal and quality metrics
        """
        if len(ppg_signal) < 30:
            return {
                'enhanced_signal': ppg_signal,
                'quality_score': 0,
                'quality_level': 'poor',
                'is_acceptable': False
            }

        # Step 1: Remove DC component and normalize
        signal_detrended = signal.detrend(ppg_signal)
        signal_normalized = self._normalize_signal(signal_detrended)

        # Step 2: Apply bandpass filter (0.5-4 Hz for heart rate)
        signal_filtered = self._bandpass_filter(signal_normalized)

        # Step 3: Smooth signal to reduce high-frequency noise
        signal_smoothed = self._smooth_signal(signal_filtered)

        # Step 4: Assess signal quality
        quality_metrics = self._assess_signal_quality(signal_smoothed)

        # Step 5: Apply adaptive filtering based on quality
        if quality_metrics['snr'] < 5:
            # Poor SNR - apply stronger filtering
            signal_enhanced = self._adaptive_filter(signal_smoothed, strength='strong')
        elif quality_metrics['snr'] < 10:
            # Moderate SNR - apply medium filtering
            signal_enhanced = self._adaptive_filter(signal_smoothed, strength='medium')
        else:
            # Good SNR - light filtering only
            signal_enhanced = self._adaptive_filter(signal_smoothed, strength='light')

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(quality_metrics)
        quality_level = self._get_quality_level(quality_score)

        return {
            'enhanced_signal': signal_enhanced,
            'raw_signal': ppg_signal,
            'quality_score': quality_score,
            'quality_level': quality_level,
            'quality_metrics': quality_metrics,
            'is_acceptable': bool(quality_score >= self.quality_thresholds['acceptable']),  # Ensure Python bool
            'confidence': self._score_to_confidence(quality_score)
        }

    def _normalize_signal(self, signal):
        """Normalize signal to [-1, 1] range"""
        if np.std(signal) == 0:
            return signal
        return (signal - np.mean(signal)) / np.std(signal)

    def _bandpass_filter(self, signal_data):
        """Apply bandpass filter for heart rate frequencies"""
        try:
            # Heart rate range: 30-180 BPM = 0.5-3 Hz
            nyquist = self.fs / 2
            low = 0.5 / nyquist
            high = min(4.0 / nyquist, 0.99)  # Ensure within Nyquist

            # Use lower order for stability
            order = 2
            b, a = butter(order, [low, high], btype='band')

            # Check filter stability
            if np.any(np.abs(np.roots(a)) >= 1):
                logger.warning("Filter unstable, using fallback")
                return signal_data

            filtered = filtfilt(b, a, signal_data)
            return filtered

        except Exception as e:
            logger.error(f"Bandpass filter failed: {e}")
            return signal_data

    def _smooth_signal(self, signal_data, window_size=5):
        """Apply moving average smoothing"""
        if len(signal_data) < window_size:
            return signal_data

        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(signal_data, kernel, mode='same')
        return smoothed

    def _adaptive_filter(self, signal_data, strength='medium'):
        """Apply adaptive filtering based on signal quality"""
        if strength == 'strong':
            # Heavy filtering for poor signals
            return self._smooth_signal(signal_data, window_size=7)
        elif strength == 'medium':
            # Moderate filtering
            return self._smooth_signal(signal_data, window_size=5)
        else:
            # Light filtering for good signals
            return self._smooth_signal(signal_data, window_size=3)

    def _assess_signal_quality(self, signal_data):
        """Comprehensive signal quality assessment"""
        metrics = {}

        # 1. Signal-to-Noise Ratio (SNR)
        metrics['snr'] = self._calculate_snr(signal_data)

        # 2. Peak consistency
        metrics['peak_consistency'] = self._assess_peak_consistency(signal_data)

        # 3. Signal stability (low variance in peak amplitudes)
        metrics['stability'] = self._assess_stability(signal_data)

        # 4. Spectral purity (how dominant is the heart rate frequency)
        metrics['spectral_purity'] = self._assess_spectral_purity(signal_data)

        # 5. Motion artifacts (sudden jumps)
        metrics['motion_score'] = self._detect_motion_artifacts(signal_data)

        return metrics

    def _calculate_snr(self, signal_data):
        """Calculate Signal-to-Noise Ratio"""
        if len(signal_data) < 60:
            return 0

        # Use frequency domain for SNR
        freqs, psd = welch(signal_data, self.fs, nperseg=min(len(signal_data)//2, 64))

        # Heart rate band (0.5-3 Hz)
        hr_band = (freqs >= 0.5) & (freqs <= 3.0)
        noise_band = (freqs > 3.0) & (freqs < self.fs/2)

        if np.any(hr_band) and np.any(noise_band):
            signal_power = np.sum(psd[hr_band])
            noise_power = np.sum(psd[noise_band])

            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
                return max(0, snr_db)

        return 0

    def _assess_peak_consistency(self, signal_data):
        """Assess consistency of detected peaks"""
        try:
            # Find peaks
            peaks, properties = find_peaks(signal_data,
                                          height=np.mean(signal_data),
                                          distance=self.fs//3)  # Min 0.33s between peaks

            if len(peaks) < 3:
                return 0

            # Calculate inter-peak intervals
            intervals = np.diff(peaks)

            if len(intervals) < 2:
                return 0

            # Consistency score based on coefficient of variation
            cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 1
            consistency = max(0, 1 - cv)  # Lower CV = higher consistency

            return consistency

        except Exception as e:
            logger.error(f"Peak consistency assessment failed: {e}")
            return 0

    def _assess_stability(self, signal_data):
        """Assess signal amplitude stability"""
        try:
            # Find peaks for amplitude analysis
            peaks, properties = find_peaks(signal_data,
                                          height=np.mean(signal_data))

            if len(peaks) < 3 or 'peak_heights' not in properties:
                return 0

            amplitudes = properties['peak_heights']

            # Stability based on amplitude variation
            cv = np.std(amplitudes) / np.mean(amplitudes) if np.mean(amplitudes) > 0 else 1
            stability = max(0, 1 - cv/2)  # Normalize to 0-1

            return stability

        except Exception:
            return 0

    def _assess_spectral_purity(self, signal_data):
        """Assess how dominant the heart rate frequency is"""
        try:
            if len(signal_data) < 60:
                return 0

            # Get power spectral density
            freqs, psd = welch(signal_data, self.fs, nperseg=min(len(signal_data)//2, 64))

            # Find dominant frequency in HR range
            hr_band = (freqs >= 0.5) & (freqs <= 3.0)

            if not np.any(hr_band):
                return 0

            hr_psd = psd[hr_band]
            max_power = np.max(hr_psd)
            total_power = np.sum(psd)

            if total_power > 0:
                purity = max_power / total_power
                return min(1.0, purity * 5)  # Scale up as it's usually small

            return 0

        except Exception:
            return 0

    def _detect_motion_artifacts(self, signal_data):
        """Detect motion artifacts (sudden jumps)"""
        try:
            # Calculate first derivative
            diff = np.diff(signal_data)

            # Detect outliers (sudden changes)
            threshold = 3 * np.std(diff)
            artifacts = np.sum(np.abs(diff) > threshold)

            # Normalize to 0-1 (fewer artifacts = higher score)
            artifact_ratio = artifacts / len(diff)
            motion_score = max(0, 1 - artifact_ratio * 10)

            return motion_score

        except Exception:
            return 0

    def _calculate_quality_score(self, metrics):
        """Calculate overall quality score from individual metrics"""
        weights = {
            'snr': 0.3,
            'peak_consistency': 0.25,
            'stability': 0.2,
            'spectral_purity': 0.15,
            'motion_score': 0.1
        }

        score = 0
        for metric, weight in weights.items():
            if metric in metrics:
                # Normalize SNR to 0-1 range
                if metric == 'snr':
                    normalized = min(1.0, metrics[metric] / 20)  # 20 dB = excellent
                else:
                    normalized = metrics[metric]

                score += normalized * weight

        return round(score, 2)

    def _get_quality_level(self, score):
        """Convert quality score to level"""
        if score >= self.quality_thresholds['excellent']:
            return 'excellent'
        elif score >= self.quality_thresholds['good']:
            return 'good'
        elif score >= self.quality_thresholds['acceptable']:
            return 'acceptable'
        else:
            return 'poor'

    def _score_to_confidence(self, score):
        """Convert quality score to confidence percentage"""
        # Map 0-1 score to 30-95% confidence
        # Never give 100% confidence for legal/medical reasons
        confidence = 30 + score * 65
        return round(confidence, 0)

    def extract_heart_rate(self, enhanced_signal, quality_score):
        """
        Extract heart rate from enhanced signal with confidence

        Args:
            enhanced_signal: Enhanced PPG signal
            quality_score: Signal quality score

        Returns:
            Dict with HR and confidence
        """
        try:
            # Find peaks with adaptive threshold
            mean_val = np.mean(enhanced_signal)
            std_val = np.std(enhanced_signal)

            # Adjust peak detection based on quality
            if quality_score > 0.7:
                # High quality - strict peak detection
                min_height = mean_val + 0.3 * std_val
                min_distance = self.fs * 0.5  # Min 0.5s between beats (120 BPM max)
            else:
                # Lower quality - more lenient
                min_height = mean_val
                min_distance = self.fs * 0.4  # Min 0.4s between beats (150 BPM max)

            peaks, properties = find_peaks(enhanced_signal,
                                          height=min_height,
                                          distance=min_distance)

            if len(peaks) < 3:
                # Fall back to FFT method
                return self._extract_hr_fft(enhanced_signal, quality_score)

            # Calculate HR from peak intervals
            intervals = np.diff(peaks) / self.fs  # Convert to seconds
            heart_rates = 60 / intervals  # Convert to BPM

            # Remove outliers
            hr_filtered = heart_rates[(heart_rates > 40) & (heart_rates < 180)]

            if len(hr_filtered) == 0:
                return self._extract_hr_fft(enhanced_signal, quality_score)

            # Use median for robustness
            hr_median = np.median(hr_filtered)

            # Apply calibration factor (dynamically loaded or default)
            hr_calibrated = hr_median * self.calibration_factor

            # Log for debugging
            logger.info(f"HR Detection: Raw={hr_median:.1f}, Calibrated={hr_calibrated:.1f}, Peaks={len(peaks)}")

            # Confidence based on consistency and quality
            hr_std = np.std(hr_filtered)
            consistency = 1 - (hr_std / hr_median) if hr_median > 0 else 0
            confidence = (quality_score * 0.6 + consistency * 0.4) * 100

            return {
                'heart_rate': round(hr_calibrated, 1),
                'confidence': round(confidence, 0),
                'method': 'peak_detection',
                'peak_count': len(peaks),
                'variability': round(hr_std, 1),
                'raw_hr': round(hr_median, 1)  # Include raw for comparison
            }

        except Exception as e:
            logger.error(f"HR extraction failed: {e}")
            return self._extract_hr_fft(enhanced_signal, quality_score)

    def _extract_hr_fft(self, signal_data, quality_score):
        """Extract HR using FFT method as fallback"""
        try:
            # Compute FFT
            fft = np.fft.rfft(signal_data)
            freqs = np.fft.rfftfreq(len(signal_data), 1/self.fs)

            # Focus on heart rate frequencies (0.7-3 Hz = 42-180 BPM)
            hr_band = (freqs >= 0.7) & (freqs <= 3.0)

            if not np.any(hr_band):
                return {
                    'heart_rate': 75,
                    'confidence': 30,
                    'method': 'fallback',
                    'error': 'No valid frequency band'
                }

            # Find dominant frequency
            power = np.abs(fft[hr_band])**2
            freqs_hr = freqs[hr_band]
            dominant_freq = freqs_hr[np.argmax(power)]

            # Convert to BPM
            heart_rate = dominant_freq * 60

            # Lower confidence for FFT method
            confidence = quality_score * 70

            return {
                'heart_rate': round(heart_rate, 1),
                'confidence': round(confidence, 0),
                'method': 'fft',
                'dominant_freq': round(dominant_freq, 2)
            }

        except Exception as e:
            logger.error(f"FFT HR extraction failed: {e}")
            return {
                'heart_rate': 75,
                'confidence': 30,
                'method': 'fallback',
                'error': str(e)
            }

    def _load_calibration(self):
        """Load calibration factor from config file if available"""
        default_factor = 0.75  # Very aggressive calibration (25% reduction) to match Apple Watch

        try:
            # Check for calibration file
            calibration_file = 'hr_calibration.json'
            if os.path.exists(calibration_file):
                with open(calibration_file, 'r') as f:
                    config = json.load(f)
                    factor = config.get('calibration_factor', default_factor)
                    logger.info(f"Loaded HR calibration factor: {factor:.3f} from {calibration_file}")
                    return factor
        except Exception as e:
            logger.warning(f"Could not load calibration: {e}")

        logger.info(f"Using default HR calibration factor: {default_factor:.3f}")
        return default_factor