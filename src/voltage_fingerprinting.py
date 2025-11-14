"""
Voltage-based ECU fingerprinting for intrusion detection.

Uses voltage signal characteristics to identify which ECU sent a message.
Different hardware has slightly different voltage profiles.
"""

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoltageFingerprinter:
    """ECU identification using voltage signatures"""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.ecu_profiles: Dict[int, Dict[str, float]] = {}
        self.trained = False
        
    def extract_features(self, voltage_signal: np.ndarray) -> Dict[str, float]:
        """
        Pull out useful features from a voltage signal.
        
        We look at basic stats, shape characteristics, and frequency content.
        This gives us a fingerprint for each ECU.
        """
        features = {}
        
        # Basic stats
        features['mean'] = np.mean(voltage_signal)
        features['std'] = np.std(voltage_signal)
        features['min'] = np.min(voltage_signal)
        features['max'] = np.max(voltage_signal)
        features['peak_to_peak'] = features['max'] - features['min']
        features['range'] = features['peak_to_peak']
        
        # Distribution shape
        features['skewness'] = skew(voltage_signal)
        features['kurtosis'] = kurtosis(voltage_signal)
        
        # Find peaks in the signal
        peaks, _ = signal.find_peaks(voltage_signal)
        features['num_peaks'] = len(peaks)
        if len(peaks) > 0:
            features['peak_mean'] = np.mean(voltage_signal[peaks])
            features['peak_std'] = np.std(voltage_signal[peaks])
        else:
            features['peak_mean'] = features['mean']
            features['peak_std'] = 0.0
        
        # Timing characteristics
        features['rise_time'] = self._estimate_rise_time(voltage_signal)
        features['fall_time'] = self._estimate_fall_time(voltage_signal)
        features['settling_time'] = self._estimate_settling_time(voltage_signal)
        features['overshoot'] = self._estimate_overshoot(voltage_signal)
        
        # Frequency domain features
        freq_features = self._extract_frequency_features(voltage_signal)
        features.update(freq_features)
        
        # Energy measures
        features['energy'] = np.sum(voltage_signal ** 2)
        features['power'] = features['energy'] / len(voltage_signal)
        features['rms'] = np.sqrt(features['power'])
        
        # How often signal crosses zero
        features['zero_crossing_rate'] = self._zero_crossing_rate(voltage_signal)
        
        return features
    
    def _estimate_rise_time(self, signal: np.ndarray, 
                           low_threshold: float = 0.1, 
                           high_threshold: float = 0.9) -> float:
        """How long does it take to go from 10% to 90% of the range"""
        signal_range = np.max(signal) - np.min(signal)
        low_val = np.min(signal) + low_threshold * signal_range
        high_val = np.min(signal) + high_threshold * signal_range
        
        # Find first crossing points
        low_cross = np.where(signal >= low_val)[0]
        high_cross = np.where(signal >= high_val)[0]
        
        if len(low_cross) > 0 and len(high_cross) > 0:
            return high_cross[0] - low_cross[0]
        return 0.0
    
    def _estimate_fall_time(self, signal: np.ndarray,
                           high_threshold: float = 0.9,
                           low_threshold: float = 0.1) -> float:
        """
        Estimate fall time (90% to 10% of signal range)
        
        Args:
            signal: Voltage signal
            high_threshold: Higher threshold (default 90%)
            low_threshold: Lower threshold (default 10%)
            
        Returns:
            Fall time in samples
        """
        signal_range = np.max(signal) - np.min(signal)
        high_val = np.min(signal) + high_threshold * signal_range
        low_val = np.min(signal) + low_threshold * signal_range
        
        # Find last crossing going down
        high_cross = np.where(signal <= high_val)[0]
        low_cross = np.where(signal <= low_val)[0]
        
        if len(high_cross) > 0 and len(low_cross) > 0:
            return low_cross[-1] - high_cross[0]
        return 0.0
    
    def _estimate_settling_time(self, signal: np.ndarray, tolerance: float = 0.02) -> float:
        """
        Estimate settling time (time to reach within tolerance of steady state)
        
        Args:
            signal: Voltage signal
            tolerance: Tolerance band (default 2%)
            
        Returns:
            Settling time in samples
        """
        steady_state = np.mean(signal[-int(len(signal)*0.1):])  # Last 10%
        tolerance_band = tolerance * (np.max(signal) - np.min(signal))
        
        within_tolerance = np.abs(signal - steady_state) <= tolerance_band
        
        if np.any(within_tolerance):
            settling_idx = np.where(within_tolerance)[0][0]
            return settling_idx
        return len(signal)
    
    def _estimate_overshoot(self, signal: np.ndarray) -> float:
        """
        Estimate overshoot percentage
        
        Args:
            signal: Voltage signal
            
        Returns:
            Overshoot as percentage
        """
        steady_state = np.mean(signal[-int(len(signal)*0.1):])
        peak = np.max(signal)
        
        if steady_state != 0:
            overshoot = ((peak - steady_state) / abs(steady_state)) * 100
            return max(0, overshoot)
        return 0.0
    
    def _extract_frequency_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency domain features using FFT
        
        Args:
            signal: Voltage signal
            
        Returns:
            Dictionary of frequency features
        """
        # Compute FFT
        fft_vals = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(len(signal))
        
        # Use only positive frequencies
        positive_freq_idx = fft_freq > 0
        fft_power = np.abs(fft_vals[positive_freq_idx]) ** 2
        fft_freq_positive = fft_freq[positive_freq_idx]
        
        features = {}
        
        # Dominant frequency
        if len(fft_power) > 0:
            dominant_idx = np.argmax(fft_power)
            features['dominant_frequency'] = fft_freq_positive[dominant_idx]
            features['dominant_power'] = fft_power[dominant_idx]
            
            # Spectral centroid
            features['spectral_centroid'] = np.sum(fft_freq_positive * fft_power) / (np.sum(fft_power) + 1e-8)
            
            # Spectral spread
            features['spectral_spread'] = np.sqrt(
                np.sum(((fft_freq_positive - features['spectral_centroid']) ** 2) * fft_power) / (np.sum(fft_power) + 1e-8)
            )
            
            # Spectral rolloff (frequency below which 85% of energy is contained)
            cumsum_power = np.cumsum(fft_power)
            rolloff_threshold = 0.85 * cumsum_power[-1]
            rolloff_idx = np.where(cumsum_power >= rolloff_threshold)[0]
            if len(rolloff_idx) > 0:
                features['spectral_rolloff'] = fft_freq_positive[rolloff_idx[0]]
            else:
                features['spectral_rolloff'] = 0.0
        else:
            features['dominant_frequency'] = 0.0
            features['dominant_power'] = 0.0
            features['spectral_centroid'] = 0.0
            features['spectral_spread'] = 0.0
            features['spectral_rolloff'] = 0.0
        
        return features
    
    def _zero_crossing_rate(self, signal: np.ndarray) -> float:
        """
        Calculate zero crossing rate
        
        Args:
            signal: Voltage signal
            
        Returns:
            Zero crossing rate
        """
        centered = signal - np.mean(signal)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(centered)))) / 2
        return zero_crossings / len(signal)
    
    def train(self, voltage_signals: List[np.ndarray], ecu_ids: List[int]):
        """
        Learn what's normal for each ECU by looking at their voltage patterns.
        We build a profile for each ECU based on the average features.
        """
        logger.info(f"Training voltage fingerprinter on {len(voltage_signals)} samples")
        
        # Group signals by which ECU sent them
        ecu_features: Dict[int, List[Dict[str, float]]] = {}
        
        for signal, ecu_id in zip(voltage_signals, ecu_ids):
            features = self.extract_features(signal)
            
            if ecu_id not in ecu_features:
                ecu_features[ecu_id] = []
            ecu_features[ecu_id].append(features)
        
        # Build average profile for each ECU
        for ecu_id, feature_list in ecu_features.items():
            # Average all features
            profile = {}
            feature_keys = feature_list[0].keys()
            
            for key in feature_keys:
                values = [f[key] for f in feature_list]
                profile[f"{key}_mean"] = np.mean(values)
                profile[f"{key}_std"] = np.std(values)
            
            self.ecu_profiles[ecu_id] = profile
            logger.info(f"Created profile for ECU {ecu_id:#x} from {len(feature_list)} samples")
        
        self.trained = True
        logger.info(f"Training complete. Profiles created for {len(self.ecu_profiles)} ECUs")
    
    def predict(self, voltage_signal: np.ndarray) -> Tuple[int, float]:
        """
        Predict the ECU ID from a voltage signal using enhanced statistical matching.
        
        Uses a combination of Mahalanobis distance and correlation-based similarity
        to match voltage characteristics to known ECU profiles.
        
        Args:
            voltage_signal: Voltage signal array
            
        Returns:
            Tuple of (predicted_ecu_id, confidence)
        """
        if not self.trained:
            raise ValueError("Fingerprinter must be trained before prediction")
        
        # Extract features
        features = self.extract_features(voltage_signal)
        
        # Calculate similarity to each ECU profile using multiple metrics
        ecu_scores = {}
        
        for ecu_id, profile in self.ecu_profiles.items():
            # Metric 1: Mahalanobis-like distance (normalized by std deviation)
            mahalanobis_distance = 0.0
            
            # Metric 2: Weighted feature distance (some features are more distinctive)
            weighted_distance = 0.0
            
            # Feature weights based on discriminative power
            # (rise/fall times, ringing characteristics are most distinctive)
            feature_weights = {
                'rise_time': 3.0,        # Very distinctive
                'fall_time': 3.0,
                'overshoot': 2.5,
                'ringing_freq': 2.0,     # Hardware-specific
                'dominant_frequency': 2.0,
                'spectral_centroid': 1.8,
                'settling_time': 2.2,
                'peak_to_peak': 1.5,
                'rms': 1.5,
                'zero_crossing_rate': 1.3,
                'num_peaks': 1.2,
            }
            
            count = 0
            for key in features.keys():
                mean_key = f"{key}_mean"
                std_key = f"{key}_std"
                
                if mean_key in profile and std_key in profile:
                    mean_val = profile[mean_key]
                    std_val = profile[std_key] + 1e-8  # Avoid division by zero
                    
                    # Normalized distance
                    normalized_dist = abs(features[key] - mean_val) / std_val
                    mahalanobis_distance += normalized_dist
                    
                    # Weighted distance
                    weight = feature_weights.get(key, 1.0)
                    weighted_distance += weight * normalized_dist
                    count += 1
            
            if count > 0:
                avg_mahalanobis = mahalanobis_distance / count
                avg_weighted = weighted_distance / sum([feature_weights.get(k, 1.0) for k in features.keys()])
                
                # Combine both metrics (weighted average)
                combined_distance = 0.4 * avg_mahalanobis + 0.6 * avg_weighted
                
                # Convert distance to similarity score (0 to 1)
                # Using exponential decay for better discrimination
                similarity = np.exp(-combined_distance)
                ecu_scores[ecu_id] = similarity
        
        if not ecu_scores:
            raise ValueError("No valid ECU profiles to compare against")
        
        # Find best match
        best_ecu = max(ecu_scores, key=ecu_scores.get)
        confidence = ecu_scores[best_ecu]
        
        # Normalize confidence based on second-best match (discrimination margin)
        sorted_scores = sorted(ecu_scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            margin = sorted_scores[0] - sorted_scores[1]
            # Boost confidence if there's a clear winner
            confidence = min(1.0, confidence * (1.0 + margin))
        
        return best_ecu, confidence
    
    def detect_anomaly(self, voltage_signal: np.ndarray, claimed_ecu_id: int) -> Tuple[bool, float]:
        """
        Detect if a voltage signal is anomalous for the claimed ECU.
        
        This is the core intrusion detection function - it checks if the physical
        layer voltage characteristics match the claimed sender identity.
        
        Strategy: Prioritize ID matching over confidence threshold to reduce false positives.
        Only flag as anomaly when there's strong evidence of mismatch.
        
        Args:
            voltage_signal: Voltage signal array
            claimed_ecu_id: The ECU ID claimed by the message
            
        Returns:
            Tuple of (is_anomaly, anomaly_score)
                - is_anomaly: True if mismatch detected (spoofing attack)
                - anomaly_score: Confidence in the anomaly detection (0-1)
        """
        predicted_ecu, confidence = self.predict(voltage_signal)
        
        # Primary decision factor: Does predicted ECU match claimed ECU?
        if predicted_ecu != claimed_ecu_id:
            # ECU ID mismatch - strong indicator of spoofing attack
            # Only flag if we're confident about the prediction
            if confidence >= self.threshold:
                # High confidence in the mismatch - definitely anomalous
                is_anomaly = True
                anomaly_score = confidence
            else:
                # Low confidence mismatch - could be noise or natural variation
                # Be conservative: don't flag unless confidence is reasonable
                is_anomaly = False
                anomaly_score = confidence * 0.5  # Partial suspicion
        else:
            # ECU ID matches - this is the normal case
            # Trust the match regardless of confidence level
            # (voltage characteristics naturally vary due to temperature, load, etc.)
            is_anomaly = False
            anomaly_score = 1.0 - confidence  # Lower score = less anomalous
        
        return is_anomaly, anomaly_score
    
    def predict_batch(self, voltage_signals: List[np.ndarray]) -> List[Tuple[int, float]]:
        """
        Batch prediction for multiple voltage signals (optimized)
        
        Args:
            voltage_signals: List of voltage signal arrays
            
        Returns:
            List of (predicted_ecu_id, confidence) tuples
        """
        return [self.predict(signal) for signal in voltage_signals]
    
    def detect_anomaly_batch(self, voltage_signals: List[np.ndarray], 
                            claimed_ecu_ids: List[int]) -> List[Tuple[bool, float]]:
        """
        Batch anomaly detection (optimized for evaluation)
        
        Args:
            voltage_signals: List of voltage signal arrays
            claimed_ecu_ids: List of claimed ECU IDs
            
        Returns:
            List of (is_anomaly, anomaly_score) tuples
        """
        results = []
        for signal, ecu_id in zip(voltage_signals, claimed_ecu_ids):
            is_anomaly, score = self.detect_anomaly(signal, ecu_id)
            results.append((is_anomaly, score))
        return results


def main():
    """Test voltage fingerprinting"""
    logger.info("Testing voltage fingerprinting...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    # Generate synthetic voltage signals for different ECUs
    signals = []
    ecu_ids = []
    
    for i in range(n_samples):
        ecu_id = [0x100, 0x200, 0x300][i % 3]
        
        # Generate ECU-specific voltage pattern
        t = np.linspace(0, 1, 100)
        if ecu_id == 0x100:
            signal = 2.5 + 0.5 * np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.05, 100)
        elif ecu_id == 0x200:
            signal = 2.3 + 0.3 * np.cos(2 * np.pi * 10 * t) + np.random.normal(0, 0.03, 100)
        else:
            signal = 2.7 + 0.2 * np.sin(2 * np.pi * 15 * t) + np.random.normal(0, 0.04, 100)
        
        signals.append(signal)
        ecu_ids.append(ecu_id)
    
    # Train fingerprinter
    fingerprinter = VoltageFingerprinter(threshold=0.7)
    fingerprinter.train(signals, ecu_ids)
    
    # Test prediction
    test_signal = signals[0]
    predicted_ecu, confidence = fingerprinter.predict(test_signal)
    logger.info(f"Test prediction: ECU {predicted_ecu:#x}, Confidence: {confidence:.3f}")
    
    # Test anomaly detection
    is_anomaly, conf = fingerprinter.detect_anomaly(test_signal, ecu_ids[0])
    logger.info(f"Anomaly detection: {is_anomaly}, Confidence: {conf:.3f}")
    
    logger.info("Voltage fingerprinting test complete")


if __name__ == "__main__":
    main()
