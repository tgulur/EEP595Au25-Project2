"""
Baseline IDS models for comparison
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from collections import defaultdict, deque
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineIDS:
    """Base class for our baseline methods"""
    
    def __init__(self, name: str):
        self.name = name
        self.trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions and scores"""
        raise NotImplementedError


class TimingBasedIDS(BaselineIDS):
    """IDS based on message timing patterns. Attacks mess with timing"""
    
    def __init__(self, threshold: float = 0.05):
        super().__init__(name="Timing-Based")
        self.threshold = threshold
        self.message_intervals: Dict[int, List[float]] = defaultdict(list)
        self.expected_intervals: Dict[int, Tuple[float, float]] = {}
        self.clock_skew: Dict[int, float] = {}
    
    def train(self, timestamps: np.ndarray, can_ids: np.ndarray, y: np.ndarray):
        """Learn normal timing patterns for each CAN ID"""
        logger.info("Training timing IDS...")
        
        # Learn intervals for each CAN ID
        for i in range(1, len(timestamps)):
            if y[i-1] == 0 and y[i] == 0:  # Both normal
                can_id = can_ids[i]
                prev_can_id = can_ids[i-1]
                
                if can_id == prev_can_id:
                    interval = timestamps[i] - timestamps[i-1]
                    if interval > 0:  # Valid interval
                        self.message_intervals[can_id].append(interval)
        
        # Calculate expected intervals (mean and std) for each CAN ID
        for can_id, intervals in self.message_intervals.items():
            if len(intervals) >= 10:  # Need enough samples
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                self.expected_intervals[can_id] = (mean_interval, std_interval)
                
                # Estimate clock skew (linear trend in intervals)
                if len(intervals) > 20:
                    x = np.arange(len(intervals))
                    slope = np.polyfit(x, intervals, 1)[0]
                    self.clock_skew[can_id] = slope
        
        logger.info(f"Learned timing profiles for {len(self.expected_intervals)} CAN IDs")
        self.trained = True
    
    def predict(self, timestamps: np.ndarray, can_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies based on timing deviations
        
        Args:
            timestamps: Message timestamps
            can_ids: CAN IDs
            
        Returns:
            Tuple of (predictions, anomaly_scores)
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = []
        scores = []
        
        # Track recent intervals for each CAN ID to detect anomalies
        recent_intervals: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        
        for i in range(1, len(timestamps)):
            can_id = can_ids[i]
            prev_can_id = can_ids[i-1]
            
            # Check if this CAN ID has expected intervals
            if can_id in self.expected_intervals:
                interval = timestamps[i] - timestamps[i-1]
                expected_mean, expected_std = self.expected_intervals[can_id]
                
                # Calculate deviation from expected interval
                if expected_std > 0:
                    deviation = abs(interval - expected_mean) / expected_std
                else:
                    deviation = abs(interval - expected_mean) if expected_mean > 0 else 0.0
                
                # Store recent interval for this CAN ID
                recent_intervals[can_id].append(interval)
                
                # Check clock skew if available
                if can_id in self.clock_skew and len(recent_intervals[can_id]) > 5:
                    # Calculate expected interval with skew
                    expected_with_skew = expected_mean + self.clock_skew[can_id] * len(recent_intervals[can_id])
                    skew_deviation = abs(interval - expected_with_skew) / (expected_std + 1e-6)
                    deviation = max(deviation, skew_deviation)
                
                # Normalize deviation to 0-1 range (use 2-sigma for more sensitive detection)
                anomaly_score = min(1.0, deviation / 2.0)  # 2-sigma rule (more sensitive)
                
                # Also check for sudden changes in interval pattern
                if len(recent_intervals[can_id]) >= 5:
                    recent_mean = np.mean(list(recent_intervals[can_id]))
                    recent_std = np.std(list(recent_intervals[can_id]))
                    if recent_std > 0:
                        pattern_deviation = abs(recent_mean - expected_mean) / (expected_std + 1e-6)
                        anomaly_score = max(anomaly_score, min(1.0, pattern_deviation / 2.0))
                
                # Lower threshold for better attack detection
                is_anomaly = anomaly_score > (self.threshold * 0.5)  # More sensitive
            else:
                # Unknown CAN ID - could be suspicious
                if can_id not in self.expected_intervals and len(self.expected_intervals) > 0:
                    # New/unseen CAN ID might be an attack
                    anomaly_score = 0.6
                    is_anomaly = True
                else:
                    anomaly_score = 0.0
                    is_anomaly = False
            
            predictions.append(int(is_anomaly))
            scores.append(anomaly_score)
        
        # First message defaults to normal
        predictions.insert(0, 0)
        scores.insert(0, 0.0)
        
        return np.array(predictions), np.array(scores)


class FrequencyBasedIDS(BaselineIDS):
    """IDS that looks for weird message frequencies. DoS attacks spike the rate"""
    
    def __init__(self, window_size: int = 100, threshold: float = 3.0):
        super().__init__(name="Frequency-Based")
        self.window_size = window_size
        self.threshold = threshold  # Increased from 2.0 to 3.0 to reduce false positives
        self.expected_frequencies: Dict[int, Tuple[float, float]] = {}
    
    def train(self, timestamps: np.ndarray, can_ids: np.ndarray, y: np.ndarray):
        """Learn normal message rates for each CAN ID"""
        logger.info("Training frequency IDS...")
        
        # Calculate frequencies in sliding windows
        frequencies: Dict[int, List[float]] = defaultdict(list)
        
        for i in range(len(timestamps) - self.window_size):
            window_data = can_ids[i:i+self.window_size]
            window_labels = y[i:i+self.window_size]
            
            # Only use normal windows
            if np.all(window_labels == 0):
                window_duration = timestamps[i+self.window_size-1] - timestamps[i]
                
                if window_duration > 0:
                    for can_id in np.unique(window_data):
                        count = np.sum(window_data == can_id)
                        freq = count / window_duration  # messages per second
                        frequencies[can_id].append(freq)
        
        # Calculate expected frequencies
        for can_id, freq_list in frequencies.items():
            if len(freq_list) >= 10:
                mean_freq = np.mean(freq_list)
                std_freq = np.std(freq_list)
                self.expected_frequencies[can_id] = (mean_freq, std_freq)
        
        logger.info(f"Learned frequency profiles for {len(self.expected_frequencies)} CAN IDs")
        self.trained = True
    
    def predict(self, timestamps: np.ndarray, can_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies based on frequency deviations
        
        Args:
            timestamps: Message timestamps
            can_ids: CAN IDs
            
        Returns:
            Tuple of (predictions, anomaly_scores)
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = []
        scores = []
        
        # Use sliding window
        for i in range(len(timestamps)):
            start_idx = max(0, i - self.window_size + 1)
            window_data = can_ids[start_idx:i+1]
            window_timestamps = timestamps[start_idx:i+1]
            
            if len(window_data) >= self.window_size:
                window_duration = window_timestamps[-1] - window_timestamps[0]
                
                if window_duration > 0:
                    # Check frequency for current message's CAN ID
                    can_id = can_ids[i]
                    
                    if can_id in self.expected_frequencies:
                        count = np.sum(window_data == can_id)
                        current_freq = count / window_duration
                        
                        expected_mean, expected_std = self.expected_frequencies[can_id]
                        
                        # Calculate deviation (use relative deviation for better sensitivity)
                        if expected_std > 0:
                            # Z-score based deviation
                            deviation = abs(current_freq - expected_mean) / expected_std
                        else:
                            # If no variance, use relative difference
                            if expected_mean > 0:
                                deviation = abs(current_freq - expected_mean) / expected_mean
                            else:
                                deviation = abs(current_freq) if current_freq > 0 else 0.0
                        
                        # Normalize to 0-1 range
                        anomaly_score = min(1.0, deviation / self.threshold)
                        
                        # Use stricter threshold: need significant deviation
                        # Also check for both too high (DoS) and too low (blocking) frequencies
                        is_anomaly = deviation > self.threshold
                        
                        # Additional check: sudden spike (DoS attack)
                        if current_freq > expected_mean * 2.0:
                            is_anomaly = True
                            anomaly_score = min(1.0, (current_freq / expected_mean) / 5.0)
                    else:
                        # Unknown CAN ID - moderate suspicion
                        anomaly_score = 0.3
                        is_anomaly = False
                else:
                    anomaly_score = 0.0
                    is_anomaly = False
            else:
                anomaly_score = 0.0
                is_anomaly = False
            
            predictions.append(int(is_anomaly))
            scores.append(anomaly_score)
        
        return np.array(predictions), np.array(scores)


class RuleBasedIDS(BaselineIDS):
    """Simple rule-based IDS. Checks for invalid IDs, wrong lengths, etc."""
    
    def __init__(self):
        super().__init__(name="Rule-Based")
        self.valid_can_ids: set = set()
        self.expected_dlc: Dict[int, int] = {}
    
    def train(self, can_ids: np.ndarray, dlc: np.ndarray, y: np.ndarray):
        """Learn what valid CAN IDs and data lengths look like"""
        logger.info("Training rule-based IDS...")
        
        # Learn valid IDs from normal traffic
        self.valid_can_ids = set(can_ids[y == 0])
        
        # Learn expected data length for each ID
        for can_id in self.valid_can_ids:
            mask = (can_ids == can_id) & (y == 0)
            if np.any(mask):
                # Most common length for this ID
                unique, counts = np.unique(dlc[mask], return_counts=True)
                self.expected_dlc[can_id] = unique[np.argmax(counts)]
        
        logger.info(f"Learned {len(self.valid_can_ids)} valid CAN IDs")
        self.trained = True
    
    def predict(self, can_ids: np.ndarray, dlc: np.ndarray, 
                data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Check rules and flag anything suspicious"""
        if not self.trained:
            raise ValueError("Need to train first")
        
        predictions = []
        scores = []
        
        for i in range(len(can_ids)):
            anomaly_flags = 0
            score = 0.0
            
            # Check 1: Is this a known CAN ID?
            if can_ids[i] not in self.valid_can_ids:
                anomaly_flags += 1
                score += 0.5
            
            # Check 2: Does the data length match what we expect?
            if can_ids[i] in self.expected_dlc:
                if dlc[i] != self.expected_dlc[can_ids[i]]:
                    anomaly_flags += 1
                    score += 0.3
            
            # Check 3: Suspicious data patterns
            data_bytes = data[i] if len(data[i]) > 0 else [0]
            
            # All zeros or all ones
            if np.all(np.array(data_bytes) == 0) or np.all(np.array(data_bytes) == 255):
                anomaly_flags += 1
                score += 0.2
            
            is_anomaly = anomaly_flags > 0
            score = min(1.0, score)
            
            predictions.append(int(is_anomaly))
            scores.append(score)
        
        return np.array(predictions), np.array(scores)


def main():
    """Test baseline models"""
    logger.info("Testing baseline models...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    timestamps = np.cumsum(np.random.uniform(0.001, 0.02, n_samples))
    can_ids = np.random.choice([0x100, 0x200, 0x300], n_samples)
    dlc = np.random.choice([8], n_samples)
    data = [list(np.random.randint(0, 256, 8)) for _ in range(n_samples)]
    labels = np.zeros(n_samples)
    labels[::10] = 1  # 10% attacks
    
    # Split data
    split = int(0.7 * n_samples)
    
    # Test timing-based IDS
    logger.info("\n=== Testing Timing-Based IDS ===")
    timing_ids = TimingBasedIDS(threshold=0.05)
    timing_ids.train(timestamps[:split], can_ids[:split], labels[:split])
    pred, score = timing_ids.predict(timestamps[split:], can_ids[split:])
    logger.info(f"Predictions: {len(pred)}, Anomalies detected: {np.sum(pred)}")
    
    # Test frequency-based IDS
    logger.info("\n=== Testing Frequency-Based IDS ===")
    freq_ids = FrequencyBasedIDS(window_size=50)
    freq_ids.train(timestamps[:split], can_ids[:split], labels[:split])
    pred, score = freq_ids.predict(timestamps[split:], can_ids[split:])
    logger.info(f"Predictions: {len(pred)}, Anomalies detected: {np.sum(pred)}")
    
    # Test rule-based IDS
    logger.info("\n=== Testing Rule-Based IDS ===")
    rule_ids = RuleBasedIDS()
    rule_ids.train(can_ids[:split], dlc[:split], labels[:split])
    pred, score = rule_ids.predict(can_ids[split:], dlc[split:], data[split:])
    logger.info(f"Predictions: {len(pred)}, Anomalies detected: {np.sum(pred)}")
    
    logger.info("\nBaseline models testing complete!")


if __name__ == "__main__":
    main()
