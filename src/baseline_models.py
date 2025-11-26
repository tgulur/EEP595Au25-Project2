"""
Baseline IDS models for comparison
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Any
from collections import defaultdict, deque
import time
import logging
logger = logging.getLogger(__name__)


class BaselineIDS:
    """Base class for our baseline methods"""
    
    def __init__(self, name: str):
        self.name = name
        self.trained = False
    
    def train(self, *args: Any, **kwargs: Any) -> Any:
        """Train the model"""
        raise NotImplementedError
    
    def predict(self, *args: Any, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
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
                mean_interval = float(np.mean(intervals))
                std_interval = float(np.std(intervals))
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
        
        for i in range(1, len(timestamps)):
            can_id = can_ids[i]
            prev_can_id = can_ids[i-1]
            
            if can_id == prev_can_id and can_id in self.expected_intervals:
                interval = timestamps[i] - timestamps[i-1]
                expected_mean, expected_std = self.expected_intervals[can_id]
                
                # Calculate deviation from expected interval
                if expected_std > 0:
                    deviation = abs(interval - expected_mean) / expected_std
                else:
                    deviation = abs(interval - expected_mean)
                
                # Normalize deviation to 0-1 range
                anomaly_score = min(1.0, deviation / 3.0)  # 3-sigma rule
                
                # Check clock skew if available
                if can_id in self.clock_skew:
                    expected_with_skew = expected_mean + self.clock_skew[can_id] * i
                    skew_deviation = abs(interval - expected_with_skew) / (expected_std + 1e-6)
                    anomaly_score = max(anomaly_score, min(1.0, skew_deviation / 3.0))
                
                is_anomaly = anomaly_score > self.threshold
            else:
                # Unknown CAN ID or first message
                anomaly_score = 0.5
                is_anomaly = False
            
            predictions.append(int(is_anomaly))
            scores.append(anomaly_score)
        
        # First message defaults to normal
        predictions.insert(0, 0)
        scores.insert(0, 0.0)
        
        return np.array(predictions), np.array(scores)


class FrequencyBasedIDS(BaselineIDS):
    """IDS that looks for weird message frequencies. DoS attacks spike the rate"""
    
    def __init__(self, window_size: int = 100, threshold: float = 2.0):
        super().__init__(name="Frequency-Based")
        self.window_size = window_size
        self.threshold = threshold
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
                mean_freq = float(np.mean(freq_list))
                std_freq = float(np.std(freq_list))
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
                        
                        # Calculate deviation
                        if expected_std > 0:
                            deviation = abs(current_freq - expected_mean) / expected_std
                        else:
                            deviation = abs(current_freq - expected_mean)
                        
                        anomaly_score = min(1.0, deviation / self.threshold)
                        is_anomaly = deviation > self.threshold
                    else:
                        anomaly_score = 0.5
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
