"""
Fusion layer - combines voltage and deep learning signals for better detection
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
import logging

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available, using alternatives")
logger = logging.getLogger(__name__)


class FusionLayer:
    """Combines voltage fingerprinting and deep learning predictions"""
    
    def __init__(self, method: str = 'stacking', combiner_model: str = 'xgboost'):
        self.method = method
        self.combiner_model_type = combiner_model
        self.combiner = None
        self.weights = {'voltage': 0.5, 'dl': 0.5}
        self.trained = False
        
        # Keep track of recent predictions
        self.trend_window = 50
        self.voltage_history = []
        self.dl_history = []
    
    def _create_combiner_model(self):
        """Build the model that combines predictions"""
        if self.combiner_model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.combiner_model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.combiner_model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=42
            )
        else:
            logger.warning(f"Unknown combiner {self.combiner_model_type}, using random_forest")
            return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def extract_fusion_features(self, voltage_score: float, dl_score: float,
                               voltage_confidence: float, dl_confidence: float) -> np.ndarray:
        """Pull together features from both detectors for fusion"""
        features = [
            voltage_score,
            dl_score,
            voltage_confidence,
            dl_confidence,
        ]
        
        # Some interaction features
        features.append(voltage_score * dl_score)  # How much they agree
        features.append(abs(voltage_score - dl_score))  # How much they disagree
        features.append(max(voltage_score, dl_score))
        features.append(min(voltage_score, dl_score))
        
        # Weight scores by confidence
        features.append(voltage_score * voltage_confidence)
        features.append(dl_score * dl_confidence)
        
        # Add trend features if available
        self.voltage_history.append(voltage_score)
        self.dl_history.append(dl_score)
        
        if len(self.voltage_history) > self.trend_window:
            self.voltage_history.pop(0)
            self.dl_history.pop(0)
        
        if len(self.voltage_history) >= 5:
            # Recent trends
            voltage_trend = np.mean(self.voltage_history[-5:])
            dl_trend = np.mean(self.dl_history[-5:])
            voltage_std = np.std(self.voltage_history[-5:])
            dl_std = np.std(self.dl_history[-5:])
        else:
            voltage_trend = voltage_score
            dl_trend = dl_score
            voltage_std = 0.0
            dl_std = 0.0
        
        features.extend([voltage_trend, dl_trend, voltage_std, dl_std])
        
        return np.array(features)
    
    def train(self, voltage_scores: np.ndarray, dl_scores: np.ndarray,
              voltage_confidences: np.ndarray, dl_confidences: np.ndarray,
              labels: np.ndarray):
        """Train the fusion. Finds best way to combine the two detectors"""
        logger.info(f"Training fusion with method: {self.method}")
        
        # Check if we have both classes in the labels
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            logger.warning(f"Only one class found in labels: {unique_labels}")
            logger.warning("Cannot train fusion layer with single-class data")
            logger.warning("Using simple majority voting instead")
            self.method = 'voting'
            self.trained = True
            return
        
        if self.method == 'adaptive':
            # Find best weights by trying different combinations
            best_accuracy = 0.0
            best_weights = {'voltage': 0.5, 'dl': 0.5}
            
            for w_voltage in np.linspace(0, 1, 21):
                w_dl = 1 - w_voltage
                fused_scores = w_voltage * voltage_scores + w_dl * dl_scores
                predictions = (fused_scores > 0.5).astype(int)
                accuracy = np.mean(predictions == labels)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = {'voltage': w_voltage, 'dl': w_dl}
            
            self.weights = best_weights
            logger.info(f"Best weights: Voltage={best_weights['voltage']:.3f}, DL={best_weights['dl']:.3f}")
            logger.info(f"Training accuracy: {best_accuracy:.3f}")
        
        elif self.method == 'stacking':
            # Build features for all samples
            X_fusion = []
            for i in range(len(voltage_scores)):
                features = self.extract_fusion_features(
                    voltage_scores[i], dl_scores[i],
                    voltage_confidences[i], dl_confidences[i]
                )
                X_fusion.append(features)
            
            X_fusion = np.array(X_fusion)
            
            # Train the combiner
            self.combiner = self._create_combiner_model()
            self.combiner.fit(X_fusion, labels)
            
            # Check training accuracy
            train_predictions = self.combiner.predict(X_fusion)
            train_accuracy = np.mean(train_predictions == labels)
            logger.info(f"Training accuracy: {train_accuracy:.3f}")
        
        elif self.method == 'voting':
            # Just use majority voting - nothing to train
            logger.info("Using majority voting - no training needed")
        
        self.trained = True
        logger.info("Fusion layer ready")
    
    def predict(self, voltage_score: float, dl_score: float,
                voltage_confidence: float, dl_confidence: float) -> Tuple[int, float]:
        """Make a combined prediction from both detectors"""
        if self.method == 'weighted_average':
            # Weighted average
            fused_score = (self.weights['voltage'] * voltage_score + 
                          self.weights['dl'] * dl_score)
            prediction = 1 if fused_score > 0.5 else 0
            
            # Average confidence too
            confidence = (self.weights['voltage'] * voltage_confidence + 
                         self.weights['dl'] * dl_confidence)
            
            return prediction, confidence
        
        elif self.method == 'stacking':
            if not self.trained or self.combiner is None:
                raise ValueError("Need to train the fusion layer first")
            
            # Get features
            features = self.extract_fusion_features(
                voltage_score, dl_score,
                voltage_confidence, dl_confidence
            )
            
            # Predict using stacking model
            prediction = self.combiner.predict(features.reshape(1, -1))[0]
            
            # Get probability if available
            if hasattr(self.combiner, 'predict_proba'):
                proba = self.combiner.predict_proba(features.reshape(1, -1))[0]
                confidence = max(proba)
            else:
                confidence = (voltage_confidence + dl_confidence) / 2
            
            return int(prediction), float(confidence)
        
        elif self.method == 'voting':
            # Simple majority vote
            voltage_pred = 1 if voltage_score > 0.5 else 0
            dl_pred = 1 if dl_score > 0.5 else 0
            
            # Take the vote
            if voltage_pred == dl_pred:
                prediction = voltage_pred
                confidence = (voltage_confidence + dl_confidence) / 2
            else:
                # They disagree - use whichever is more confident
                if voltage_confidence > dl_confidence:
                    prediction = voltage_pred
                    confidence = voltage_confidence
                else:
                    prediction = dl_pred
                    confidence = dl_confidence
            
            return prediction, confidence
        
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")
    
    def predict_batch(self, voltage_scores: np.ndarray, dl_scores: np.ndarray,
                     voltage_confidences: np.ndarray, dl_confidences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run predictions on a batch of samples"""
        predictions = []
        confidences = []
        
        for i in range(len(voltage_scores)):
            pred, conf = self.predict(
                voltage_scores[i], dl_scores[i],
                voltage_confidences[i], dl_confidences[i]
            )
            predictions.append(pred)
            confidences.append(conf)
        
        return np.array(predictions), np.array(confidences)
    
    def update_weights_adaptive(self, voltage_accuracy: float, dl_accuracy: float):
        """Adjust weights based on how well each detector is doing"""
        if self.method != 'weighted_average':
            logger.warning("Adaptive weight update only for weighted_average method")
            return
        
        total = voltage_accuracy + dl_accuracy
        if total > 0:
            self.weights['voltage'] = voltage_accuracy / total
            self.weights['dl'] = dl_accuracy / total
            logger.info(f"Updated weights: Voltage={self.weights['voltage']:.3f}, DL={self.weights['dl']:.3f}")


class AdaptiveFusion(FusionLayer):
    """Adaptive fusion that adjusts weights on the fly"""
    
    def __init__(self, window_size: int = 50, update_frequency: int = 10):
        super().__init__(method='weighted_average')
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.prediction_count = 0
        
        # Keep track of performance
        self.voltage_correct = []
        self.dl_correct = []
    
    def predict_adaptive(self, voltage_score: float, dl_score: float,
                        voltage_confidence: float, dl_confidence: float,
                        true_label: Optional[int] = None) -> Tuple[int, float]:
        """Make prediction and optionally update weights if we know the true label"""
        # Get prediction
        prediction, confidence = self.predict(voltage_score, dl_score, 
                                             voltage_confidence, dl_confidence)
        
        # Update tracking if we have the label
        if true_label is not None:
            voltage_pred = 1 if voltage_score > 0.5 else 0
            dl_pred = 1 if dl_score > 0.5 else 0
            
            self.voltage_correct.append(int(voltage_pred == true_label))
            self.dl_correct.append(int(dl_pred == true_label))
            
            # Keep window size fixed
            if len(self.voltage_correct) > self.window_size:
                self.voltage_correct.pop(0)
                self.dl_correct.pop(0)
            
            # Periodically update weights
            self.prediction_count += 1
            if self.prediction_count % self.update_frequency == 0 and len(self.voltage_correct) >= 10:
                voltage_acc = np.mean(self.voltage_correct)
                dl_acc = np.mean(self.dl_correct)
                self.update_weights_adaptive(voltage_acc, dl_acc)
        
        return prediction, confidence


def main():
    """Quick test of the fusion methods"""
    logger.info("Testing fusion...")
    
    # Some random test data
    np.random.seed(42)
    n_samples = 1000
    
    # Fake scores and confidences
    voltage_scores = np.random.rand(n_samples)
    dl_scores = np.random.rand(n_samples)
    voltage_confidences = np.random.uniform(0.6, 1.0, n_samples)
    dl_confidences = np.random.uniform(0.6, 1.0, n_samples)
    
    # Labels somewhat correlated with scores
    labels = ((voltage_scores + dl_scores) / 2 > 0.5).astype(int)
    
    # Try weighted average
    logger.info("\n=== Weighted Average ===")
    fusion_wa = FusionLayer(method='weighted_average')
    fusion_wa.train(voltage_scores, dl_scores, voltage_confidences, dl_confidences, labels)
    pred, conf = fusion_wa.predict(0.7, 0.8, 0.9, 0.85)
    logger.info(f"Test prediction: {pred}, confidence: {conf:.3f}")
    
    # Try stacking
    logger.info("\n=== Stacking ===")
    fusion_stack = FusionLayer(method='stacking', combiner_model='random_forest')
    fusion_stack.train(voltage_scores, dl_scores, voltage_confidences, dl_confidences, labels)
    pred, conf = fusion_stack.predict(0.7, 0.8, 0.9, 0.85)
    logger.info(f"Sample prediction: {pred}, confidence: {conf:.3f}")
    
    # Test voting
    logger.info("\n=== Testing Voting ===")
    fusion_vote = FusionLayer(method='voting')
    fusion_vote.train(voltage_scores, dl_scores, voltage_confidences, dl_confidences, labels)
    pred, conf = fusion_vote.predict(0.7, 0.8, 0.9, 0.85)
    logger.info(f"Sample prediction: {pred}, confidence: {conf:.3f}")
    
    # Test adaptive fusion
    logger.info("\n=== Testing Adaptive Fusion ===")
    adaptive = AdaptiveFusion(window_size=50, update_frequency=10)
    adaptive.train(voltage_scores, dl_scores, voltage_confidences, dl_confidences, labels)
    for i in range(20):
        pred, conf = adaptive.predict_adaptive(
            voltage_scores[i], dl_scores[i],
            voltage_confidences[i], dl_confidences[i],
            labels[i]
        )
    
    logger.info("Fusion layer testing complete!")


if __name__ == "__main__":
    main()
