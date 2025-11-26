"""
CNN and LSTM models for CAN intrusion detection.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from typing import Tuple, Dict, Optional, List, Any
import logging
import time
import os
logger = logging.getLogger(__name__)

# Configure TensorFlow for CPU optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
tf.config.set_soft_device_placement(True)

# Optimize CPU performance
tf.config.threading.set_intra_op_parallelism_threads(0)  # Auto-detect optimal threads
tf.config.threading.set_inter_op_parallelism_threads(0)

# Enable oneDNN optimizations for CPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

logger.info("TensorFlow CPU optimizations enabled")
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"Available devices: {[d.name for d in tf.config.list_physical_devices()]}")


class CANIDSModel:
    """Base class for our IDS models"""
    
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.model: Optional[keras.Model] = None
        self.history: Optional[Any] = None
        self.training_time: float = 0.0
        self.warmed_up: bool = False
        
    def build_model(self, input_shape: Tuple[int, ...]) -> keras.Model:
        """Build the model architecture"""
        raise NotImplementedError("Subclasses must implement build_model")
    
    def warmup(self, input_shape: Tuple[int, ...], n_warmup: int = 10) -> None:
        """
        Warm up the model with dummy predictions to optimize performance.
        This helps initialize internal TensorFlow optimizations.
        """
        if self.model is None:
            logger.warning(f"Cannot warm up {self.model_name} - model not built yet")
            return
        
        if self.warmed_up:
            logger.info(f"{self.model_name} already warmed up")
            return
            
        logger.info(f"Warming up {self.model_name} with {n_warmup} iterations...")
        dummy_input = np.random.randn(1, *input_shape)
        
        # Run several dummy predictions
        start_time = time.time()
        for _ in range(n_warmup):
            _ = self.model.predict(dummy_input, verbose=0)
        
        warmup_time = time.time() - start_time
        logger.info(f"Warmup complete in {warmup_time:.3f}s (avg: {warmup_time/n_warmup*1000:.2f}ms per prediction)")
        self.warmed_up = True
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 50, batch_size: int = 32, 
              learning_rate: float = 0.001,
              early_stopping: bool = True,
              patience: int = 10) -> Dict[str, Any]:
        """Train the model. Pretty standard stuff."""
        if self.model is None:
            raise ValueError("Need to build the model first!")
        
        # Warm up the model before training
        if not self.warmed_up:
            self.warmup(X_train.shape[1:])
        
        # Standard binary classification setup
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        # Callbacks for better training
        callback_list = []
        
        if early_stopping and X_val is not None:
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            callback_list.append(early_stop)
        
        # Reduce LR if we plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # Training
        logger.info(f"Training {self.model_name}...")
        start_time = time.time()
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        self.training_time = time.time() - start_time
        logger.info(f"Done training in {self.training_time:.2f}s")
        
        # Return training history if available
        if self.history is None:
            return {}

        return getattr(self.history, 'history', {})
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions. Returns both binary predictions and probabilities."""
        if self.model is None:
            raise ValueError("Model needs to be built and trained first")
        
        # Ensure model is warmed up
        if not self.warmed_up:
            self.warmup(X.shape[1:])
        
        probabilities = self.model.predict(X, batch_size=batch_size, verbose=0)
        predictions = (probabilities > 0.5).astype(int).flatten()
        
        return predictions, probabilities.flatten()
    
    def predict_with_latency(self, X: np.ndarray, n_iterations: int = 100) -> Dict[str, float]:
        """
        Measure prediction latency
        
        Args:
            X: Input features (single sample or batch)
            n_iterations: Number of iterations for latency measurement
            
        Returns:
            Dictionary with latency statistics
        """
        if len(X.shape) == 2:
            # Add batch dimension
            X = np.expand_dims(X, 0)
        
        # Ensure model is warmed up
        if not self.warmed_up:
            self.warmup(X.shape[1:])
        
        # Measure latency
        latencies = []
        if self.model is None:
            raise ValueError("Model needs to be built before measuring latency")
        for _ in range(n_iterations):
            start = time.time()
            _ = self.model.predict(X[:1], verbose=0)
            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)
        
        # Ensure we return native Python floats for typing consistency
        return {
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'median_latency_ms': float(np.median(latencies))
        }
    
    def save_model(self, path: str) -> None:
        """Save model to file"""
        if self.model is not None:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model from file"""
        self.model = keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")


class CNNModel(CANIDSModel):
    """CNN for pattern recognition in CAN messages"""
    
    def __init__(self, filters: List[int] = [64, 128, 256],
                 kernel_size: int = 3,
                 dropout: float = 0.3):
        super().__init__(model_name="CNN")
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
    
    def build_model(self, input_shape: Tuple[int, ...]) -> keras.Model:
        """Build the CNN. Each conv block has conv -> batch norm -> pooling -> dropout"""
        logger.info(f"Building CNN with input shape {input_shape}")
        
        # Input is already (sequence_length, features) for Conv1D
        input_layer = layers.Input(shape=input_shape)
        x = input_layer
        
        # Stack of conv blocks
        for i, num_filters in enumerate(self.filters):
            x = layers.Conv1D(
                filters=num_filters,
                kernel_size=self.kernel_size,
                padding='same',
                activation='relu',
                name=f'conv1d_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            x = layers.MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}')(x)
            x = layers.Dropout(self.dropout, name=f'dropout_{i+1}')(x)
        
        # Flatten and dense layers
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        x = layers.Dense(128, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout, name='dropout_final')(x)
        x = layers.Dense(64, activation='relu', name='dense_2')(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        self.model = models.Model(inputs=input_layer, outputs=output, name='CNN_IDS')
        
        if self.model is not None:
            logger.info(f"CNN built: {self.model.count_params():,} parameters")
        else:
            logger.warning("CNN built but model is None")
        
        return self.model


class LSTMModel(CANIDSModel):
    """LSTM for catching temporal patterns"""
    
    def __init__(self, hidden_units: List[int] = [128, 64],
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        super().__init__(model_name="LSTM")
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.bidirectional = bidirectional
    
    def build_model(self, input_shape: Tuple[int, ...]) -> keras.Model:
        """Build LSTM. Can go bidirectional for better context"""
        logger.info(f"Building LSTM with input shape {input_shape}")
        
        input_layer = layers.Input(shape=input_shape)
        x = input_layer
        
        # Stack LSTM layers
        for i, units in enumerate(self.hidden_units):
            return_sequences = (i < len(self.hidden_units) - 1)
            
            lstm_layer = layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                recurrent_dropout=self.dropout,
                name=f'lstm_{i+1}'
            )
            
            if self.bidirectional:
                x = layers.Bidirectional(lstm_layer, name=f'bidirectional_{i+1}')(x)
            else:
                x = lstm_layer(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout, name='dropout_final')(x)
        x = layers.Dense(32, activation='relu', name='dense_2')(x)
        
        # Output
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        self.model = models.Model(inputs=input_layer, outputs=output, name='LSTM_IDS')
        
        if self.model is not None:
            logger.info(f"LSTM built: {self.model.count_params():,} parameters")
        else:
            logger.warning("LSTM built but model is None")
        
        return self.model


class HybridCNNLSTM(CANIDSModel):
    """Hybrid model - CNN for features, LSTM for time patterns"""
    
    def __init__(self, cnn_filters: List[int] = [64, 128],
                 lstm_units: List[int] = [64],
                 dropout: float = 0.3):
        super().__init__(model_name="CNN-LSTM")
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout = dropout
    
    def build_model(self, input_shape: Tuple[int, ...]) -> keras.Model:
        """Build hybrid. CNN extracts features, LSTM handles sequences"""
        logger.info(f"Building hybrid CNN-LSTM with input shape {input_shape}")
        
        input_layer = layers.Input(shape=input_shape)
        x = input_layer
        
        # CNN for feature extraction
        for i, num_filters in enumerate(self.cnn_filters):
            x = layers.Conv1D(num_filters, 3, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(2)(x)
        
        # LSTM for temporal stuff
        for units in self.lstm_units:
            x = layers.LSTM(units, return_sequences=False, dropout=self.dropout)(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        
        # Output
        output = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = models.Model(inputs=input_layer, outputs=output, name='Hybrid_CNN_LSTM_IDS')
        
        if self.model is not None:
            logger.info(f"Hybrid built: {self.model.count_params():,} parameters")
        else:
            logger.warning("Hybrid built but model is None")
        
        return self.model


def main():
    """Quick test of the models"""
    logger.info("Testing models...")
    
    # Some random data to test with
    X_train = np.random.randn(1000, 100, 10)
    y_train = np.random.randint(0, 2, 1000)
    X_val = np.random.randn(200, 100, 10)
    y_val = np.random.randint(0, 2, 200)
    
    # Test CNN
    logger.info("\n=== Testing CNN ===")
    cnn = CNNModel()
    cnn.build_model(input_shape=(100, 10))
    if cnn.model is not None:
        cnn.model.summary()
    else:
        logger.warning("CNN model is None; cannot show summary")
    
    # Test LSTM
    logger.info("\n=== Testing LSTM ===")
    lstm = LSTMModel()
    lstm.build_model(input_shape=(100, 10))
    if lstm.model is not None:
        lstm.model.summary()
    else:
        logger.warning("LSTM model is None; cannot show summary")
    
    # Test Hybrid
    logger.info("\n=== Testing Hybrid ===")
    hybrid = HybridCNNLSTM()
    hybrid.build_model(input_shape=(100, 10))
    if hybrid.model is not None:
        hybrid.model.summary()
    else:
        logger.warning("Hybrid model is None; cannot show summary")
    
    logger.info("\nAll tests passed!")


if __name__ == "__main__":
    main()
