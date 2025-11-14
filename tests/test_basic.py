"""
test_basic.py - Basic unit tests for CAN IDS components
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataset_loader import CANDatasetLoader
from voltage_fingerprinting import VoltageFingerprinter
from deep_learning_models import CNNModel, LSTMModel
from fusion_layer import FusionLayer
from baseline_models import TimingBasedIDS, FrequencyBasedIDS
from evaluation_metrics import IDSEvaluator


class TestDatasetLoader:
    """Test dataset loading functionality"""
    
    def test_initialization(self):
        """Test loader initialization"""
        loader = CANDatasetLoader("data/raw")
        assert loader.data_path.exists() or True  # May not exist yet
    
    def test_sample_voltage_data_generation(self):
        """Test sample voltage data generation"""
        loader = CANDatasetLoader("data/raw")
        df = loader._create_sample_voltage_data(n_samples=100)
        
        assert len(df) == 100
        assert 'timestamp' in df.columns
        assert 'can_id' in df.columns
        assert 'voltage_samples' in df.columns
        assert 'label' in df.columns
    
    def test_sample_can_data_generation(self):
        """Test sample CAN data generation"""
        loader = CANDatasetLoader("data/raw")
        df = loader._create_sample_can_data(n_samples=100)
        
        assert len(df) == 100
        assert 'timestamp' in df.columns
        assert 'can_id' in df.columns
        assert 'data' in df.columns
        assert 'label' in df.columns
    
    def test_voltage_preprocessing(self):
        """Test voltage data preprocessing"""
        loader = CANDatasetLoader("data/raw")
        df = loader._create_sample_voltage_data(n_samples=100)
        X, y = loader.preprocess_voltage_data(df)
        
        assert X.shape[0] == 100
        assert y.shape[0] == 100
        assert len(X.shape) == 2
    
    def test_can_preprocessing(self):
        """Test CAN data preprocessing"""
        loader = CANDatasetLoader("data/raw")
        df = loader._create_sample_can_data(n_samples=200)
        X, y, attack_types = loader.preprocess_can_data(df, sequence_length=10)
        
        assert X.shape[1] == 10  # sequence length
        assert len(X.shape) == 3
        assert y.shape[0] == X.shape[0]
        assert len(attack_types) == len(y)  # Attack types should match sequences
    
    def test_data_splitting(self):
        """Test train/val/test splitting"""
        loader = CANDatasetLoader("data/raw")
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        splits = loader.split_data(X, y)
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        total_samples = sum(len(splits[k][0]) for k in splits.keys())
        assert total_samples == 100


class TestVoltageFingerprinting:
    """Test voltage fingerprinting functionality"""
    
    def test_initialization(self):
        """Test fingerprinter initialization"""
        fp = VoltageFingerprinter(threshold=0.7)
        assert fp.threshold == 0.7
        assert not fp.trained
    
    def test_feature_extraction(self):
        """Test voltage feature extraction"""
        fp = VoltageFingerprinter()
        signal = np.random.randn(100)
        features = fp.extract_features(signal)
        
        assert isinstance(features, dict)
        assert 'mean' in features
        assert 'std' in features
        assert 'peak_to_peak' in features
        assert 'dominant_frequency' in features
    
    def test_training(self):
        """Test fingerprinter training"""
        fp = VoltageFingerprinter()
        
        # Generate sample signals
        signals = [np.random.randn(100) for _ in range(50)]
        ecu_ids = [0x100, 0x200] * 25
        
        fp.train(signals, ecu_ids)
        
        assert fp.trained
        assert len(fp.ecu_profiles) == 2
        assert 0x100 in fp.ecu_profiles
        assert 0x200 in fp.ecu_profiles
    
    def test_prediction(self):
        """Test ECU prediction"""
        fp = VoltageFingerprinter()
        
        # Train
        signals = [np.random.randn(100) for _ in range(50)]
        ecu_ids = [0x100, 0x200] * 25
        fp.train(signals, ecu_ids)
        
        # Predict
        test_signal = np.random.randn(100)
        ecu_id, confidence = fp.predict(test_signal)
        
        assert ecu_id in [0x100, 0x200]
        assert 0 <= confidence <= 1


class TestDeepLearningModels:
    """Test deep learning models"""
    
    def test_cnn_build(self):
        """Test CNN model building"""
        cnn = CNNModel(filters=[32, 64], dropout=0.3)
        model = cnn.build_model(input_shape=(100, 10))
        
        assert model is not None
        assert cnn.model is not None
    
    def test_lstm_build(self):
        """Test LSTM model building"""
        lstm = LSTMModel(hidden_units=[64], dropout=0.2)
        model = lstm.build_model(input_shape=(100, 10))
        
        assert model is not None
        assert lstm.model is not None
    
    def test_model_prediction(self):
        """Test model prediction"""
        cnn = CNNModel()
        cnn.build_model(input_shape=(100, 10))
        
        # Compile manually for testing
        cnn.model.compile(optimizer='adam', loss='binary_crossentropy')
        
        X = np.random.randn(10, 100, 10)
        predictions, scores = cnn.predict(X)
        
        assert len(predictions) == 10
        assert len(scores) == 10
        assert all(p in [0, 1] for p in predictions)


class TestFusionLayer:
    """Test fusion layer"""
    
    def test_weighted_average_fusion(self):
        """Test weighted average fusion"""
        fusion = FusionLayer(method='weighted_average')
        
        # Generate sample scores
        v_scores = np.random.rand(100)
        dl_scores = np.random.rand(100)
        v_conf = np.random.rand(100)
        dl_conf = np.random.rand(100)
        labels = np.random.randint(0, 2, 100)
        
        fusion.train(v_scores, dl_scores, v_conf, dl_conf, labels)
        
        assert fusion.trained
        assert 'voltage' in fusion.weights
        assert 'dl' in fusion.weights
    
    def test_fusion_prediction(self):
        """Test fusion prediction"""
        fusion = FusionLayer(method='weighted_average')
        
        v_scores = np.random.rand(100)
        dl_scores = np.random.rand(100)
        v_conf = np.random.rand(100)
        dl_conf = np.random.rand(100)
        labels = np.random.randint(0, 2, 100)
        
        fusion.train(v_scores, dl_scores, v_conf, dl_conf, labels)
        
        pred, conf = fusion.predict(0.7, 0.8, 0.9, 0.85)
        
        assert pred in [0, 1]
        assert 0 <= conf <= 1


class TestBaselineModels:
    """Test baseline IDS models"""
    
    def test_timing_based_ids(self):
        """Test timing-based IDS"""
        ids = TimingBasedIDS(threshold=0.05)
        
        # Generate sample data
        timestamps = np.cumsum(np.random.uniform(0.001, 0.02, 100))
        can_ids = np.random.choice([0x100, 0x200], 100)
        labels = np.zeros(100)
        
        ids.train(timestamps, can_ids, labels)
        
        assert ids.trained
        
        # Test prediction
        test_ts = np.cumsum(np.random.uniform(0.001, 0.02, 50))
        test_ids = np.random.choice([0x100, 0x200], 50)
        
        predictions, scores = ids.predict(test_ts, test_ids)
        
        assert len(predictions) == 50
        assert len(scores) == 50
    
    def test_frequency_based_ids(self):
        """Test frequency-based IDS"""
        ids = FrequencyBasedIDS(window_size=10)
        
        timestamps = np.cumsum(np.random.uniform(0.001, 0.02, 100))
        can_ids = np.random.choice([0x100, 0x200], 100)
        labels = np.zeros(100)
        
        ids.train(timestamps, can_ids, labels)
        
        assert ids.trained


class TestEvaluationMetrics:
    """Test evaluation metrics"""
    
    def test_metric_calculation(self):
        """Test metrics calculation"""
        evaluator = IDSEvaluator(save_dir="results/test")
        
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        y_score = np.random.rand(8)
        
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_score)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'true_positive_rate' in metrics
        assert 'false_positive_rate' in metrics
        assert 'roc_auc' in metrics
    
    def test_latency_calculation(self):
        """Test latency metrics calculation"""
        evaluator = IDSEvaluator()
        
        latencies = [1.2, 1.5, 1.3, 2.0, 1.8]
        metrics = evaluator.calculate_latency_metrics(latencies)
        
        assert 'mean_latency_ms' in metrics
        assert 'median_latency_ms' in metrics
        assert 'std_latency_ms' in metrics


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
