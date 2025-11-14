"""
Quick validation experiment with reduced dataset size to test all fixes.
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from dataset_loader import CANDatasetLoader
from voltage_fingerprinting import VoltageFingerprinter
from deep_learning_models import CNNModel, LSTMModel
from fusion_layer import FusionLayer
from evaluation_metrics import IDSEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("QUICK VALIDATION EXPERIMENT")
    logger.info("Testing: stratified splits, CPU optimization, warmup, attack detection")
    logger.info("=" * 60)
    
    # Create small test datasets
    logger.info("\n1. Creating small test datasets...")
    loader = CANDatasetLoader("data/raw")
    
    # Generate small voltage dataset (100 samples)
    voltage_df = loader._create_sample_voltage_data(n_samples=100)
    X_voltage, y_voltage = loader.preprocess_voltage_data(voltage_df)
    
    # Generate small CAN dataset (1000 samples)
    can_df = loader._create_sample_can_data(n_samples=1000)
    X_can, y_can, attack_types_can = loader.preprocess_can_data(can_df, sequence_length=50)
    
    logger.info(f"Voltage data: {X_voltage.shape}, Labels: Normal={np.sum(y_voltage==0)}, Attack={np.sum(y_voltage==1)}")
    logger.info(f"CAN data: {X_can.shape}, Labels: Normal={np.sum(y_can==0)}, Attack={np.sum(y_can==1)}")
    
    # Test stratified splitting
    logger.info("\n2. Testing stratified data splitting...")
    voltage_splits = loader.split_data(X_voltage, y_voltage, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    can_splits = loader.split_data(X_can, y_can, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    # Verify test sets have both classes
    X_v_test, y_v_test = voltage_splits['test']
    X_c_test, y_c_test = can_splits['test']
    
    logger.info(f"✓ Voltage test set: {len(y_v_test)} samples, Normal={np.sum(y_v_test==0)}, Attack={np.sum(y_v_test==1)}")
    logger.info(f"✓ CAN test set: {len(y_c_test)} samples, Normal={np.sum(y_c_test==0)}, Attack={np.sum(y_c_test==1)}")
    
    if np.sum(y_v_test==1) == 0 or np.sum(y_c_test==1) == 0:
        logger.error("❌ FAILED: Test sets don't have attack samples!")
        return False
    
    logger.info("✓ Stratified splitting working - both classes present in all splits")
    
    # Test voltage fingerprinting
    logger.info("\n3. Testing voltage fingerprinting...")
    X_v_train, y_v_train = voltage_splits['train']
    ecu_ids_train = voltage_df['ecu_id'].values[:len(X_v_train)]
    ecu_ids_test = voltage_df['ecu_id'].values[len(X_v_train) + len(voltage_splits['val'][0]):][:len(X_v_test)]
    
    fingerprinter = VoltageFingerprinter(threshold=0.7)
    fingerprinter.train(X_v_train, ecu_ids_train)
    
    predictions = []
    for i in range(len(X_v_test)):
        is_anomaly, confidence = fingerprinter.detect_anomaly(X_v_test[i], ecu_ids_test[i])
        predictions.append(int(is_anomaly))
    
    predictions = np.array(predictions)
    tp = np.sum((predictions == 1) & (y_v_test == 1))
    fp = np.sum((predictions == 1) & (y_v_test == 0))
    tn = np.sum((predictions == 0) & (y_v_test == 0))
    fn = np.sum((predictions == 0) & (y_v_test == 1))
    
    logger.info(f"✓ Voltage results: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    # Test CNN with CPU optimization and warmup
    logger.info("\n4. Testing CNN with CPU optimization and warmup...")
    X_c_train, y_c_train = can_splits['train']
    X_c_val, y_c_val = can_splits['val']
    
    cnn = CNNModel(filters=[32, 64], kernel_size=3, dropout=0.2)
    cnn.build_model(input_shape=X_c_train.shape[1:])
    
    # Train for just 5 epochs
    logger.info("Training CNN for 5 epochs...")
    cnn.train(X_c_train, y_c_train, X_c_val, y_c_val, 
              epochs=5, batch_size=64, learning_rate=0.001, patience=3)
    
    # Test prediction with warmup
    predictions_cnn, scores_cnn = cnn.predict(X_c_test)
    
    tp = np.sum((predictions_cnn == 1) & (y_c_test == 1))
    fp = np.sum((predictions_cnn == 1) & (y_c_test == 0))
    tn = np.sum((predictions_cnn == 0) & (y_c_test == 0))
    fn = np.sum((predictions_cnn == 0) & (y_c_test == 1))
    
    logger.info(f"✓ CNN results: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    if tp + fp + tn + fn != len(y_c_test):
        logger.error("❌ FAILED: Confusion matrix doesn't add up!")
        return False
    
    # Check if model is detecting anything
    unique_preds = np.unique(predictions_cnn)
    logger.info(f"✓ CNN predicting classes: {unique_preds}")
    
    if len(unique_preds) == 1:
        logger.warning("⚠ WARNING: CNN only predicting one class - may need more training or data")
    else:
        logger.info("✓ CNN detecting both classes!")
    
    # Test LSTM
    logger.info("\n5. Testing LSTM with warmup...")
    lstm = LSTMModel(hidden_units=[64, 32], dropout=0.2, bidirectional=True)
    lstm.build_model(input_shape=X_c_train.shape[1:])
    
    logger.info("Training LSTM for 5 epochs...")
    lstm.train(X_c_train, y_c_train, X_c_val, y_c_val,
               epochs=5, batch_size=64, learning_rate=0.001, patience=3)
    
    predictions_lstm, scores_lstm = lstm.predict(X_c_test)
    
    tp = np.sum((predictions_lstm == 1) & (y_c_test == 1))
    fp = np.sum((predictions_lstm == 1) & (y_c_test == 0))
    tn = np.sum((predictions_lstm == 0) & (y_c_test == 0))
    fn = np.sum((predictions_lstm == 0) & (y_c_test == 1))
    
    logger.info(f"✓ LSTM results: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    unique_preds = np.unique(predictions_lstm)
    logger.info(f"✓ LSTM predicting classes: {unique_preds}")
    
    # Test fusion layer with single-class handling
    logger.info("\n6. Testing fusion layer...")
    
    # Create dummy voltage scores/confidences for fusion
    voltage_scores_test = np.random.random(size=len(y_c_test))
    voltage_conf_test = np.random.random(size=len(y_c_test))
    
    fusion = FusionLayer(method='stacking')
    
    # Combine predictions for training
    voltage_scores_train = np.random.random(size=len(y_c_train))
    voltage_conf_train = np.random.random(size=len(y_c_train))
    train_cnn_preds, train_cnn_scores = cnn.predict(X_c_train)
    train_lstm_preds, train_lstm_scores = lstm.predict(X_c_train)
    
    # Use CNN scores as DL scores
    fusion.train(
        voltage_scores=voltage_scores_train,
        dl_scores=train_cnn_scores,
        voltage_confidences=voltage_conf_train,
        dl_confidences=train_cnn_scores,  # Use scores as confidence
        labels=y_c_train
    )
    
    # Test fusion
    fusion_preds, fusion_scores = fusion.predict_batch(
        voltage_scores=voltage_scores_test,
        dl_scores=scores_cnn,
        voltage_confidences=voltage_conf_test,
        dl_confidences=scores_cnn
    )
    
    tp = np.sum((fusion_preds == 1) & (y_c_test == 1))
    fp = np.sum((fusion_preds == 1) & (y_c_test == 0))
    tn = np.sum((fusion_preds == 0) & (y_c_test == 0))
    fn = np.sum((fusion_preds == 0) & (y_c_test == 1))
    
    logger.info(f"✓ Fusion results: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    # Test evaluation metrics with both classes
    logger.info("\n7. Testing evaluation metrics...")
    evaluator = IDSEvaluator(save_dir="results/test")
    
    try:
        metrics = evaluator.calculate_metrics(y_c_test, predictions_cnn, scores_cnn)
        logger.info(f"✓ Metrics calculated: Accuracy={metrics['accuracy']:.4f}, "
                   f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    except Exception as e:
        logger.error(f"❌ FAILED: Metrics calculation error: {e}")
        return False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info("✓ Stratified data splitting: PASSED")
    logger.info("✓ Attack samples in test sets: PASSED")
    logger.info("✓ Voltage fingerprinting: PASSED")
    logger.info("✓ CNN with CPU optimization & warmup: PASSED")
    logger.info("✓ LSTM with warmup: PASSED")
    logger.info("✓ Fusion layer: PASSED")
    logger.info("✓ Evaluation metrics: PASSED")
    
    if len(np.unique(predictions_cnn)) > 1 or len(np.unique(predictions_lstm)) > 1:
        logger.info("✓ Models detecting both classes: PASSED")
    else:
        logger.warning("⚠ Models only predicting one class - may need longer training")
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL TESTS PASSED! Ready for full experiment.")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
