"""
Test improved voltage fingerprinting with realistic synthetic data
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
from dataset_loader import CANDatasetLoader
from voltage_fingerprinting import VoltageFingerprinter
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def test_voltage_fingerprinting():
    """Test the improved voltage fingerprinting system"""
    
    print("\n" + "="*70)
    print("Testing Improved Voltage Fingerprinting")
    print("="*70)
    
    # Generate realistic synthetic data
    print("\n[1/4] Generating realistic voltage data with hardware-specific characteristics...")
    loader = CANDatasetLoader(data_path='.')
    voltage_df = loader._create_sample_voltage_data(n_samples=500)
    
    print(f"   Generated {len(voltage_df)} voltage samples")
    print(f"   Normal: {np.sum(voltage_df['label']==0)}, Attacks: {np.sum(voltage_df['label']==1)}")
    
    # Preprocess data
    print("\n[2/4] Preprocessing voltage data...")
    X, y = loader.preprocess_voltage_data(voltage_df)
    
    # Split data
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_size = int(0.7 * n_samples)
    test_size = n_samples - train_size
    
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    ecu_ids_train = voltage_df.iloc[train_idx]['ecu_id'].values
    ecu_ids_test = voltage_df.iloc[test_idx]['ecu_id'].values
    
    print(f"   Train set: {len(X_train)} samples (Normal={np.sum(y_train==0)}, Attack={np.sum(y_train==1)})")
    print(f"   Test set: {len(X_test)} samples (Normal={np.sum(y_test==0)}, Attack={np.sum(y_test==1)})")
    
    # Train voltage fingerprinter
    print("\n[3/4] Training voltage fingerprinter...")
    fingerprinter = VoltageFingerprinter(threshold=0.4)  # Adjusted threshold for better balance
    
    # Only train on normal samples
    normal_mask = y_train == 0
    fingerprinter.train(X_train[normal_mask], ecu_ids_train[normal_mask])
    
    # Test detection
    print("\n[4/4] Testing anomaly detection...")
    predictions = []
    
    for signal, ecu_id in zip(X_test, ecu_ids_test):
        is_anomaly, score = fingerprinter.detect_anomaly(signal, ecu_id)
        predictions.append(1 if is_anomaly else 0)
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    print("\n" + "="*70)
    print("Results - Improved Voltage Fingerprinting")
    print("="*70)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {tp} (Attacks correctly detected)")
    print(f"  True Negatives:  {tn} (Normal correctly classified)")
    print(f"  False Positives: {fp} (Normal wrongly flagged as attack)")
    print(f"  False Negatives: {fn} (Attacks missed)")
    
    # Detection rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nDetection Rates:")
    print(f"  True Positive Rate (TPR):  {tpr:.4f} (Sensitivity)")
    print(f"  False Positive Rate (FPR): {fpr:.4f}")
    print(f"  True Negative Rate (TNR):  {tnr:.4f} (Specificity)")
    
    # Analysis
    print("\n" + "="*70)
    print("Analysis")
    print("="*70)
    
    if accuracy > 0.8:
        print("✅ EXCELLENT: Voltage fingerprinting achieving >80% accuracy!")
    elif accuracy > 0.6:
        print("✓ GOOD: Voltage fingerprinting working reasonably well")
    else:
        print("⚠ NEEDS IMPROVEMENT: Consider adjusting threshold or feature extraction")
    
    if tpr > 0.8:
        print(f"✅ HIGH DETECTION RATE: Catching {tpr*100:.1f}% of attacks")
    elif tpr > 0.6:
        print(f"✓ MODERATE DETECTION: Catching {tpr*100:.1f}% of attacks")
    else:
        print(f"⚠ LOW DETECTION: Only catching {tpr*100:.1f}% of attacks")
    
    if fpr < 0.2:
        print(f"✅ LOW FALSE ALARMS: Only {fpr*100:.1f}% false positive rate")
    elif fpr < 0.4:
        print(f"✓ MODERATE FALSE ALARMS: {fpr*100:.1f}% false positive rate")
    else:
        print(f"⚠ HIGH FALSE ALARMS: {fpr*100:.1f}% false positive rate")
    
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70 + "\n")
    
    # Show a few example predictions
    print("Sample Predictions (first 10 test samples):")
    print("-" * 70)
    for i in range(min(10, len(X_test))):
        pred_ecu, conf = fingerprinter.predict(X_test[i])
        is_anom, score = fingerprinter.detect_anomaly(X_test[i], ecu_ids_test[i])
        actual = "ATTACK" if y_test[i] == 1 else "Normal"
        detected = "ATTACK" if is_anom else "Normal"
        status = "✓" if (y_test[i] == predictions[i]) else "✗"
        
        print(f"{status} Sample {i+1}: Actual={actual:6s}, Detected={detected:6s}, "
              f"Claimed ECU=0x{ecu_ids_test[i]:03X}, Predicted=0x{pred_ecu:03X}, Conf={conf:.3f}")
    
    return accuracy, precision, recall, f1


if __name__ == "__main__":
    test_voltage_fingerprinting()
