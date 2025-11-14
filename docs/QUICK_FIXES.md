# Quick Fixes for Immediate Performance Improvements

## Critical Issues to Fix First

### 1. Fix LSTM Model Predicting All Zeros (CRITICAL)

**Problem**: LSTM model is predicting all zeros, detecting no attacks.

**Root Cause**: Severe class imbalance - model learns to always predict majority class.

**Quick Fix** - Add class weights to `src/deep_learning_models.py`:

```python
# In CANIDSModel.train() method, around line 87:

from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

# Add to model.fit() call:
self.history = self.model.fit(
    X_train, y_train,
    validation_data=validation_data,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callback_list,
    class_weight=class_weight_dict,  # ADD THIS LINE
    verbose=1
)
```

### 2. Reduce Voltage Fingerprinting False Positives (CRITICAL)

**Problem**: 97.66% FPR - almost all normal traffic flagged as attacks.

**Quick Fix** - Update threshold in `config/config.yaml`:

```yaml
voltage:
  anomaly_threshold: 0.7  # Increase from 0.4 to 0.7
```

**Better Fix** - Improve anomaly detection logic in `src/voltage_fingerprinting.py`:

```python
# In detect_anomaly() method, replace lines 349-390 with:

def detect_anomaly(self, voltage_signal: np.ndarray, claimed_ecu_id: int) -> Tuple[bool, float]:
    """Detect if a voltage signal is anomalous for the claimed ECU."""
    predicted_ecu, confidence = self.predict(voltage_signal)
    
    # Only flag as anomaly if:
    # 1. ECU mismatch AND
    # 2. High confidence in the mismatch (above threshold)
    if predicted_ecu != claimed_ecu_id:
        # ECU mismatch detected
        if confidence >= self.threshold:
            # High confidence mismatch = strong anomaly signal
            is_anomaly = True
            anomaly_score = confidence
        else:
            # Low confidence mismatch - could be noise/variation
            # Be conservative: don't flag unless very confident
            is_anomaly = False
            anomaly_score = confidence * 0.5  # Partial suspicion
    else:
        # ECU matches - this is normal
        # But check if confidence is suspiciously low
        if confidence < 0.3:
            # Very low confidence match might indicate spoofing
            is_anomaly = True
            anomaly_score = 1.0 - confidence
        else:
            is_anomaly = False
            anomaly_score = 1.0 - confidence  # Lower score = less anomalous
    
    return is_anomaly, anomaly_score
```

### 3. Add Data Augmentation for Attacks

**Problem**: Not enough attack samples for models to learn attack patterns.

**Quick Fix** - Add to `src/dataset_loader.py`:

```python
# Add this method to CANDatasetLoader class:

def augment_attack_data(self, X_attack, y_attack, factor=2):
    """Augment attack sequences to balance dataset"""
    if len(X_attack) == 0:
        return X_attack, y_attack
    
    augmented_X = []
    augmented_y = []
    
    # Original data
    augmented_X.append(X_attack)
    augmented_y.append(y_attack)
    
    # Add noise
    for _ in range(factor):
        noise = np.random.normal(0, 0.01, X_attack.shape)
        augmented_X.append(X_attack + noise)
        augmented_y.append(y_attack)
    
    # Time warping (slight speed variations)
    for _ in range(factor):
        warped = []
        for seq in X_attack:
            # Simple time warping: randomly skip or duplicate timesteps
            indices = np.arange(len(seq))
            warp_indices = np.random.choice(indices, size=len(seq), replace=True)
            warped.append(seq[warp_indices])
        augmented_X.append(np.array(warped))
        augmented_y.append(y_attack)
    
    return np.concatenate(augmented_X), np.concatenate(augmented_y)

# Use in preprocess_can_data():
def preprocess_can_data(self, df: pd.DataFrame, sequence_length: int = 100, 
                       augment_attacks: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    # ... existing code ...
    
    X_seq = np.array(sequences)
    y_seq = np.array(labels)
    
    # Augment attack sequences if requested
    if augment_attacks:
        attack_mask = y_seq == 1
        if np.any(attack_mask):
            X_attack = X_seq[attack_mask]
            y_attack = y_seq[attack_mask]
            X_normal = X_seq[~attack_mask]
            y_normal = y_seq[~attack_mask]
            
            X_attack_aug, y_attack_aug = self.augment_attack_data(X_attack, y_attack)
            
            X_seq = np.concatenate([X_normal, X_attack_aug])
            y_seq = np.concatenate([y_normal, y_attack_aug])
            
            # Shuffle
            indices = np.random.permutation(len(X_seq))
            X_seq = X_seq[indices]
            y_seq = y_seq[indices]
    
    return X_seq, y_seq
```

### 4. Fix Fusion Layer Training

**Problem**: Fusion results are simulated, not actually trained.

**Quick Fix** - Update `main_experiment.py` to properly train fusion:

```python
# In train_fusion_layer() method, around line 395:

def train_fusion_layer(self, voltage_model, dl_models, data: dict):
    """Train fusion layer"""
    logger.info("\n" + "="*60)
    logger.info("Training Fusion Layer")
    logger.info("="*60)
    
    # Get test data
    X_voltage_test, y_voltage_test = data['voltage']['test']
    X_can_test, y_can_test = data['can']['test']
    
    # Ensure same number of samples
    min_samples = min(len(X_voltage_test), len(X_can_test))
    
    # Get voltage scores
    voltage_scores = self.results['voltage']['scores'][:min_samples]
    voltage_confidences = 1.0 - voltage_scores
    
    # Get DL scores from both CNN and LSTM
    cnn_scores = self.results['deep_learning']['CNN']['scores'][:min_samples]
    lstm_scores = self.results['deep_learning']['LSTM']['scores'][:min_samples]
    cnn_confidences = np.abs(cnn_scores - 0.5) * 2
    lstm_confidences = np.abs(lstm_scores - 0.5) * 2
    
    y_test = y_can_test[:min_samples]
    
    # Split for training/testing fusion (70/30)
    split_idx = int(0.7 * len(voltage_scores))
    
    # Train fusion with all three components
    fusion = FusionLayer(
        method=self.config['fusion']['method'],
        combiner_model=self.config['fusion']['combiner']['model']
    )
    
    # Create combined DL scores (average of CNN and LSTM)
    dl_scores_combined = (cnn_scores + lstm_scores) / 2
    dl_confidences_combined = (cnn_confidences + lstm_confidences) / 2
    
    # Train fusion
    fusion.train(
        voltage_scores[:split_idx],
        dl_scores_combined[:split_idx],
        voltage_confidences[:split_idx],
        dl_confidences_combined[:split_idx],
        y_test[:split_idx]
    )
    
    # Test fusion
    predictions, confidences = fusion.predict_batch(
        voltage_scores[split_idx:],
        dl_scores_combined[split_idx:],
        voltage_confidences[split_idx:],
        dl_confidences_combined[split_idx:]
    )
    
    # Calculate metrics
    metrics = self.evaluator.calculate_metrics(
        y_test[split_idx:],
        predictions,
        confidences
    )
    
    # ... rest of method ...
```

## Testing the Fixes

After applying these fixes, run:

```bash
# Run experiment
python main_experiment.py --config config/config.yaml

# Check results
python scripts/view_results.sh

# Run ablation study
python run_ablation_study.py
```

## Expected Improvements

After these quick fixes:
- **LSTM**: Should detect attacks (TPR > 80%)
- **Voltage**: FPR should drop from 97.66% to < 20%
- **Fusion**: Should show actual trained performance, not simulated

## Next Steps

Once these critical fixes are in place, implement the more advanced improvements from `IMPROVEMENT_RECOMMENDATIONS.md`:
1. Focal loss for hard examples
2. Attention mechanisms
3. Adaptive threshold tuning
4. Cross-validation
5. Ensemble methods

