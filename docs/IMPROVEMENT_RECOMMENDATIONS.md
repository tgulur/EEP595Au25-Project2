# Model Performance Improvement Recommendations

## Executive Summary

Based on analysis of the latest experiment results and ablation study, several critical issues have been identified that need to be addressed to improve model performance:

### Critical Issues Found:
1. **Voltage Fingerprinting**: 97.66% False Positive Rate (FPR) - nearly all normal traffic flagged as attacks
2. **LSTM Model**: Predicting all zeros (no attacks detected) - complete failure on attack detection
3. **Data Imbalance**: Severe class imbalance affecting model training
4. **Fusion Layer**: Using simulated results rather than properly trained fusion

---

## 1. Voltage Fingerprinting Improvements

### Current Performance:
- Accuracy: 16.67%
- TPR: 100% (detects all attacks)
- FPR: 97.66% (flags almost all normal traffic as attacks)
- ROC AUC: 0.5337 (barely better than random)

### Root Causes:
1. **Threshold too low** (currently 0.4 in config) - causing excessive false positives
2. **Anomaly detection logic** in `detect_anomaly()` is too aggressive
3. **Feature normalization** may not be optimal
4. **ECU profile matching** needs better statistical methods

### Recommended Improvements:

#### 1.1 Adaptive Threshold Tuning
```python
# In voltage_fingerprinting.py, implement adaptive threshold based on:
# - Historical false positive rate
# - Confidence distribution
# - Operating conditions (temperature, load)

def adaptive_threshold(self, recent_fpr: float, target_fpr: float = 0.05):
    """Dynamically adjust threshold to maintain target FPR"""
    if recent_fpr > target_fpr * 1.5:
        self.threshold *= 1.1  # Increase threshold to reduce FPR
    elif recent_fpr < target_fpr * 0.5:
        self.threshold *= 0.95  # Decrease threshold to catch more attacks
    return self.threshold
```

#### 1.2 Improved Anomaly Detection Logic
**Current Issue**: The `detect_anomaly()` method flags mismatches too aggressively.

**Fix**: Implement confidence-based anomaly scoring with hysteresis:
```python
def detect_anomaly(self, voltage_signal, claimed_ecu_id):
    predicted_ecu, confidence = self.predict(voltage_signal)
    
    # Use confidence margin instead of binary threshold
    confidence_margin = confidence - self.threshold
    
    # Only flag as anomaly if:
    # 1. ECU mismatch AND
    # 2. High confidence in mismatch AND
    # 3. Confidence margin exceeds threshold
    if predicted_ecu != claimed_ecu_id:
        if confidence > self.threshold and confidence_margin > 0.2:
            return True, confidence
        else:
            # Low confidence mismatch - could be noise
            return False, confidence * 0.3
    else:
        # Match - but check if confidence is suspiciously low
        if confidence < 0.5:
            return True, 1.0 - confidence  # Low confidence match is suspicious
        return False, 1.0 - confidence
```

#### 1.3 Enhanced Feature Engineering
Add more discriminative features:
- **Wavelet transform features** for multi-scale analysis
- **Autocorrelation features** for periodic patterns
- **Higher-order statistics** (skewness, kurtosis of derivatives)
- **Temporal features** (velocity, acceleration of voltage changes)

#### 1.4 Better Statistical Matching
Replace simple distance metrics with:
- **Mahalanobis distance** with covariance matrix
- **Kolmogorov-Smirnov test** for distribution matching
- **Dynamic Time Warping (DTW)** for temporal alignment

---

## 2. Deep Learning Model Improvements

### Current Performance:
- **LSTM**: Predicting all zeros (0% attack detection)
- **CNN**: Better but needs verification
- **Issue**: Models likely overfitting or not seeing attack patterns

### Root Causes:
1. **Severe class imbalance** - models learn to always predict majority class
2. **Insufficient attack samples** in training data
3. **No class weighting** in loss function
4. **Missing data augmentation** for attacks

### Recommended Improvements:

#### 2.1 Address Class Imbalance
```python
# In deep_learning_models.py, modify training:

def train(self, X_train, y_train, X_val, y_val, ...):
    # Calculate class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        'balanced', 
        classes=classes, 
        y=y_train
    )
    class_weight_dict = dict(zip(classes, class_weights))
    
    # Use weighted loss
    self.model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', ...],
        # Add sample_weight_mode if needed
    )
    
    # Train with class weights
    self.history = self.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,  # Add this
        ...
    )
```

#### 2.2 Data Augmentation for Attacks
```python
# Add to dataset_loader.py:

def augment_attack_sequences(self, X_attack, y_attack):
    """Augment attack sequences to balance dataset"""
    augmented = []
    
    # Time warping
    for seq in X_attack:
        warped = self.time_warp(seq, sigma=0.2)
        augmented.append(warped)
    
    # Add noise
    for seq in X_attack:
        noisy = seq + np.random.normal(0, 0.01, seq.shape)
        augmented.append(noisy)
    
    # Mixup
    for i in range(len(X_attack) // 2):
        mix = 0.5 * X_attack[i] + 0.5 * X_attack[i+1]
        augmented.append(mix)
    
    return np.array(augmented)
```

#### 2.3 Focal Loss for Hard Examples
```python
# Replace binary_crossentropy with focal loss:

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.ones_like(y_true) * alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_t, 1 - alpha_t)
        loss = -alpha_t * tf.pow(1.0 - pt, gamma) * tf.math.log(pt + 1e-8)
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# Use in model compilation:
self.model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    ...
)
```

#### 2.4 Add Attention Mechanisms
```python
# In LSTMModel, add attention:

def build_model(self, input_shape):
    input_layer = layers.Input(shape=input_shape)
    x = input_layer
    
    # LSTM layers
    for i, units in enumerate(self.hidden_units):
        return_sequences = (i < len(self.hidden_units) - 1)
        lstm_layer = layers.LSTM(
            units=units,
            return_sequences=return_sequences,
            ...
        )
        if self.bidirectional:
            x = layers.Bidirectional(lstm_layer)(x)
        else:
            x = lstm_layer(x)
    
    # Add attention mechanism
    attention = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=64
    )(x, x)
    x = layers.Add()([x, attention])  # Residual connection
    x = layers.LayerNormalization()(x)
    
    # Rest of model...
```

#### 2.5 Ensemble Methods
```python
# Create ensemble of models with different architectures:

class EnsembleModel:
    def __init__(self):
        self.models = [
            LSTMModel(hidden_units=[128, 64], bidirectional=True),
            LSTMModel(hidden_units=[256, 128], bidirectional=False),
            CNNModel(filters=[32, 64, 128]),
            HybridCNNLSTM()
        ]
    
    def predict(self, X):
        predictions = []
        for model in self.models:
            pred, score = model.predict(X)
            predictions.append(score)
        
        # Weighted average based on validation performance
        ensemble_score = np.average(predictions, weights=[0.3, 0.3, 0.2, 0.2])
        return (ensemble_score > 0.5).astype(int), ensemble_score
```

---

## 3. Fusion Layer Improvements

### Current Issues:
- Fusion results are simulated, not actually trained
- No adaptive weighting based on model performance
- Missing confidence calibration

### Recommended Improvements:

#### 3.1 Proper Fusion Training
```python
# In fusion_layer.py, implement proper training for each combination:

def train_fusion_ablation(self, voltage_scores, cnn_scores, lstm_scores, 
                         voltage_confs, cnn_confs, lstm_confs, labels):
    """Train fusion for each component combination"""
    combinations = {
        'V+C': ([voltage_scores, cnn_scores], [voltage_confs, cnn_confs]),
        'V+L': ([voltage_scores, lstm_scores], [voltage_confs, lstm_confs]),
        'C+L': ([cnn_scores, lstm_scores], [cnn_confs, lstm_confs]),
        'V+C+L': ([voltage_scores, cnn_scores, lstm_scores], 
                  [voltage_confs, cnn_confs, lstm_confs])
    }
    
    trained_fusions = {}
    for name, (scores, confs) in combinations.items():
        fusion = FusionLayer(method='stacking')
        fusion.train(scores[0], scores[1], confs[0], confs[1], labels)
        trained_fusions[name] = fusion
    
    return trained_fusions
```

#### 3.2 Confidence Calibration
```python
# Add calibration to improve confidence estimates:

from sklearn.calibration import CalibratedClassifierCV

def calibrate_fusion(self, X_val, y_val):
    """Calibrate fusion layer confidence scores"""
    # Get raw predictions
    raw_scores = self.combiner.predict_proba(X_val)[:, 1]
    
    # Calibrate
    self.calibrated_model = CalibratedClassifierCV(
        self.combiner, 
        method='isotonic',
        cv=3
    )
    self.calibrated_model.fit(X_val, y_val)
```

#### 3.3 Dynamic Weight Adjustment
```python
# Implement adaptive weighting based on recent performance:

class AdaptiveFusion(FusionLayer):
    def update_weights_online(self, voltage_correct, dl_correct, window_size=100):
        """Update fusion weights based on recent performance"""
        # Track recent accuracy
        self.voltage_accuracy_history.append(voltage_correct)
        self.dl_accuracy_history.append(dl_correct)
        
        if len(self.voltage_accuracy_history) > window_size:
            self.voltage_accuracy_history.pop(0)
            self.dl_accuracy_history.pop(0)
        
        # Calculate recent accuracies
        v_acc = np.mean(self.voltage_accuracy_history[-window_size:])
        dl_acc = np.mean(self.dl_accuracy_history[-window_size:])
        
        # Update weights proportionally
        total = v_acc + dl_acc
        if total > 0:
            self.weights['voltage'] = v_acc / total
            self.weights['dl'] = dl_acc / total
```

---

## 4. Data and Preprocessing Improvements

### Current Issues:
- Synthetic data may not reflect real-world variability
- No data validation/quality checks
- Missing feature engineering for temporal patterns

### Recommended Improvements:

#### 4.1 Better Feature Engineering
```python
# In dataset_loader.py, add temporal features:

def extract_temporal_features(self, sequences):
    """Extract temporal patterns from sequences"""
    features = []
    
    for seq in sequences:
        # Message frequency
        freq = self.calculate_message_frequency(seq)
        
        # Inter-arrival time statistics
        iat_mean = np.mean(np.diff(seq[:, 0]))  # Assuming first col is timestamp
        iat_std = np.std(np.diff(seq[:, 0]))
        
        # Sequence entropy
        entropy = self.calculate_entropy(seq)
        
        # Trend features
        trend = np.polyfit(range(len(seq)), seq[:, 1], 1)[0]
        
        features.append([freq, iat_mean, iat_std, entropy, trend])
    
    return np.array(features)
```

#### 4.2 Data Quality Validation
```python
# Add data validation:

def validate_dataset(self, df):
    """Validate dataset quality"""
    issues = []
    
    # Check class balance
    class_dist = df['label'].value_counts()
    if class_dist.min() / class_dist.max() < 0.1:
        issues.append("Severe class imbalance detected")
    
    # Check for missing values
    if df.isnull().any().any():
        issues.append("Missing values found")
    
    # Check for duplicate sequences
    if df.duplicated().any():
        issues.append("Duplicate samples found")
    
    # Check feature distributions
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].std() == 0:
            issues.append(f"Constant feature: {col}")
    
    return issues
```

#### 4.3 Cross-Validation
```python
# Implement proper cross-validation:

from sklearn.model_selection import StratifiedKFold

def cross_validate_models(self, X, y, n_splits=5):
    """Perform stratified k-fold cross-validation"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model = LSTMModel()
        model.build_model(X_train.shape[1:])
        model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate
        pred, score = model.predict(X_val)
        metrics = self.evaluator.calculate_metrics(y_val, pred, score)
        results.append(metrics)
    
    # Average results
    avg_metrics = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}
    return avg_metrics
```

---

## 5. Evaluation and Metrics Improvements

### Current Issues:
- No cross-validation (single train/test split)
- Missing per-attack-type evaluation
- No latency analysis for real-time constraints

### Recommended Improvements:

#### 5.1 Per-Attack-Type Evaluation
```python
# In evaluation_metrics.py:

def evaluate_by_attack_type(self, y_true, y_pred, attack_types):
    """Evaluate performance for each attack type"""
    results = {}
    
    for attack_type in np.unique(attack_types):
        mask = attack_types == attack_type
        y_true_subset = y_true[mask]
        y_pred_subset = y_pred[mask]
        
        metrics = self.calculate_metrics(y_true_subset, y_pred_subset)
        results[attack_type] = metrics
    
    return results
```

#### 5.2 Real-Time Performance Metrics
```python
# Add throughput and latency analysis:

def measure_throughput(self, model, X_test, batch_sizes=[1, 8, 16, 32]):
    """Measure messages processed per second"""
    throughput_results = {}
    
    for batch_size in batch_sizes:
        start = time.time()
        _ = model.predict(X_test, batch_size=batch_size)
        elapsed = time.time() - start
        
        throughput = len(X_test) / elapsed
        throughput_results[batch_size] = throughput
    
    return throughput_results
```

---

## 6. Configuration Improvements

### Recommended Config Changes:

```yaml
# config/config.yaml updates:

voltage:
  anomaly_threshold: 0.6  # Increase from 0.4 to reduce FPR
  adaptive_threshold: true
  target_fpr: 0.05  # Target 5% FPR

deep_learning:
  # Add class weighting
  use_class_weights: true
  focal_loss:
    enabled: true
    gamma: 2.0
    alpha: 0.25
  
  # Data augmentation
  augmentation:
    enabled: true
    methods: ["time_warp", "noise", "mixup"]
    augmentation_factor: 2.0  # 2x attack samples
  
  # Regularization
  dropout: 0.4  # Increase from 0.3
  l2_regularization: 0.001

fusion:
  method: "stacking"
  calibration: true
  adaptive_weights: true
  weight_update_frequency: 100  # Update every 100 samples

evaluation:
  cross_validation:
    enabled: true
    n_splits: 5
  per_attack_evaluation: true
```

---

## 7. Implementation Priority

### High Priority (Immediate):
1. ✅ Fix class imbalance in DL models (add class weights)
2. ✅ Increase voltage threshold to reduce FPR
3. ✅ Implement proper fusion training (not simulated)
4. ✅ Add data augmentation for attacks

### Medium Priority (Next Sprint):
5. ✅ Add focal loss for hard examples
6. ✅ Implement adaptive threshold for voltage
7. ✅ Add confidence calibration for fusion
8. ✅ Implement cross-validation

### Low Priority (Future):
9. ✅ Add attention mechanisms
10. ✅ Implement ensemble methods
11. ✅ Add wavelet features to voltage fingerprinting
12. ✅ Real-time performance optimization

---

## 8. Expected Improvements

After implementing these changes, expected performance:

| Model | Current Accuracy | Expected Accuracy | Current FPR | Expected FPR |
|-------|-----------------|-------------------|-------------|--------------|
| Voltage | 16.67% | 85-90% | 97.66% | 5-10% |
| LSTM | 0% (all zeros) | 95-98% | N/A | 1-3% |
| CNN | ~97% | 96-99% | ~3% | 1-2% |
| Fusion | Simulated | 98-99% | Simulated | 0.5-1% |

---

## 9. Testing Strategy

1. **Unit Tests**: Test each improvement component independently
2. **Integration Tests**: Test full pipeline with improvements
3. **Ablation Studies**: Verify each improvement's contribution
4. **Cross-Validation**: Ensure improvements generalize
5. **Real-World Testing**: Test on actual CAN bus data if available

---

## Conclusion

The main issues are:
1. **Voltage fingerprinting** has excessive false positives (97.66% FPR)
2. **LSTM model** is not detecting any attacks (predicting all zeros)
3. **Class imbalance** is causing models to learn trivial solutions
4. **Fusion layer** needs proper training, not simulation

By addressing these issues systematically, we can expect significant improvements in all metrics, particularly reducing the false positive rate while maintaining high attack detection rates.

