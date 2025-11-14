# Dataset Structure and Shapes

This document describes the structure and shapes of the generated datasets at each stage of preprocessing.

## 1. Raw Voltage Data (DataFrame)

**Format**: Pandas DataFrame  
**Shape**: `(n_samples, 6)` where `n_samples` is typically 1000-10000

**Columns**:
- `timestamp`: Float - Time of measurement
- `can_id`: Integer - CAN message identifier (e.g., 0x100, 0x200)
- `ecu_id`: Integer - ECU identifier (same as can_id)
- `voltage_samples`: List[float] - 100 voltage measurements per message
- `label`: Integer - 0 (normal) or 1 (attack)
- `attack_type`: String - 'normal', 'spoofing', etc.

**Example**:
```
timestamp: 0.0
can_id: 256
ecu_id: 256
voltage_samples: [0.0, 0.45, 0.92, 1.24, ..., 2.37]  # 100 values
label: 0
attack_type: 'normal'
```

**Voltage Sample Details**:
- Each voltage sample contains **100 measurements** (time series)
- Represents physical layer voltage signature during CAN message transmission
- Captures hardware characteristics: rise time, ringing, overshoot, noise
- Sampling rate: 1 GHz (1000 MHz)
- Voltage range: -0.5V to 3.5V (typical CAN: 0V recessive, 2.5V dominant)

---

## 2. Preprocessed Voltage Data (NumPy Arrays)

**Format**: NumPy arrays  
**X_voltage shape**: `(n_samples, 100)`
- `n_samples`: Number of voltage traces
- `100`: Voltage measurements per trace (time series)

**y_voltage shape**: `(n_samples,)`
- Binary labels: 0 (normal) or 1 (attack)

**Preprocessing**:
- Extracts voltage samples from DataFrame
- Normalizes each sample: zero mean, unit variance
- Formula: `(X - mean) / std`

**Example**:
```python
X_voltage.shape = (1000, 100)  # 1000 samples, 100 voltage points each
y_voltage.shape = (1000,)      # 1000 labels
```

---

## 3. Raw CAN Data (DataFrame)

**Format**: Pandas DataFrame  
**Shape**: `(n_messages, 6)` where `n_messages` is typically 10000-100000

**Columns**:
- `timestamp`: Float - Time of message
- `can_id`: Integer - CAN message identifier
- `dlc`: Integer - Data Length Code (typically 8)
- `data`: List[int] - 8 bytes of CAN payload (0-255 each)
- `label`: Integer - 0 (normal) or 1 (attack)
- `attack_type`: String - 'normal', 'dos', 'fuzzing', 'spoofing', 'replay'

**Example**:
```
timestamp: 0.0
can_id: 256
dlc: 8
data: [167, 30, 100, 211, 64, 109, 209, 212]  # 8 bytes
label: 1
attack_type: 'dos'
```

**Attack Types**:
- `normal`: Legitimate CAN traffic
- `dos`: Denial of Service (flooding)
- `fuzzing`: Random/invalid data injection
- `spoofing`: Impersonating another ECU
- `replay`: Replaying old messages

---

## 4. Preprocessed CAN Data (Sequences)

**Format**: NumPy arrays  
**X_can shape**: `(n_sequences, sequence_length, 10)`
- `n_sequences`: Number of sequences (sliding windows)
- `sequence_length`: Time steps per sequence (typically 50-100)
- `10`: Features per time step (can_id + dlc + 8 data bytes)

**y_can shape**: `(n_sequences,)`
- Binary labels: 0 (normal) or 1 (attack)

**attack_types shape**: `(n_sequences,)`
- String array: Attack type for each sequence

**Preprocessing**:
1. Extracts features: `[can_id, dlc, byte0, byte1, ..., byte7]` = 10 features
2. Normalizes features: `(X - mean) / std` per feature
3. Creates sliding window sequences:
   - Window size: `sequence_length` (e.g., 50 or 100)
   - Stride: 1 (overlapping windows)
   - Label: Majority vote of labels in window
   - Attack type: Most common attack type in window (if attack)

**Example**:
```python
# With 200 messages and sequence_length=50:
X_can.shape = (150, 50, 10)  # 150 sequences, 50 time steps, 10 features
y_can.shape = (150,)          # 150 labels
attack_types.shape = (150,)   # 150 attack type strings
```

**Feature Breakdown (per time step)**:
- Feature 0: `can_id` (normalized)
- Feature 1: `dlc` (normalized)
- Features 2-9: `data[0]` through `data[7]` (normalized)

---

## 5. Data Splits

After preprocessing, data is split into train/validation/test sets using **stratified sampling** (maintains class balance).

**Default Ratios**:
- Train: 70%
- Validation: 15%
- Test: 15%

**Voltage Splits Example**:
```python
X_train: (700, 100)   # 700 samples, 100 voltage points
X_val:   (150, 100)   # 150 samples
X_test:  (150, 100)   # 150 samples
```

**CAN Splits Example**:
```python
X_train: (7000, 50, 10)   # 7000 sequences, 50 time steps, 10 features
X_val:   (1500, 50, 10)  # 1500 sequences
X_test:  (1500, 50, 10)  # 1500 sequences
```

**Stratified Split**:
- Ensures each split has both normal and attack samples
- Prevents class imbalance issues
- Maintains attack type distribution

---

## 6. Model Input Requirements

### Voltage Fingerprinting Model
- **Input**: `(batch_size, 100)` - Voltage time series
- **Output**: Anomaly score and confidence

### CNN Model
- **Input**: `(batch_size, sequence_length, 10)` - CAN sequences
- **Expected**: `(batch_size, 50, 10)` or `(batch_size, 100, 10)`
- **Output**: Binary classification (0/1) and confidence scores

### LSTM Model
- **Input**: `(batch_size, sequence_length, 10)` - CAN sequences
- **Expected**: `(batch_size, 50, 10)` or `(batch_size, 100, 10)`
- **Output**: Binary classification (0/1) and confidence scores

### Fusion Layer
- **Input**: 
  - Voltage scores: `(n_samples,)`
  - Deep learning scores: `(n_samples,)`
  - Confidences: `(n_samples,)`
- **Output**: Combined predictions and scores

---

## 7. Typical Dataset Sizes

**Small Experiment** (for quick testing):
- Voltage: 100-1000 samples
- CAN: 1000-10000 messages → ~750-7500 sequences

**Full Experiment** (production):
- Voltage: 1000-10000 samples
- CAN: 10000-100000 messages → ~7500-75000 sequences

**Sequence Count Formula**:
```
n_sequences = n_messages - sequence_length + 1
```

For example:
- 10,000 messages with sequence_length=100 → 9,901 sequences
- 10,000 messages with sequence_length=50 → 9,951 sequences

---

## 8. Data Flow Summary

```
Raw Voltage DataFrame (n, 6)
    ↓ preprocess_voltage_data()
X_voltage (n, 100), y_voltage (n,)
    ↓ split_data()
X_train (n*0.7, 100), X_val (n*0.15, 100), X_test (n*0.15, 100)

Raw CAN DataFrame (m, 6)
    ↓ preprocess_can_data(sequence_length=50)
X_can (m-49, 50, 10), y_can (m-49,), attack_types (m-49,)
    ↓ split_data()
X_train (s*0.7, 50, 10), X_val (s*0.15, 50, 10), X_test (s*0.15, 50, 10)
    where s = m - 49
```

---

## 9. Key Characteristics

### Voltage Data
- **Purpose**: Physical layer ECU fingerprinting
- **Uniqueness**: Each ECU has distinct hardware signature
- **Attack Detection**: Spoofing attacks show mismatched voltage signatures
- **Normalization**: Per-sample normalization (preserves relative signal shape)

### CAN Data
- **Purpose**: Message-level pattern detection
- **Temporal**: Sequences capture timing and content patterns
- **Attack Detection**: Detects anomalies in message patterns, rates, and content
- **Normalization**: Per-feature normalization (across all samples)

### Attack Types
- Tracked at sequence level for detailed evaluation
- Enables per-attack-type performance metrics
- Helps identify which attacks are easier/harder to detect

