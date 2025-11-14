# Per-Attack-Type Evaluation Feature

## Overview

The system now tracks and evaluates detection performance for each attack type separately. This allows you to see which attack types are detected well and which ones need improvement.

## Attack Types in Dataset

The dataset includes the following attack types:

1. **normal** - Normal CAN bus traffic (not an attack)
2. **spoofing** - ECU spoofing attacks (voltage fingerprinting detects these)
3. **dos** - Denial of Service attacks (flooding the bus)
4. **fuzzing** - Fuzzing attacks (random/invalid data)
5. **replay** - Replay attacks (replaying old messages)

## How It Works

### 1. Data Preprocessing

The `preprocess_can_data` method now returns three values:
- `X_seq`: Sequence features
- `y_seq`: Sequence labels (0=normal, 1=attack)
- `attack_types_seq`: Attack type for each sequence

Attack types are preserved through the preprocessing pipeline by:
- Tracking the most common attack type in each sequence window
- Maintaining attack type information through train/val/test splits

### 2. Per-Attack-Type Evaluation

After each model evaluation, the system:
1. Groups predictions by attack type
2. Calculates metrics (accuracy, TPR, FPR, precision, recall, F1) for each attack type
3. Generates a detailed report showing performance breakdown

### 3. Generated Reports

Each model (CNN, LSTM, Fusion) now generates:
- **Standard report** (`report.txt`) - Overall performance metrics
- **Attack type report** (`attack_type_report.txt`) - Per-attack-type breakdown

## Report Format

The attack type report includes:

### Summary Table
```
Attack Type    Samples    Attacks    Accuracy    TPR        FPR        Precision  Recall     F1-Score
----------------------------------------------------------------------------------------------------
dos            150        30         0.9800      1.0000     0.0167     0.8571     1.0000     0.9231
fuzzing        150        30         0.9933      0.9667     0.0000     1.0000     0.9667     0.9831
normal         1000       0          0.9950      0.0000     0.0050     0.0000     0.0000     0.0000
replay         150        30         0.9867      1.0000     0.0083     0.9375     1.0000     0.9677
spoofing       150        30         0.9800      1.0000     0.0167     0.8571     1.0000     0.9231
```

### Detailed Metrics
For each attack type, the report shows:
- Sample count (total sequences)
- Attack count (number of actual attacks)
- Normal count (number of normal samples)
- Accuracy, Precision, Recall, F1-Score
- TPR (True Positive Rate) and FPR (False Positive Rate)
- ROC AUC (if available)
- Confusion matrix breakdown (TP, TN, FP, FN)

## Where to Find Reports

After running an experiment, check each model's folder:

```
results/
└── YYYYMMDD_HHMMSS/
    ├── cnn/
    │   ├── report.txt                    # Overall metrics
    │   └── attack_type_report.txt        # Per-attack-type breakdown
    ├── lstm/
    │   ├── report.txt
    │   └── attack_type_report.txt
    └── fusion/
        ├── report.txt
        └── attack_type_report.txt
```

## Example Usage

The per-attack-type evaluation runs automatically during experiment execution. You'll see output like:

```
============================================================
Evaluation Report: CNN
============================================================
...
[Overall metrics]

================================================================================
PER-ATTACK-TYPE EVALUATION REPORT
================================================================================

Attack Type    Samples    Attacks    Accuracy    TPR        FPR        Precision  Recall     F1-Score
----------------------------------------------------------------------------------------------------
dos            150        30         0.9800      1.0000     0.0167     0.8571     1.0000     0.9231
...
```

## Interpreting Results

### Key Metrics to Watch

1. **TPR (True Positive Rate)** - How well the model detects each attack type
   - 1.0000 = Perfect detection
   - < 0.90 = Missing some attacks of this type

2. **FPR (False Positive Rate)** - False alarms for each attack type
   - 0.0000 = No false alarms
   - > 0.10 = Too many false alarms

3. **Accuracy** - Overall correctness for this attack type
   - Consider both TPR and FPR

### Example Analysis

If you see:
```
spoofing       150        30         0.8500      0.8000     0.1200     0.6667     0.8000     0.7273
```

This means:
- **80% TPR**: Detects 24 out of 30 spoofing attacks (missing 6)
- **12% FPR**: 12% of normal traffic flagged as spoofing
- **66.67% Precision**: When it flags spoofing, it's correct 2/3 of the time
- **Overall**: Needs improvement - missing attacks and has false alarms

## Benefits

1. **Identify Weak Points**: See which attack types are hardest to detect
2. **Targeted Improvements**: Focus model improvements on specific attack types
3. **Comprehensive Analysis**: Understand model behavior across different attack scenarios
4. **Research Insights**: Understand which attacks are easier/harder to detect

## Technical Details

### Attack Type Assignment

For sequences (sliding windows):
- If sequence label = 1 (attack): Use most common attack type in the window
- If sequence label = 0 (normal): Assign 'normal'

This ensures each sequence has a clear attack type classification.

### Metrics Calculation

For each attack type:
- Metrics are calculated only on samples of that type
- Normal samples are included in the calculation (for FPR)
- Attack samples are included (for TPR)

This gives you a complete picture of how the model handles each attack type.

---

*Feature added: 2025-11-13*
*Available in: CNN, LSTM, and Fusion models*

