# Deep Learning and Voltage Fingerprinting for CAN Intrusion Detection

## Project Overview

This project implements an advanced intrusion detection system (IDS) for automotive Controller Area Network (CAN) bus by combining:
- **Voltage Fingerprinting**: Physical-layer ECU identification using voltage signatures
- **Deep Learning**: CNN and LSTM models for attack pattern recognition
- **Decision-Level Fusion**: Adaptive fusion layer combining both approaches

**Team Members**: Tejas Gulur, Keerthi Pobba, Bryan Gonzalez

---

## Features

### Core Components
1. **Voltage Fingerprinting Module** (`src/voltage_fingerprinting.py`)
   - ECU identification from voltage signatures
   - Statistical and frequency domain feature extraction
   - Anomaly detection based on voltage deviations

2. **Deep Learning Models** (`src/deep_learning_models.py`)
   - CNN for spatial pattern recognition
   - LSTM for temporal sequence analysis
   - Hybrid CNN-LSTM architecture

3. **Fusion Layer** (`src/fusion_layer.py`)
   - Weighted average fusion
   - Stacking ensemble with XGBoost/Random Forest
   - Adaptive online weight adjustment

4. **Baseline Models** (`src/baseline_models.py`)
   - Timing-based IDS (clock skew detection)
   - Frequency-based IDS
   - Rule-based IDS

5. **Evaluation Framework** (`src/evaluation_metrics.py`)
   - Comprehensive metrics (TPR, FPR, F1, AUC)
   - Latency measurements
   - Visualization tools

### Attack Scenarios
- **Spoofing**: Impersonating legitimate ECUs
- **DoS**: Denial of Service attacks flooding the bus
- **Fuzzing**: Random invalid messages
- **Replay**: Replaying captured messages
- **Masquerade**: Valid format with wrong ECU

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Linux OS (for SocketCAN support)
- sudo privileges (for virtual CAN setup)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd EEP595Au25-Project2

# Install dependencies
pip install -r requirements.txt

# Setup virtual CAN interface (requires sudo)
sudo bash scripts/setup_vcan.sh
```

---

## Dataset Information

### Supported Datasets

1. **CANMAP Voltage Dataset**
   - Source: [IEEE DataPort](https://ieee-dataport.org/documents/canmap-voltage-dataset-mapping-can-bus)
   - Contains voltage traces from CAN bus for ECU fingerprinting
   - Place in: `data/raw/canmap_voltage/`

2. **ROAD CAN IDS Dataset**
   - Source: [SciDB](https://www.scidb.cn/en/detail?dataSetId=80083f761e5c4008a34a4e29a9f8fe42)
   - Contains labeled CAN messages with various attack types
   - Place in: `data/raw/road_can_ids/`

### Dataset Structure
```
data/
├── raw/
│   ├── canmap_voltage/     # Voltage traces
│   └── road_can_ids/       # CAN message logs
└── processed/              # Preprocessed data
```

---

## Usage

### 1. Virtual CAN Setup and Traffic Generation

```bash
# Setup virtual CAN (run once)
sudo bash scripts/setup_vcan.sh

# Generate normal traffic
python scripts/generate_can_traffic.py --duration 60 --attack normal

# Generate traffic with attacks
python scripts/generate_can_traffic.py --duration 60 --attack spoofing --attack-rate 0.1
python scripts/generate_can_traffic.py --duration 60 --attack dos --attack-rate 0.2
python scripts/generate_can_traffic.py --duration 60 --attack fuzzing --attack-rate 0.15
```

### 2. Data Loading and Preprocessing

```python
from src.dataset_loader import CANDatasetLoader

# Initialize loader
loader = CANDatasetLoader("data/raw")

# Load datasets
voltage_df = loader.load_canmap_voltage_dataset()
can_df = loader.load_road_dataset()

# Preprocess
X_voltage, y_voltage = loader.preprocess_voltage_data(voltage_df)
X_can, y_can = loader.preprocess_can_data(can_df, sequence_length=100)

# Split data
voltage_splits = loader.split_data(X_voltage, y_voltage)
can_splits = loader.split_data(X_can, y_can)
```

### 3. Training Models

#### Voltage Fingerprinting
```python
from src.voltage_fingerprinting import VoltageFingerprinter

fingerprinter = VoltageFingerprinter(threshold=0.7)
fingerprinter.train(voltage_signals, ecu_ids)

# Predict
predicted_ecu, confidence = fingerprinter.predict(test_signal)
is_anomaly, conf = fingerprinter.detect_anomaly(test_signal, claimed_ecu_id)
```

#### Deep Learning Models
```python
from src.deep_learning_models import CNNModel, LSTMModel

# CNN
cnn = CNNModel(filters=[64, 128, 256], dropout=0.3)
cnn.build_model(input_shape=(100, 10))
cnn.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

# LSTM
lstm = LSTMModel(hidden_units=[128, 64], bidirectional=True)
lstm.build_model(input_shape=(100, 10))
lstm.train(X_train, y_train, X_val, y_val, epochs=50)
```

#### Fusion Layer
```python
from src.fusion_layer import FusionLayer

# Train fusion
fusion = FusionLayer(method='stacking', combiner_model='xgboost')
fusion.train(voltage_scores, dl_scores, voltage_confs, dl_confs, labels)

# Predict
prediction, confidence = fusion.predict(v_score, dl_score, v_conf, dl_conf)
```

### 4. Evaluation

```python
from src.evaluation_metrics import IDSEvaluator

evaluator = IDSEvaluator(save_dir="results")

# Calculate metrics
metrics = evaluator.calculate_metrics(y_true, y_pred, y_score)

# Generate visualizations
evaluator.plot_confusion_matrix(y_true, y_pred, save_name="confusion_matrix.png")
evaluator.plot_roc_curve(y_true, y_score, save_name="roc_curve.png")

# Generate report
report = evaluator.generate_report("Model Name", metrics, save_name="report.txt")
print(report)
```

### 5. Baseline Comparison

```python
from src.baseline_models import TimingBasedIDS, FrequencyBasedIDS

# Timing-based
timing_ids = TimingBasedIDS(threshold=0.05)
timing_ids.train(timestamps, can_ids, labels)
predictions, scores = timing_ids.predict(test_timestamps, test_can_ids)

# Frequency-based
freq_ids = FrequencyBasedIDS(window_size=100)
freq_ids.train(timestamps, can_ids, labels)
predictions, scores = freq_ids.predict(test_timestamps, test_can_ids)
```

---

## Project Structure

```
EEP595Au25-Project2/
├── config/
│   └── config.yaml              # Configuration parameters
├── data/
│   ├── raw/                     # Raw datasets
│   └── processed/               # Preprocessed data
├── models/                      # Saved models
├── notebooks/                   # Jupyter notebooks
├── results/                     # Evaluation results
├── scripts/
│   ├── setup_vcan.sh           # Virtual CAN setup
│   └── generate_can_traffic.py # Traffic generation
├── src/
│   ├── dataset_loader.py       # Dataset loading
│   ├── voltage_fingerprinting.py  # Voltage-based IDS
│   ├── deep_learning_models.py # DL models
│   ├── fusion_layer.py         # Decision fusion
│   ├── baseline_models.py      # Baseline methods
│   └── evaluation_metrics.py   # Evaluation tools
├── tests/                       # Unit tests
├── .gitignore
├── requirements.txt
├── README.md
├── proposal.txt                 # Project proposal
└── feedback.txt                 # Feedback notes
```

---

## Evaluation Metrics

The system is evaluated using:

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: Attack prediction accuracy
- **Recall (TPR)**: Attack detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **FPR**: False positive rate
- **AUC-ROC**: Area under ROC curve

### Performance Metrics
- **Latency**: Mean, median, P95, P99
- **Throughput**: Messages processed per second

### Cross-Vehicle Generalization
- Train on one vehicle, test on others
- Measure transfer learning effectiveness

---

## Experimental Results

Results will be saved in `results/` directory including:
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Model comparison charts
- Latency comparisons
- Detailed text reports

---

## Configuration

Edit `config/config.yaml` to adjust:
- Model hyperparameters
- Training parameters
- Dataset paths
- Evaluation settings
- Attack scenarios

---

## References

1. Guiqi Zhang and Yufeng Li. 2024. "Voltage inspector: Sender identification for in-vehicle CAN bus using voltage slice." *Comput. Secur.* 145, C (Oct 2024). https://doi.org/10.1016/j.cose.2024.104017

2. M. Zhang, J. Li, Y. Lai, S. Huan and W. Shang, "A Lightweight Voltage-Based ECU Fingerprint Intrusion Detection System for In-Vehicle CAN Bus," in *IEEE Transactions on Vehicular Technology*, vol. 74, no. 10, pp. 15536-15548, Oct. 2025, doi: 10.1109/TVT.2025.3570961.

3. Z. Xu et al., "Deep learning-based Intrusion Detection Systems: A survey," arXiv.org, https://arxiv.org/abs/2504.07839

4. Saravanan R, et al. "Optimal attention deep learning based in-vehicle intrusion detection and classification model on CAN messages." *Sci Rep.* 2025 Sep 30;15(1):33952. doi: 10.1038/s41598-025-10637-3.

5. Ashton McEntarffer, et al. 2025. "Towards a Comprehensive Evaluation of Voltage-Based Fingerprinting for the CAN Bus." *Proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing*. https://doi.org/10.1145/3672608.3707981

### Datasets
- CANMAP Voltage Dataset: https://ieee-dataport.org/documents/canmap-voltage-dataset-mapping-can-bus
- ROAD CAN IDS Dataset: https://www.scidb.cn/en/detail?dataSetId=80083f761e5c4008a34a4e29a9f8fe42

---

## Contributing

This is an academic project for EEP595 Autumn 2025.

---

## License

Academic use only.

---

## Contact

- Tejas Gulur
- Keerthi Pobba
- Bryan Gonzalez

University of Washington, EEP595 Autumn 2025