# CAN Intrusion Detection System with Voltage Fingerprinting and Deep Learning

## Project Overview

This project implements a comprehensive intrusion detection system (IDS) for automotive Controller Area Network (CAN) bus by combining multiple detection approaches:

- **Voltage Fingerprinting**: Physical-layer ECU identification using voltage signatures
- **Deep Learning Models**: CNN and LSTM architectures for attack pattern recognition
- **Decision-Level Fusion**: Adaptive fusion layer combining multiple detection methods
- **Baseline Models**: Traditional timing and frequency-based detection methods

**Team Members**: Tejas Gulur, Keerthi Pobba, Bryan Gonzalez  
**Course**: EEP595 Autumn 2025, University of Washington

---

## Features

### Core Components
1. **Voltage Fingerprinting Module** - ECU identification from voltage signatures with anomaly detection
2. **Deep Learning Models** - CNN, LSTM, and Hybrid CNN-LSTM architectures optimized for CAN message sequences
3. **Fusion Layer** - Multiple fusion strategies (weighted average, stacking, adaptive weighting)
4. **Baseline Models** - Timing-based IDS, Frequency-based IDS, and Rule-based IDS
5. **Comprehensive Evaluation** - Detailed metrics, visualizations, and performance analysis

### Attack Detection
- **Spoofing**: ECU impersonation attacks
- **DoS**: Denial of Service flooding attacks
- **Fuzzing**: Random invalid message injection
- **Replay**: Captured message replay attacks
- **Masquerade**: Valid format with incorrect ECU source

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Linux OS (recommended for CAN bus simulation)
- GPU with CUDA support (optional, will fallback to CPU)
- sudo privileges (for virtual CAN setup)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd EEP595Au25-Project2

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Setup virtual CAN interface for real CAN simulation
sudo bash scripts/setup_vcan.sh
```

### Running the Complete Experiment

The easiest way to run the full system is using the main experiment script:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run complete experiment (voltage fingerprinting + DL models + fusion + baselines)
python main_experiment.py --config config/config.yaml

# Or run in background with nohup (recommended for long experiments)
nohup python main_experiment.py --config config/config.yaml > experiment_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Note**: The experiment will automatically:
- Generate synthetic datasets if real data is not available
- Train all models (voltage, CNN, LSTM, fusion, baselines)
- Evaluate performance and generate comparisons
- Save results to `results/YYYYMMDD_HHMMSS/` directory
- Create visualizations and detailed reports

### GPU Support

The system automatically detects and uses GPU acceleration when available:
- **GPU Mode**: Uses NVIDIA GPU with CUDA (much faster training)
- **CPU Mode**: Falls back to CPU-only mode (works but slower)

If you encounter GPU compatibility issues (common with TensorFlow 2.20+ and CUDA 13.0), the system will automatically use CPU mode.

---

## Dataset Information

The system supports two types of datasets:

### 1. CANMAP Voltage Dataset
- **Source**: IEEE DataPort
- **Purpose**: Voltage-based ECU fingerprinting
- **Location**: `data/raw/canmap_voltage/`
- **Features**: Physical layer voltage traces for ECU identification

### 2. ROAD CAN IDS Dataset
- **Source**: SciDB
- **Purpose**: CAN message-based intrusion detection
- **Location**: `data/raw/road_can_ids/`
- **Features**: Labeled CAN messages with various attack types

### Data Generation
If datasets are not available, the system automatically generates realistic synthetic data for testing and development.

---

## Manual Usage (Advanced)

For custom usage or development, you can use individual components:

### Data Loading
```python
from src.dataset_loader import CANDatasetLoader

loader = CANDatasetLoader("data/raw")
voltage_data = loader.load_canmap_voltage_dataset()
can_data = loader.load_road_dataset()
```

### Model Training
```python
# Voltage Fingerprinting
from src.voltage_fingerprinting import VoltageFingerprinter
vf = VoltageFingerprinter()
vf.train(voltage_signals, ecu_labels)

# Deep Learning Models
from src.deep_learning_models import CNNModel, LSTMModel
cnn = CNNModel()
cnn.train(X_train, y_train)

# Fusion Layer
from src.fusion_layer import FusionLayer
fusion = FusionLayer(method='stacking')
fusion.train(voltage_scores, dl_scores, labels)
```

### Evaluation
```python
from src.evaluation_metrics import IDSEvaluator
evaluator = IDSEvaluator(save_dir="results")
metrics = evaluator.calculate_metrics(y_true, y_pred, y_scores)
evaluator.generate_report("Model Name", metrics)
```

---

## Project Structure

```
EEP595Au25-Project2/
├── main_experiment.py          # Main experiment runner
├── config/
│   └── config.yaml            # Configuration parameters
├── data/
│   ├── raw/                   # Raw datasets (optional)
│   └── processed/             # Preprocessed/generated data
├── docs/                      # Documentation
├── logs/                      # Experiment logs
├── models/                    # Saved trained models
├── notebooks/                 # Jupyter notebooks for analysis
├── results/                   # Experiment results and reports
│   └── YYYYMMDD_HHMMSS/      # Timestamped result directories
├── scripts/
│   ├── setup_vcan.sh         # Virtual CAN setup
│   ├── generate_can_traffic.py # CAN traffic generation
│   ├── run_experiment_nohup.sh # Background experiment runner
│   └── show_results.sh       # Results visualization
├── src/                       # Source code
│   ├── dataset_loader.py      # Data loading and preprocessing
│   ├── voltage_fingerprinting.py # Voltage-based detection
│   ├── deep_learning_models.py # CNN/LSTM models
│   ├── fusion_layer.py        # Decision fusion
│   ├── baseline_models.py     # Traditional methods
│   └── evaluation_metrics.py  # Evaluation framework
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
├── pytest.ini                # Test configuration
└── README.md                 # This file
```

---

## Results and Output

Experiments save results to timestamped directories in `results/`:

```
results/20251115_143000/
├── voltage_report.txt         # Voltage fingerprinting results
├── cnn_report.txt            # CNN model results
├── lstm_report.txt           # LSTM model results
├── fusion_report.txt         # Fusion model results
├── timing_report.txt         # Baseline timing results
├── frequency_report.txt      # Baseline frequency results
├── comparison_table.txt      # Model comparison
├── voltage_confusion_matrix.png
├── voltage_roc_curve.png
├── cnn_confusion_matrix.png
└── ... (additional plots)
```

### Key Metrics
- **Accuracy**: Overall correctness
- **TPR (Recall)**: Attack detection rate
- **FPR**: False positive rate
- **F1-Score**: Balanced accuracy metric
- **AUC-ROC**: Discrimination ability
- **Latency**: Processing time per message

---

## Configuration

Modify `config/config.yaml` to adjust:
- Model architectures and hyperparameters
- Training parameters (epochs, batch size, learning rate)
- Dataset paths and preprocessing settings
- Fusion methods and weights
- Evaluation metrics and thresholds

---

## Scripts and Tools

### Experiment Management
```bash
# Run experiment in background
./scripts/run_experiment_nohup.sh

# View results
./scripts/show_results.sh

# Run tests
./scripts/run_tests.sh
```

### CAN Traffic Simulation
```bash
# Setup virtual CAN
sudo ./scripts/setup_vcan.sh

# Generate normal traffic
python scripts/generate_can_traffic.py --duration 60 --attack normal

# Generate attack traffic
python scripts/generate_can_traffic.py --duration 60 --attack spoofing --attack-rate 0.1
```

---

## Troubleshooting

### Common Issues

**GPU Not Available**
- System falls back to CPU automatically
- Check GPU compatibility with TensorFlow version
- For GPU issues, contact system administrator

**Missing Datasets**
- System generates synthetic data automatically
- Place real datasets in `data/raw/` for better results

**Memory Issues**
- Reduce batch size in `config/config.yaml`
- Use CPU mode for large models

**Virtual CAN Setup**
- Requires sudo privileges
- Only needed for real CAN bus simulation

---

## References

1. Zhang, G. and Li, Y. (2024). "Voltage inspector: Sender identification for in-vehicle CAN bus using voltage slice." *Computers & Security*, 145, 104017.

2. Zhang, M., et al. (2025). "A Lightweight Voltage-Based ECU Fingerprint Intrusion Detection System for In-Vehicle CAN Bus." *IEEE Transactions on Vehicular Technology*.

3. Xu, Z., et al. (2025). "Deep learning-based Intrusion Detection Systems: A survey." arXiv preprint.

4. Saravanan, R., et al. (2025). "Optimal attention deep learning based in-vehicle intrusion detection and classification model on CAN messages." *Scientific Reports*.

5. McEntarffer, A., et al. (2025). "Towards a Comprehensive Evaluation of Voltage-Based Fingerprinting for the CAN Bus." *Proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing*.

### Datasets
- CANMAP Voltage Dataset: https://ieee-dataport.org/documents/canmap-voltage-dataset-mapping-can-bus
- ROAD CAN IDS Dataset: https://www.scidb.cn/en/detail?dataSetId=80083f761e5c4008a34a4e29a9f8fe42

### Manual Usage (Advanced)

For development or custom analysis, individual components can be used:

```python
# Example: Load and preprocess data
from src.dataset_loader import CANDatasetLoader
loader = CANDatasetLoader("data/raw")
X_can, y_can = loader.preprocess_can_data(loader.load_road_dataset())

# Example: Train individual model
from src.deep_learning_models import CNNModel
cnn = CNNModel()
cnn.train(X_train, y_train)

# Example: Evaluate results
from src.evaluation_metrics import IDSEvaluator
evaluator = IDSEvaluator(save_dir="results")
metrics = evaluator.calculate_metrics(y_true, y_pred, y_scores)
```

See source code documentation for detailed API usage.

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

## Contact

- Tejas Gulur
- Keerthi Pobba
- Bryan Gonzalez

University of Washington, EEP595 Autumn 2025. 

## NOTICE

This README was written by hand and then formatted nicely to README using AI