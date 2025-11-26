# Quick Start Guide

## CAN Bus Intrusion Detection System

This guide will help you get started quickly with the project.

---

## 1. Initial Setup (5 minutes)

```bash
# Clone or navigate to the repository
cd EEP595Au25-Project2

# Run setup script
bash setup_project.sh

# Activate virtual environment
source venv/bin/activate
```

---

## 2. Setup Virtual CAN (Optional, for live testing)

```bash
# Setup virtual CAN interface (requires sudo)
sudo bash scripts/setup_vcan.sh

# Verify virtual CAN is running
ip link show vcan0
```

---

## 3. Quick Test with Sample Data

```python
# Start Python
python3

# Test data loading
from src.dataset_loader import CANDatasetLoader
loader = CANDatasetLoader("data/raw")
voltage_df = loader.load_canmap_voltage_dataset()
can_df = loader.load_road_dataset()
print("✓ Data loading works!")

# Test voltage fingerprinting
from src.voltage_fingerprinting import VoltageFingerprinter
fingerprinter = VoltageFingerprinter()
X, y = loader.preprocess_voltage_data(voltage_df)
ecu_ids = voltage_df['ecu_id'].values[:len(X)]
fingerprinter.train(X[:100], ecu_ids[:100])
print("✓ Voltage fingerprinting works!")

# Test deep learning models
from src.deep_learning_models import CNNModel
cnn = CNNModel()
X_can, y_can = loader.preprocess_can_data(can_df, sequence_length=100)
cnn.build_model(input_shape=X_can.shape[1:])
print("✓ Deep learning models work!")
print("✓ All components working!")
```

---

## 4. Run Full Experiment

### Option A: Use Default Configuration

```bash
# Run complete experiment with default settings
python main_experiment.py
```

### Option B: Custom Configuration

```bash
# Edit config file
nano config/config.yaml

# Run with custom config
python main_experiment.py --config config/config.yaml
```

Results will be saved in `results/[timestamp]/`

---

## 5. Interactive Exploration

```bash
# Start Jupyter Notebook
jupyter notebook notebooks/demo_notebook.ipynb
```

Then follow the notebook cells to:
- Load and explore datasets
- Train individual models
- Visualize results
- Compare performance

---

## 6. Generate CAN Traffic (Virtual Testing)

```bash
# Terminal 1: Monitor CAN traffic
candump vcan0

# Terminal 2: Generate normal traffic
python scripts/generate_can_traffic.py --duration 60 --attack normal

# Terminal 3: Generate attacks
python scripts/generate_can_traffic.py --duration 30 --attack spoofing --attack-rate 0.1
python scripts/generate_can_traffic.py --duration 30 --attack dos --attack-rate 0.2
```

---

## 7. Common Tasks

### Train Individual Models

```python
from src.dataset_loader import CANDatasetLoader
from src.deep_learning_models import CNNModel, LSTMModel
from src.evaluation_metrics import IDSEvaluator

# Load data
loader = CANDatasetLoader("data/raw")
can_df = loader.load_road_dataset()
X, y = loader.preprocess_can_data(can_df)
splits = loader.split_data(X, y)

# Train CNN
cnn = CNNModel()
cnn.build_model(input_shape=X.shape[1:])
cnn.train(splits['train'][0], splits['train'][1], 
          splits['val'][0], splits['val'][1], epochs=20)

# Evaluate
evaluator = IDSEvaluator()
predictions, scores = cnn.predict(splits['test'][0])
metrics = evaluator.calculate_metrics(splits['test'][1], predictions, scores)
print(metrics)
```

### Test Fusion Layer

```python
from src.fusion_layer import FusionLayer

# Assuming you have voltage and DL scores
fusion = FusionLayer(method='stacking', combiner_model='xgboost')
fusion.train(voltage_scores, dl_scores, voltage_conf, dl_conf, labels)

# Predict
pred, conf = fusion.predict(v_score, dl_score, v_conf, dl_conf)
```

### Compare with Baselines

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

## 8. Project Structure Reference

```
├── config/config.yaml          # Configuration parameters
├── data/                       # Datasets (download required)
├── src/                        # Source code modules
│   ├── dataset_loader.py
│   ├── voltage_fingerprinting.py
│   ├── deep_learning_models.py
│   ├── fusion_layer.py
│   ├── baseline_models.py
│   └── evaluation_metrics.py
├── scripts/                    # Utility scripts
├── notebooks/                  # Jupyter notebooks
├── main_experiment.py          # Main experiment script
└── results/                    # Output results
```

---

## 9. Troubleshooting

### Issue: "Module not found"
```bash
# Make sure you're in the project directory and venv is activated
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "Dataset not found"
```bash
# Check dataset path in config
cat config/config.yaml | grep path

# Or use sample data (automatically generated)
python -c "from src.dataset_loader import CANDatasetLoader; loader = CANDatasetLoader('data/raw'); df = loader.load_canmap_voltage_dataset()"
```

### Issue: Virtual CAN not working
```bash
# Reload kernel modules
sudo modprobe can
sudo modprobe vcan

# Re-run setup
sudo bash scripts/setup_vcan.sh
```

### Issue: CUDA/GPU errors with TensorFlow
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Or install CPU-only TensorFlow
pip install tensorflow-cpu
```

---

## 10. Next Steps

After completing the quick start:

1. **Download Real Datasets**: See `DATASETS.md` for instructions
2. **Tune Hyperparameters**: Edit `config/config.yaml`
3. **Run Full Evaluation**: Use `main_experiment.py`
4. **Cross-Vehicle Testing**: Implement train-on-one-test-on-another scenarios
5. **Extend Models**: Add new architectures or fusion methods
6. **Document Results**: Use the generated reports and visualizations

---

## 11. Getting Help

- Check documentation: `README.md`, `DATASETS.md`
- Review code comments in `src/` modules
- Examine example notebook: `notebooks/demo_notebook.ipynb`
- Review configuration: `config/config.yaml`

---

## 12. Expected Outputs

After running experiments, you should see:

```
results/[timestamp]/
├── voltage_confusion_matrix.png
├── voltage_roc_curve.png
├── voltage_report.txt
├── cnn_confusion_matrix.png
├── cnn_roc_curve.png
├── cnn_report.txt
├── lstm_confusion_matrix.png
├── lstm_roc_curve.png
├── lstm_report.txt
├── fusion_confusion_matrix.png
├── fusion_roc_curve.png
├── fusion_report.txt
└── model_comparison.png
```

---

## Quick Commands Cheat Sheet

```bash
# Setup
bash setup_project.sh
source venv/bin/activate

# Virtual CAN
sudo bash scripts/setup_vcan.sh
candump vcan0

# Generate traffic
python scripts/generate_can_traffic.py --duration 60 --attack spoofing

# Run experiment
python main_experiment.py

# Jupyter notebook
jupyter notebook notebooks/demo_notebook.ipynb

# Test individual module
python src/voltage_fingerprinting.py
python src/deep_learning_models.py
python src/fusion_layer.py
```

---

Happy experimenting! 🚗🔒
