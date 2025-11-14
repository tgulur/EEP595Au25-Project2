# CAN Bus Intrusion Detection

EEP595 Project - Autumn 2025

**Team**: Tejas Gulur, Keerthi Pobba, Bryan Gonzalez

## Overview

Building an IDS for CAN bus that uses voltage fingerprinting + deep learning to detect attacks.

Main components:
- Voltage fingerprinting (ECU identification from voltage signals)
- CNN/LSTM models for attack detection
- Fusion layer to combine both

## Setup

```bash
pip install -r requirements.txt
```

Need Python 3.8+ and Linux for SocketCAN.

## Running

```bash
python main_experiment.py
```

Results go to `results/` folder. Each run gets a timestamped folder.

## Datasets

We're using:
- CANMAP Voltage Dataset (for voltage fingerprinting)
- ROAD CAN IDS Dataset (for CAN messages)

Put them in `data/raw/canmap_voltage/` and `data/raw/road_can_ids/` respectively.

If you don't have them, the code generates synthetic data (not as good but works for testing).

## Code Structure

- `src/` - main code
  - `voltage_fingerprinting.py` - voltage-based detection
  - `deep_learning_models.py` - CNN/LSTM
  - `fusion_layer.py` - combines predictions
  - `dataset_loader.py` - loads data
  - `evaluation_metrics.py` - metrics and plots
- `config/config.yaml` - hyperparameters and settings
- `main_experiment.py` - main script
- `tests/` - unit tests

## How It Works

**Voltage Fingerprinting**: Each ECU has unique hardware that creates different voltage patterns. We build profiles and check if signals match.

**Deep Learning**: CNN for spatial patterns, LSTM for temporal sequences in CAN messages.

**Fusion**: Combines voltage + DL predictions. Tried weighted average and stacking (XGBoost).

## Results

Check `results/` folder. We save:
- Confusion matrices
- ROC curves  
- Metrics reports
- Per-attack-type evaluation

More detailed analysis in `docs/ANALYSIS_SUMMARY.md` (if we've written it yet).

## Configuration

Edit `config/config.yaml` to change hyperparameters, training settings, etc.

## Notes

- Baseline models (timing/frequency based) are disabled - they weren't working well
- Had some issues with class imbalance (LSTM was predicting all zeros at first)
- Voltage fingerprinting had false positives initially, had to tune thresholds
- Data alignment was tricky - make sure attack types match labels throughout pipeline
- Ablation study runs automatically after main experiment

## Testing

```bash
pytest tests/
```

## TODOs / Future Work

- [ ] Better class imbalance handling
- [ ] More fusion methods to try
- [ ] Real-time detection (currently batch)
- [ ] Test on more datasets
- [ ] Adaptive threshold tuning

## References

Main papers we used:
- IdentifierIDS paper (Deng et al., IEEE TIFS 2024) - voltage fingerprinting
- Various deep learning for CAN bus papers (see code comments)

Full citations are in the code where we use the methods.

## Contact

Tejas Gulur, Keerthi Pobba, Bryan Gonzalez
UW EEP595 Autumn 2025
