"""`src` package init for canids project.

This file keeps the code importable as `src.<module>`. Do not add heavy logic here.
"""

__all__ = []
"""
CAN Bus Intrusion Detection System

A comprehensive system combining voltage fingerprinting and deep learning
for automotive CAN bus intrusion detection.
"""

__version__ = "1.0.0"
__author__ = "Tejas Gulur, Keerthi Pobba, Bryan Gonzalez"

from .dataset_loader import CANDatasetLoader
from .voltage_fingerprinting import VoltageFingerprinter
from .deep_learning_models import CNNModel, LSTMModel, HybridCNNLSTM
from .fusion_layer import FusionLayer, AdaptiveFusion
from .baseline_models import TimingBasedIDS, FrequencyBasedIDS, RuleBasedIDS
from .evaluation_metrics import IDSEvaluator

__all__ = [
    'CANDatasetLoader',
    'VoltageFingerprinter',
    'CNNModel',
    'LSTMModel',
    'HybridCNNLSTM',
    'FusionLayer',
    'AdaptiveFusion',
    'TimingBasedIDS',
    'FrequencyBasedIDS',
    'RuleBasedIDS',
    'IDSEvaluator'
]
