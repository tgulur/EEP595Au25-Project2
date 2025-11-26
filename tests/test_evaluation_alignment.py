import numpy as np
from pathlib import Path

from src import evaluation


class DummyEvaluatorCapture:
    def __init__(self):
        self.calls = {}

    def calculate_metrics(self, *args, **kwargs):
        return {}

    def plot_detection_heatmap(self, y_test_ref, predictions_dict, save_name=None):
        # record length of y_test_ref and which models provided predictions
        self.calls['heatmap'] = {
            'y_len': len(y_test_ref) if y_test_ref is not None else None,
            'models': list(predictions_dict.keys()),
            'save_name': save_name
        }

    def plot_comprehensive_comparison(self, comprehensive_results, y_test_ref, save_name=None):
        self.calls['comprehensive'] = {
            'y_len': len(y_test_ref) if y_test_ref is not None else None,
            'models': list(comprehensive_results.keys()),
            'save_name': save_name
        }

    def compare_models(self, *args, **kwargs):
        self.calls['compare_models'] = True

    def plot_attack_detection_timeline(self, *args, **kwargs):
        self.calls.setdefault('timelines', []).append(True)


def test_align_predictions_intersection(tmp_path):
    evaluator = DummyEvaluatorCapture()

    # Build results where Voltage, CNN and LSTM provide test_indices with an intersection [1,2]
    results = {
        'voltage': {
            'metrics': {},
            'predictions': np.array([0, 1, 0, 1]),
            'y_test': np.array([0, 1, 0, 1]),
            'test_indices': np.array([0, 1, 2, 3])
        },
        'deep_learning': {
            'CNN': {
                'metrics': {},
                'predictions': np.array([9, 1, 1, 9]),
                'y_test': np.array([9, 1, 1, 9]),
                'test_indices': np.array([1, 2, 3, 4])
            },
            'LSTM': {
                'metrics': {},
                'predictions': np.array([8, 1, 1, 8]),
                'y_test': np.array([8, 1, 1, 8]),
                'test_indices': np.array([1, 2, 5, 6])
            }
        },
        'fusion': {
            'metrics': {},
            # fusion omitted indices to emulate partial availability
            'predictions': np.array([0, 1]),
            'y_test': np.array([0, 1])
        }
    }

    evaluation.compare_and_visualize(evaluator, results, str(tmp_path))

    # Alignment intersection should be [1,2] -> y_len == 2
    assert 'heatmap' in evaluator.calls
    assert evaluator.calls['heatmap']['y_len'] == 2
    assert set(evaluator.calls['heatmap']['models']).issuperset({'Voltage', 'CNN', 'LSTM'})

    assert 'comprehensive' in evaluator.calls
    assert evaluator.calls['comprehensive']['y_len'] == 2
    assert set(evaluator.calls['comprehensive']['models']).issuperset({'Voltage', 'CNN', 'LSTM'})


def test_align_predictions_no_overlap_fallback(tmp_path):
    evaluator = DummyEvaluatorCapture()

    # Build results with no overlapping indices between voltage and CNN
    results = {
        'voltage': {
            'metrics': {},
            'predictions': np.array([0, 1]),
            'y_test': np.array([0, 1]),
            'test_indices': np.array([0, 1])
        },
        'deep_learning': {
            'CNN': {
                'metrics': {},
                'predictions': np.array([0, 1, 1]),
                'y_test': np.array([0, 1, 1]),
                'test_indices': np.array([10, 11, 12])
            },
            'LSTM': {
                'metrics': {},
                'predictions': np.array([0, 0, 1]),
                'y_test': np.array([0, 0, 1]),
                'test_indices': np.array([13, 14, 15])
            }
        },
        'fusion': {'metrics': {}, 'predictions': np.array([0]), 'y_test': np.array([0])}
    }

    evaluation.compare_and_visualize(evaluator, results, str(tmp_path))

    # With no overlap, function should fall back and produce a heatmap using voltage reference
    assert 'heatmap' in evaluator.calls
    assert evaluator.calls['heatmap']['y_len'] == 2
    # Comprehensive comparison should not be called because lengths differ (2 vs 3)
    assert 'comprehensive' not in evaluator.calls
