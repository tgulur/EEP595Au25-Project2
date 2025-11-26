import os
from pathlib import Path

import numpy as np

from src import evaluation


class DummyEvaluator:
    def __init__(self):
        self.calls = {}

    def calculate_metrics(self, y_true, predictions, scores):
        self.calls['calculate_metrics'] = True
        return {
            'accuracy': 0.92,
            'precision': 0.90,
            'recall': 0.88,
            'f1_score': 0.89,
            'mean_latency_ms': 4.2,
            'true_positive_rate': 0.88,
            'false_positive_rate': 0.03,
        }

    def plot_confusion_matrix(self, *args, **kwargs):
        self.calls.setdefault('plots', []).append(('confusion_matrix', kwargs.get('save_name')))

    def plot_roc_curve(self, *args, **kwargs):
        self.calls.setdefault('plots', []).append(('roc_curve', kwargs.get('save_name')))

    def generate_report(self, model_name, metrics, save_name=None):
        self.calls['report'] = (model_name, save_name)
        return f"Report for {model_name}"

    def compare_models(self, comparison_results, metric_names, save_name=None):
        self.calls['compare_models'] = (list(comparison_results.keys()), metric_names, save_name)

    def plot_attack_detection_timeline(self, y_test, preds, model_name=None, save_name=None):
        self.calls.setdefault('timelines', []).append((model_name, save_name))

    def plot_detection_heatmap(self, y_test_ref, predictions_dict, save_name=None):
        self.calls['heatmap'] = save_name

    def plot_comprehensive_comparison(self, comprehensive_results, y_test_ref, save_name=None):
        self.calls['comprehensive'] = save_name


def test_evaluate_model_calls_visualizers(tmp_path):
    evaluator = DummyEvaluator()

    y_true = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 1, 1])
    scores = np.array([0.1, 0.9, 0.6, 0.8])

    metrics = evaluation.evaluate_model(
        evaluator, y_true, preds, scores, model_name='TestModel', save_prefix=str(tmp_path / 'out')
    )

    # metrics should come from our dummy evaluator
    assert isinstance(metrics, dict)
    assert evaluator.calls.get('calculate_metrics') is True
    # report should have been generated and recorded
    assert evaluator.calls.get('report')[0] == 'TestModel'


def test_compare_and_visualize_generates_table_and_calls_plots(tmp_path):
    evaluator = DummyEvaluator()

    # Build minimal results structure expected by compare_and_visualize
    results = {
        'voltage': {
            'metrics': {'accuracy': 0.9, 'true_positive_rate': 0.8, 'false_positive_rate': 0.02},
            'predictions': np.array([0, 1, 0]),
            'y_test': np.array([0, 1, 0]),
        },
        'deep_learning': {
            'CNN': {
                'metrics': {'accuracy': 0.93, 'true_positive_rate': 0.85, 'false_positive_rate': 0.03},
                'predictions': np.array([0, 1, 1]),
                'y_test': np.array([0, 1, 1]),
            },
            'LSTM': {
                'metrics': {'accuracy': 0.91, 'true_positive_rate': 0.82, 'false_positive_rate': 0.04},
                'predictions': np.array([0, 0, 1]),
                'y_test': np.array([0, 0, 1]),
            },
        },
        'fusion': {
            'metrics': {'accuracy': 0.95, 'true_positive_rate': 0.9, 'false_positive_rate': 0.01},
            'predictions': np.array([0, 1, 1]),
            'y_test': np.array([0, 1, 1]),
        },
    }

    # Call function under test
    evaluation.compare_and_visualize(evaluator, results, str(tmp_path))

    # Table should be created under results_dir/comparison
    table_file = Path(tmp_path) / 'comparison' / 'model_comparison_table.txt'
    assert table_file.exists(), f"Expected comparison table at {table_file}"

    # compare_models should have been called and timelines recorded for models
    assert 'compare_models' in evaluator.calls
    assert 'timelines' in evaluator.calls
    # detection heatmap and comprehensive comparison calls should be recorded
    assert 'heatmap' in evaluator.calls
    assert 'comprehensive' in evaluator.calls


def test_run_ablation_study_writes_reports_and_images(tmp_path):
    # This function writes ablation reports and images under results_dir/ablation
    results = {
        'voltage': {'metrics': {'accuracy': 0.85, 'true_positive_rate': 0.8, 'false_positive_rate': 0.05}},
        'deep_learning': {
            'CNN': {'metrics': {'accuracy': 0.9, 'true_positive_rate': 0.85, 'false_positive_rate': 0.03}},
            'LSTM': {'metrics': {'accuracy': 0.88, 'true_positive_rate': 0.83, 'false_positive_rate': 0.04}},
        },
        'fusion': {'metrics': {'accuracy': 0.92, 'true_positive_rate': 0.88, 'false_positive_rate': 0.02}},
    }

    # evaluator not used directly here, but signature requires it
    evaluation.run_ablation_study(None, results, str(tmp_path))

    ablation_dir = Path(tmp_path) / 'ablation'
    report = ablation_dir / 'ablation_results.txt'
    image = ablation_dir / 'ablation_comparison.png'

    assert report.exists(), "Ablation report should be written"
    assert image.exists(), "Ablation visualization image should be written"
