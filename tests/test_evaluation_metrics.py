#!/usr/bin/env python3
"""
Unit tests for evaluation metrics and visualizations with edge case handling
"""

import sys
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from evaluation_metrics import IDSEvaluator


class TestIDSEvaluator:
    """Test suite for IDSEvaluator with edge cases"""
    
    @pytest.fixture
    def evaluator(self, tmp_path):
        """Create evaluator with temporary directory"""
        return IDSEvaluator(save_dir=tmp_path)
    
    @pytest.fixture
    def normal_data(self):
        """Normal test data"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.85, 0.6, 0.75, 0.4, 0.15])
        return y_true, y_pred, y_scores
    
    # ========== Metrics Calculation Tests ==========
    
    def test_calculate_metrics_normal(self, evaluator, normal_data):
        """Test metrics calculation with normal data"""
        y_true, y_pred, y_scores = normal_data
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_scores)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
    
    def test_calculate_metrics_all_correct(self, evaluator):
        """Test with perfect predictions"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        y_scores = np.array([0.1, 0.9, 0.2, 0.85, 0.15, 0.95])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_scores)
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
    
    def test_calculate_metrics_all_wrong(self, evaluator):
        """Test with completely wrong predictions"""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.85, 0.1, 0.2, 0.15])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_scores)
        assert metrics['accuracy'] == 0.0
    
    def test_calculate_metrics_all_normal(self, evaluator):
        """Test with only normal traffic (edge case)"""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 1])  # One false positive
        y_scores = np.array([0.1, 0.2, 0.15, 0.3, 0.6])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_scores)
        assert metrics['accuracy'] == 0.8
        assert metrics['recall'] == 0.0  # No attacks to recall
    
    def test_calculate_metrics_all_attacks(self, evaluator):
        """Test with only attacks (edge case)"""
        y_true = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0])  # One missed
        y_scores = np.array([0.9, 0.8, 0.85, 0.75, 0.4])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_scores)
        assert metrics['accuracy'] == 0.8
        assert metrics['recall'] == 0.8
    
    def test_calculate_metrics_single_sample(self, evaluator):
        """Test with single sample (edge case)"""
        y_true = np.array([1])
        y_pred = np.array([1])
        y_scores = np.array([0.9])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_scores)
        assert metrics['accuracy'] == 1.0
    
    def test_calculate_metrics_empty_arrays(self, evaluator):
        """Test with empty arrays (edge case)"""
        with pytest.raises((ValueError, IndexError)):
            evaluator.calculate_metrics(np.array([]), np.array([]), np.array([]))
    
    def test_calculate_metrics_mismatched_lengths(self, evaluator):
        """Test with mismatched array lengths"""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])  # Wrong length
        y_scores = np.array([0.1, 0.9, 0.2])
        
        with pytest.raises((ValueError, IndexError)):
            evaluator.calculate_metrics(y_true, y_pred, y_scores)
    
    # ========== Confusion Matrix Tests ==========
    
    def test_plot_confusion_matrix_normal(self, evaluator, normal_data, tmp_path):
        """Test confusion matrix plotting with normal data"""
        y_true, y_pred, _ = normal_data
        save_path = tmp_path / "confusion_matrix.png"
        
        evaluator.plot_confusion_matrix(
            y_true, y_pred,
            save_name=save_path.name
        )
        
        assert save_path.exists()
        assert save_path.stat().st_size > 0
    
    def test_plot_confusion_matrix_binary_only(self, evaluator, tmp_path):
        """Test confusion matrix with only two classes"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        evaluator.plot_confusion_matrix(
            y_true, y_pred,
            save_name="test_cm.png"
        )
        
        assert (tmp_path / "test_cm.png").exists()
    
    # ========== ROC Curve Tests ==========
    
    def test_plot_roc_curve_normal(self, evaluator, normal_data, tmp_path):
        """Test ROC curve plotting"""
        y_true, _, y_scores = normal_data
        
        evaluator.plot_roc_curve(
            y_true, y_scores,
            save_name="roc_curve.png"
        )
        
        assert (tmp_path / "roc_curve.png").exists()
    
    def test_plot_roc_curve_perfect_scores(self, evaluator, tmp_path):
        """Test ROC curve with perfect separation"""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.9, 0.95])
        
        evaluator.plot_roc_curve(
            y_true, y_scores,
            save_name="perfect_roc.png"
        )
        
        assert (tmp_path / "perfect_roc.png").exists()
    
    # ========== Attack Timeline Tests ==========
    
    def test_plot_timeline_normal(self, evaluator, tmp_path):
        """Test attack timeline with normal data"""
        y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
        
        evaluator.plot_attack_detection_timeline(
            y_true, y_pred,
            model_name="Test Model",
            save_name="timeline.png"
        )
        
        assert (tmp_path / "timeline.png").exists()
    
    def test_plot_timeline_no_attacks(self, evaluator, tmp_path):
        """Test timeline with no attacks (edge case)"""
        y_true = np.zeros(20)
        y_pred = np.zeros(20)
        
        evaluator.plot_attack_detection_timeline(
            y_true, y_pred,
            model_name="No Attacks",
            save_name="no_attacks.png"
        )
        
        assert (tmp_path / "no_attacks.png").exists()
    
    def test_plot_timeline_all_attacks(self, evaluator, tmp_path):
        """Test timeline with all attacks (edge case)"""
        y_true = np.ones(20)
        y_pred = np.ones(20)
        
        evaluator.plot_attack_detection_timeline(
            y_true, y_pred,
            model_name="All Attacks",
            save_name="all_attacks.png"
        )
        
        assert (tmp_path / "all_attacks.png").exists()
    
    def test_plot_timeline_no_model_name(self, evaluator, tmp_path):
        """Test timeline without model name"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        evaluator.plot_attack_detection_timeline(
            y_true, y_pred,
            save_name="no_name.png"
        )
        
        assert (tmp_path / "no_name.png").exists()
    
    def test_plot_timeline_with_timestamps(self, evaluator, tmp_path):
        """Test timeline with custom timestamps"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        timestamps = np.array([0.0, 0.5, 1.0, 1.5])
        
        evaluator.plot_attack_detection_timeline(
            y_true, y_pred,
            timestamps=timestamps,
            model_name="Custom Time",
            save_name="custom_time.png"
        )
        
        assert (tmp_path / "custom_time.png").exists()
    
    # ========== Detection Heatmap Tests ==========
    
    def test_plot_heatmap_normal(self, evaluator, tmp_path):
        """Test detection heatmap with multiple models"""
        y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1])
        predictions_dict = {
            'Model A': np.array([0, 0, 1, 1, 1, 0, 0, 1]),
            'Model B': np.array([0, 0, 1, 1, 0, 0, 1, 1]),
            'Model C': np.array([0, 0, 1, 0, 0, 0, 1, 0])
        }
        
        evaluator.plot_detection_heatmap(
            y_true, predictions_dict,
            save_name="heatmap.png"
        )
        
        assert (tmp_path / "heatmap.png").exists()
    
    def test_plot_heatmap_no_attacks(self, evaluator, tmp_path):
        """Test heatmap with no attacks (edge case)"""
        y_true = np.zeros(10)
        predictions_dict = {
            'Model A': np.zeros(10),
            'Model B': np.zeros(10)
        }
        
        # Should handle gracefully or skip
        evaluator.plot_detection_heatmap(
            y_true, predictions_dict,
            save_name="no_attacks_heatmap.png"
        )
        
        # May not create file if no attacks, that's OK
    
    def test_plot_heatmap_single_model(self, evaluator, tmp_path):
        """Test heatmap with single model"""
        y_true = np.array([0, 1, 1, 0])
        predictions_dict = {
            'Only Model': np.array([0, 1, 1, 0])
        }
        
        evaluator.plot_detection_heatmap(
            y_true, predictions_dict,
            save_name="single_model_heatmap.png"
        )
        
        assert (tmp_path / "single_model_heatmap.png").exists()
    
    # ========== Comprehensive Comparison Tests ==========
    
    def test_plot_comprehensive_normal(self, evaluator, tmp_path):
        """Test comprehensive comparison with normal data"""
        y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1])
        
        results_dict = {
            'Model A': {
                'metrics': {
                    'accuracy': 0.95,
                    'precision': 0.90,
                    'recall': 0.95,
                    'f1_score': 0.925,
                    'true_positive_rate': 0.95,
                    'false_positive_rate': 0.05,
                    'true_positive': 4,
                    'true_negative': 3,
                    'false_positive': 0,
                    'false_negative': 1
                },
                'predictions': np.array([0, 0, 1, 1, 1, 0, 0, 1])
            }
        }
        
        evaluator.plot_comprehensive_comparison(
            results_dict, y_true,
            save_name="comprehensive.png"
        )
        
        assert (tmp_path / "comprehensive.png").exists()
    
    def test_plot_comprehensive_multiple_models(self, evaluator, tmp_path):
        """Test comprehensive comparison with multiple models"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        
        results_dict = {
            'Model A': {
                'metrics': {
                    'accuracy': 0.83,
                    'precision': 0.75,
                    'recall': 1.0,
                    'f1_score': 0.857,
                    'true_positive_rate': 1.0,
                    'false_positive_rate': 0.33,
                    'true_positive': 3,
                    'true_negative': 2,
                    'false_positive': 1,
                    'false_negative': 0
                },
                'predictions': np.array([0, 1, 1, 1, 0, 1])
            },
            'Model B': {
                'metrics': {
                    'accuracy': 0.67,
                    'precision': 0.67,
                    'recall': 0.67,
                    'f1_score': 0.67,
                    'true_positive_rate': 0.67,
                    'false_positive_rate': 0.33,
                    'true_positive': 2,
                    'true_negative': 2,
                    'false_positive': 1,
                    'false_negative': 1
                },
                'predictions': np.array([0, 1, 1, 1, 0, 0])
            }
        }
        
        evaluator.plot_comprehensive_comparison(
            results_dict, y_true,
            save_name="multi_model_comprehensive.png"
        )
        
        assert (tmp_path / "multi_model_comprehensive.png").exists()
    
    # ========== Report Generation Tests ==========
    
    def test_generate_report_normal(self, evaluator, normal_data, tmp_path):
        """Test report generation"""
        y_true, y_pred, y_scores = normal_data
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_scores)
        
        report = evaluator.generate_report(
            "Test Model",
            metrics,
            save_name="report.txt"
        )
        
        assert isinstance(report, str)
        assert "Test Model" in report
        assert "Accuracy" in report
        assert (tmp_path / "report.txt").exists()
    
    def test_generate_report_perfect_performance(self, evaluator, tmp_path):
        """Test report with perfect metrics"""
        metrics = {
            'accuracy': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0,
            'true_positive': 10,
            'true_negative': 10,
            'false_positive': 0,
            'false_negative': 0,
            'true_positive_rate': 1.0,
            'false_positive_rate': 0.0
        }
        
        report = evaluator.generate_report(
            "Perfect Model",
            metrics,
            save_name="perfect_report.txt"
        )
        
        assert "1.0000" in report or "100%" in report
        assert (tmp_path / "perfect_report.txt").exists()
    
    # ========== Model Comparison Tests ==========
    
    def test_compare_models_normal(self, evaluator, tmp_path):
        """Test model comparison visualization"""
        results_dict = {
            'Model A': {'accuracy': 0.95, 'precision': 0.90, 'recall': 0.95, 'f1_score': 0.925},
            'Model B': {'accuracy': 0.88, 'precision': 0.85, 'recall': 0.90, 'f1_score': 0.875}
        }
        
        evaluator.compare_models(
            results_dict,
            metric_names=['accuracy', 'precision', 'recall', 'f1_score'],
            save_name="comparison.png"
        )
        
        assert (tmp_path / "comparison.png").exists()
    
    def test_compare_models_single(self, evaluator, tmp_path):
        """Test comparison with single model"""
        results_dict = {
            'Only Model': {'accuracy': 0.95, 'precision': 0.90}
        }
        
        evaluator.compare_models(
            results_dict,
            metric_names=['accuracy', 'precision'],
            save_name="single_comparison.png"
        )
        
        assert (tmp_path / "single_comparison.png").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
