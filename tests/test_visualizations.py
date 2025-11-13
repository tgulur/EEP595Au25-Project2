#!/usr/bin/env python3
"""
Unit tests for visualization functions with pytest integration
Tests the comprehensive attack detection visualizations
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation_metrics import IDSEvaluator


@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory for test visualizations"""
    return tmp_path


@pytest.fixture
def evaluator(output_dir):
    """Create evaluator instance with test output directory"""
    return IDSEvaluator(save_dir=output_dir)


@pytest.fixture
def sample_timeline_data():
    """Generate sample data for timeline tests"""
    n_samples = 200
    y_true = np.zeros(n_samples)
    
    # Add attack periods
    y_true[50:70] = 1  # Attack period 1
    y_true[120:140] = 1  # Attack period 2
    y_true[170:185] = 1  # Attack period 3
    
    # Create predictions with some errors
    y_pred = y_true.copy()
    y_pred[55:58] = 0  # Some missed detections
    y_pred[90:95] = 1  # Some false alarms
    y_pred[175:180] = 0  # More missed detections
    
    return y_true, y_pred


@pytest.fixture
def sample_heatmap_data():
    """Generate sample data for heatmap tests"""
    n_samples = 150
    y_true = np.zeros(n_samples)
    y_true[30:50] = 1  # 20 attacks
    y_true[80:95] = 1  # 15 attacks
    y_true[120:135] = 1  # 15 attacks
    
    # Create predictions for different models
    predictions_dict = {
        'Voltage': np.zeros(n_samples),
        'CNN': y_true.copy(),
        'LSTM': y_true.copy(),
        'Fusion': y_true.copy()
    }
    
    # Voltage model - moderate performance
    predictions_dict['Voltage'][30:40] = 1  # Detects only 10/20
    predictions_dict['Voltage'][80:88] = 1  # Detects only 8/15
    predictions_dict['Voltage'][120:129] = 1  # Detects only 9/15
    predictions_dict['Voltage'][55:65] = 1  # 10 false positives
    
    # CNN model - very good performance
    predictions_dict['CNN'][32:35] = 0  # Missed 3
    predictions_dict['CNN'][60:62] = 1  # 2 false positives
    
    # LSTM model - good performance
    predictions_dict['LSTM'][35:40] = 0  # Missed 5
    predictions_dict['LSTM'][85:88] = 0  # Missed 3
    predictions_dict['LSTM'][60:65] = 1  # 5 false positives
    
    return y_true, predictions_dict


@pytest.fixture
def sample_comprehensive_data():
    """Generate sample data for comprehensive comparison"""
    n_samples = 100
    y_true = np.zeros(n_samples)
    y_true[20:40] = 1  # 20 attacks
    y_true[60:75] = 1  # 15 attacks
    
    results_dict = {
        'Voltage': {
            'metrics': {
                'accuracy': 0.77,
                'precision': 0.70,
                'recall': 0.70,
                'f1_score': 0.700,
                'true_positive_rate': 0.70,
                'false_positive_rate': 0.25,
                'true_positive': 24,
                'true_negative': 50,
                'false_positive': 15,
                'false_negative': 11
            },
            'predictions': np.concatenate([
                np.zeros(20),
                np.ones(14), np.zeros(6),  # Miss 6 attacks
                np.ones(8), np.zeros(12),  # 8 false positives
                np.ones(10), np.zeros(5),  # Miss 5 attacks
                np.ones(7), np.zeros(18)   # 7 false positives
            ])
        },
        'CNN': {
            'metrics': {
                'accuracy': 0.95,
                'precision': 0.90,
                'recall': 0.95,
                'f1_score': 0.925,
                'true_positive_rate': 0.95,
                'false_positive_rate': 0.08,
                'true_positive': 33,
                'true_negative': 61,
                'false_positive': 4,
                'false_negative': 2
            },
            'predictions': np.concatenate([
                np.zeros(20),
                np.ones(18), np.zeros(2),  # Miss 2 attacks
                np.zeros(20),
                np.ones(13), np.zeros(2),  # Miss 2 attacks
                np.ones(4), np.zeros(21)   # 4 false positives
            ])
        },
        'LSTM': {
            'metrics': {
                'accuracy': 0.88,
                'precision': 0.82,
                'recall': 0.85,
                'f1_score': 0.835,
                'true_positive_rate': 0.85,
                'false_positive_rate': 0.15,
                'true_positive': 30,
                'true_negative': 55,
                'false_positive': 10,
                'false_negative': 5
            },
            'predictions': np.concatenate([
                np.zeros(20),
                np.ones(17), np.zeros(3),  # Miss 3 attacks
                np.ones(5), np.zeros(15),  # 5 false positives
                np.ones(13), np.zeros(2),  # Miss 2 attacks
                np.ones(5), np.zeros(20)   # 5 false positives
            ])
        }
    }
    
    return y_true, results_dict


class TestAttackDetectionTimeline:
    """Test suite for attack detection timeline visualization"""
    
    @pytest.mark.visualization
    def test_timeline_basic(self, evaluator, sample_timeline_data, output_dir):
        """Test basic timeline generation"""
        y_true, y_pred = sample_timeline_data
        
        evaluator.plot_attack_detection_timeline(
            y_true,
            y_pred,
            model_name="Test Sample",
            save_name="test_attack_timeline.png"
        )
        
        output_file = output_dir / "test_attack_timeline.png"
        assert output_file.exists(), "Timeline plot should be created"
        assert output_file.stat().st_size > 0, "Timeline plot should not be empty"
    
    @pytest.mark.visualization
    def test_timeline_no_model_name(self, evaluator, sample_timeline_data, output_dir):
        """Test timeline without model name"""
        y_true, y_pred = sample_timeline_data
        
        evaluator.plot_attack_detection_timeline(
            y_true,
            y_pred,
            save_name="test_no_name.png"
        )
        
        assert (output_dir / "test_no_name.png").exists()
    
    @pytest.mark.edge_case
    def test_timeline_perfect_detection(self, evaluator, output_dir):
        """Test timeline with perfect detection"""
        y_true = np.array([0, 0, 1, 1, 1, 0, 0])
        y_pred = y_true.copy()
        
        evaluator.plot_attack_detection_timeline(
            y_true, y_pred,
            model_name="Perfect",
            save_name="perfect_timeline.png"
        )
        
        assert (output_dir / "perfect_timeline.png").exists()
    
    @pytest.mark.edge_case
    def test_timeline_all_wrong(self, evaluator, output_dir):
        """Test timeline with all wrong predictions"""
        y_true = np.array([0, 0, 1, 1, 1, 0, 0])
        y_pred = 1 - y_true  # Invert
        
        evaluator.plot_attack_detection_timeline(
            y_true, y_pred,
            model_name="Wrong",
            save_name="wrong_timeline.png"
        )
        
        assert (output_dir / "wrong_timeline.png").exists()


class TestDetectionHeatmap:
    """Test suite for detection heatmap visualization"""
    
    @pytest.mark.visualization
    def test_heatmap_basic(self, evaluator, sample_heatmap_data, output_dir):
        """Test basic heatmap generation"""
        y_true, predictions_dict = sample_heatmap_data
        
        evaluator.plot_detection_heatmap(
            y_true,
            predictions_dict,
            save_name="test_detection_heatmap.png"
        )
        
        output_file = output_dir / "test_detection_heatmap.png"
        assert output_file.exists(), "Heatmap should be created"
        assert output_file.stat().st_size > 0, "Heatmap should not be empty"
    
    @pytest.mark.edge_case
    def test_heatmap_single_model(self, evaluator, output_dir):
        """Test heatmap with single model"""
        y_true = np.array([0, 1, 1, 0, 1])
        predictions_dict = {
            'Only Model': np.array([0, 1, 1, 0, 1])
        }
        
        evaluator.plot_detection_heatmap(
            y_true, predictions_dict,
            save_name="single_model_heatmap.png"
        )
        
        assert (output_dir / "single_model_heatmap.png").exists()
    
    @pytest.mark.edge_case
    def test_heatmap_no_attacks(self, evaluator, output_dir):
        """Test heatmap with no attacks - should handle gracefully"""
        y_true = np.zeros(10)
        predictions_dict = {
            'Model A': np.zeros(10),
            'Model B': np.zeros(10)
        }
        
        # Should not raise an error
        evaluator.plot_detection_heatmap(
            y_true, predictions_dict,
            save_name="no_attacks_heatmap.png"
        )


class TestComprehensiveComparison:
    """Test suite for comprehensive comparison visualization"""
    
    @pytest.mark.visualization
    def test_comprehensive_basic(self, evaluator, sample_comprehensive_data, output_dir):
        """Test basic comprehensive comparison"""
        y_true, results_dict = sample_comprehensive_data
        
        evaluator.plot_comprehensive_comparison(
            results_dict,
            y_true,
            save_name="test_comprehensive_comparison.png"
        )
        
        output_file = output_dir / "test_comprehensive_comparison.png"
        assert output_file.exists(), "Comprehensive comparison should be created"
        assert output_file.stat().st_size > 0, "Comprehensive comparison should not be empty"
    
    @pytest.mark.edge_case
    def test_comprehensive_single_model(self, evaluator, output_dir):
        """Test comprehensive comparison with single model"""
        y_true = np.array([0, 1, 0, 1])
        results_dict = {
            'Only Model': {
                'metrics': {
                    'accuracy': 1.0,
                    'precision': 1.0,
                    'recall': 1.0,
                    'f1_score': 1.0,
                    'true_positive_rate': 1.0,
                    'false_positive_rate': 0.0,
                    'true_positive': 2,
                    'true_negative': 2,
                    'false_positive': 0,
                    'false_negative': 0
                },
                'predictions': np.array([0, 1, 0, 1])
            }
        }
        
        evaluator.plot_comprehensive_comparison(
            results_dict, y_true,
            save_name="single_model_comprehensive.png"
        )
        
        assert (output_dir / "single_model_comprehensive.png").exists()


class TestVisualizationIntegration:
    """Integration tests for all visualizations together"""
    
    @pytest.mark.integration
    def test_all_visualizations(self, evaluator, sample_timeline_data, 
                               sample_heatmap_data, sample_comprehensive_data, output_dir):
        """Test generating all visualizations in sequence"""
        # Timeline
        y_true_timeline, y_pred_timeline = sample_timeline_data
        evaluator.plot_attack_detection_timeline(
            y_true_timeline, y_pred_timeline,
            model_name="Integration Test",
            save_name="integration_timeline.png"
        )
        
        # Heatmap
        y_true_heatmap, predictions_dict = sample_heatmap_data
        evaluator.plot_detection_heatmap(
            y_true_heatmap, predictions_dict,
            save_name="integration_heatmap.png"
        )
        
        # Comprehensive
        y_true_comp, results_dict = sample_comprehensive_data
        evaluator.plot_comprehensive_comparison(
            results_dict, y_true_comp,
            save_name="integration_comprehensive.png"
        )
        
        # Verify all created
        assert (output_dir / "integration_timeline.png").exists()
        assert (output_dir / "integration_heatmap.png").exists()
        assert (output_dir / "integration_comprehensive.png").exists()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
