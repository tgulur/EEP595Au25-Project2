"""
Test that results are automatically organized into subfolders
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
import numpy as np
from src.evaluation_metrics import IDSEvaluator


class TestResultOrganization:
    """Test automatic subfolder organization for experiment results"""
    
    @pytest.fixture
    def temp_results_dir(self):
        """Create temporary results directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_score = np.random.rand(100)
        return y_true, y_pred, y_score
    
    def test_evaluator_creates_subdirectories(self, temp_results_dir, sample_data):
        """Test that IDSEvaluator creates subdirectories when save_name contains paths"""
        evaluator = IDSEvaluator(temp_results_dir)
        y_true, y_pred, y_score = sample_data
        
        # Test voltage subfolder
        evaluator.plot_confusion_matrix(
            y_true, y_pred,
            title="Test Confusion Matrix",
            save_name="voltage/confusion_matrix.png"
        )
        
        # Test CNN subfolder
        evaluator.plot_roc_curve(
            y_true, y_score,
            title="Test ROC Curve",
            save_name="cnn/roc_curve.png"
        )
        
        # Test LSTM subfolder
        evaluator.generate_report(
            "Test Model",
            {'accuracy': 0.95, 'precision': 0.94},
            save_name="lstm/report.txt"
        )
        
        # Test fusion subfolder
        evaluator.plot_confusion_matrix(
            y_true, y_pred,
            save_name="fusion/confusion.png"
        )
        
        # Test baselines subfolder
        evaluator.plot_roc_curve(
            y_true, y_score,
            save_name="baselines/timing_roc.png"
        )
        
        # Test comparison subfolder
        evaluator.generate_report(
            "Comparison",
            {'accuracy': 0.96},
            save_name="comparison/comparison_table.txt"
        )
        
        # Verify all subdirectories were created
        results_path = Path(temp_results_dir)
        
        assert (results_path / "voltage").exists(), "voltage/ subfolder not created"
        assert (results_path / "cnn").exists(), "cnn/ subfolder not created"
        assert (results_path / "lstm").exists(), "lstm/ subfolder not created"
        assert (results_path / "fusion").exists(), "fusion/ subfolder not created"
        assert (results_path / "baselines").exists(), "baselines/ subfolder not created"
        assert (results_path / "comparison").exists(), "comparison/ subfolder not created"
    
    def test_voltage_folder_structure(self, temp_results_dir, sample_data):
        """Test voltage folder gets created with expected files"""
        evaluator = IDSEvaluator(temp_results_dir)
        y_true, y_pred, y_score = sample_data
        
        # Create voltage files
        evaluator.plot_confusion_matrix(
            y_true, y_pred,
            save_name="voltage/voltage_confusion_matrix.png"
        )
        evaluator.plot_roc_curve(
            y_true, y_score,
            save_name="voltage/voltage_roc_curve.png"
        )
        evaluator.generate_report(
            "Voltage Fingerprinting",
            {'accuracy': 0.95, 'precision': 0.94, 'recall': 0.93},
            save_name="voltage/voltage_report.txt"
        )
        
        voltage_dir = Path(temp_results_dir) / "voltage"
        assert voltage_dir.exists()
        assert (voltage_dir / "voltage_confusion_matrix.png").exists()
        assert (voltage_dir / "voltage_roc_curve.png").exists()
        assert (voltage_dir / "voltage_report.txt").exists()
    
    def test_cnn_folder_structure(self, temp_results_dir, sample_data):
        """Test CNN folder gets created with expected files"""
        evaluator = IDSEvaluator(temp_results_dir)
        y_true, y_pred, y_score = sample_data
        
        # Create CNN files
        evaluator.plot_confusion_matrix(
            y_true, y_pred,
            save_name="cnn/cnn_confusion_matrix.png"
        )
        evaluator.plot_roc_curve(
            y_true, y_score,
            save_name="cnn/cnn_roc_curve.png"
        )
        evaluator.generate_report(
            "CNN",
            {'accuracy': 0.96, 'precision': 0.95, 'recall': 0.94},
            save_name="cnn/cnn_report.txt"
        )
        
        cnn_dir = Path(temp_results_dir) / "cnn"
        assert cnn_dir.exists()
        assert (cnn_dir / "cnn_confusion_matrix.png").exists()
        assert (cnn_dir / "cnn_roc_curve.png").exists()
        assert (cnn_dir / "cnn_report.txt").exists()
    
    def test_lstm_folder_structure(self, temp_results_dir, sample_data):
        """Test LSTM folder gets created with expected files"""
        evaluator = IDSEvaluator(temp_results_dir)
        y_true, y_pred, y_score = sample_data
        
        # Create LSTM files
        evaluator.plot_confusion_matrix(
            y_true, y_pred,
            save_name="lstm/lstm_confusion_matrix.png"
        )
        evaluator.plot_roc_curve(
            y_true, y_score,
            save_name="lstm/lstm_roc_curve.png"
        )
        evaluator.generate_report(
            "LSTM",
            {'accuracy': 0.97, 'precision': 0.96, 'recall': 0.95},
            save_name="lstm/lstm_report.txt"
        )
        
        lstm_dir = Path(temp_results_dir) / "lstm"
        assert lstm_dir.exists()
        assert (lstm_dir / "lstm_confusion_matrix.png").exists()
        assert (lstm_dir / "lstm_roc_curve.png").exists()
        assert (lstm_dir / "lstm_report.txt").exists()
    
    def test_fusion_folder_structure(self, temp_results_dir, sample_data):
        """Test fusion folder gets created with expected files"""
        evaluator = IDSEvaluator(temp_results_dir)
        y_true, y_pred, y_score = sample_data
        
        # Create fusion files
        evaluator.plot_confusion_matrix(
            y_true, y_pred,
            save_name="fusion/fusion_confusion_matrix.png"
        )
        evaluator.plot_roc_curve(
            y_true, y_score,
            save_name="fusion/fusion_roc_curve.png"
        )
        evaluator.generate_report(
            "Fusion Layer",
            {'accuracy': 0.98, 'precision': 0.97, 'recall': 0.96},
            save_name="fusion/fusion_report.txt"
        )
        
        fusion_dir = Path(temp_results_dir) / "fusion"
        assert fusion_dir.exists()
        assert (fusion_dir / "fusion_confusion_matrix.png").exists()
        assert (fusion_dir / "fusion_roc_curve.png").exists()
        assert (fusion_dir / "fusion_report.txt").exists()
    
    def test_baselines_folder_structure(self, temp_results_dir, sample_data):
        """Test baselines folder gets created with expected files"""
        evaluator = IDSEvaluator(temp_results_dir)
        y_true, y_pred, y_score = sample_data
        
        # Create baseline files
        evaluator.plot_confusion_matrix(
            y_true, y_pred,
            save_name="baselines/timing_confusion_matrix.png"
        )
        evaluator.plot_roc_curve(
            y_true, y_score,
            save_name="baselines/timing_roc_curve.png"
        )
        evaluator.generate_report(
            "Timing-Based IDS",
            {'accuracy': 0.92},
            save_name="baselines/timing_report.txt"
        )
        
        evaluator.plot_confusion_matrix(
            y_true, y_pred,
            save_name="baselines/frequency_confusion_matrix.png"
        )
        evaluator.plot_roc_curve(
            y_true, y_score,
            save_name="baselines/frequency_roc_curve.png"
        )
        evaluator.generate_report(
            "Frequency-Based IDS",
            {'accuracy': 0.91},
            save_name="baselines/frequency_report.txt"
        )
        
        baselines_dir = Path(temp_results_dir) / "baselines"
        assert baselines_dir.exists()
        assert (baselines_dir / "timing_confusion_matrix.png").exists()
        assert (baselines_dir / "timing_roc_curve.png").exists()
        assert (baselines_dir / "timing_report.txt").exists()
        assert (baselines_dir / "frequency_confusion_matrix.png").exists()
        assert (baselines_dir / "frequency_roc_curve.png").exists()
        assert (baselines_dir / "frequency_report.txt").exists()
    
    def test_comparison_folder_structure(self, temp_results_dir, sample_data):
        """Test comparison folder gets created with expected files"""
        evaluator = IDSEvaluator(temp_results_dir)
        y_true, y_pred, y_score = sample_data
        
        # Create comparison files
        comparison_results = {
            'Voltage': {'accuracy': 0.95, 'precision': 0.94},
            'CNN': {'accuracy': 0.96, 'precision': 0.95},
            'LSTM': {'accuracy': 0.97, 'precision': 0.96}
        }
        
        evaluator.compare_models(
            comparison_results,
            save_name="comparison/model_comparison.png"
        )
        
        evaluator.plot_attack_detection_timeline(
            y_true, y_pred,
            model_name="Voltage",
            save_name="comparison/voltage_attack_timeline.png"
        )
        
        evaluator.plot_detection_heatmap(
            y_true,
            {'CNN': y_pred, 'LSTM': y_pred},
            save_name="comparison/detection_heatmap.png"
        )
        
        # Create comparison table
        evaluator.generate_report(
            "Model Comparison",
            {'accuracy': 0.97},
            save_name="comparison/model_comparison_table.txt"
        )
        
        comparison_dir = Path(temp_results_dir) / "comparison"
        assert comparison_dir.exists()
        assert (comparison_dir / "model_comparison.png").exists()
        assert (comparison_dir / "voltage_attack_timeline.png").exists()
        assert (comparison_dir / "detection_heatmap.png").exists()
        assert (comparison_dir / "model_comparison_table.txt").exists()
    
    def test_complete_experiment_structure(self, temp_results_dir, sample_data):
        """Test complete experiment folder structure matches expected layout"""
        evaluator = IDSEvaluator(temp_results_dir)
        y_true, y_pred, y_score = sample_data
        
        # Expected subfolder structure from attached folder
        expected_subfolders = [
            "voltage",
            "cnn",
            "lstm",
            "fusion",
            "baselines",
            "comparison"
        ]
        
        # Create files in each subfolder
        for subfolder in expected_subfolders:
            evaluator.plot_confusion_matrix(
                y_true, y_pred,
                save_name=f"{subfolder}/test_confusion.png"
            )
            evaluator.generate_report(
                f"{subfolder.title()} Model",
                {'accuracy': 0.95},
                save_name=f"{subfolder}/report.txt"
            )
        
        # Verify all expected subfolders exist
        results_path = Path(temp_results_dir)
        for subfolder in expected_subfolders:
            subfolder_path = results_path / subfolder
            assert subfolder_path.exists(), f"{subfolder}/ not created"
            assert subfolder_path.is_dir(), f"{subfolder}/ is not a directory"
            assert (subfolder_path / "report.txt").exists(), f"{subfolder}/report.txt not found"
    
    def test_nested_subfolder_creation(self, temp_results_dir, sample_data):
        """Test that deeply nested paths are created correctly"""
        evaluator = IDSEvaluator(temp_results_dir)
        y_true, y_pred, y_score = sample_data
        
        # Test deeply nested path
        evaluator.plot_confusion_matrix(
            y_true, y_pred,
            save_name="comparison/deep/nested/path/confusion.png"
        )
        
        nested_path = Path(temp_results_dir) / "comparison" / "deep" / "nested" / "path"
        assert nested_path.exists()
        assert (nested_path / "confusion.png").exists()
    
    def test_ablation_folder_structure(self, temp_results_dir, sample_data):
        """Test ablation study folder gets created"""
        evaluator = IDSEvaluator(temp_results_dir)
        y_true, y_pred, y_score = sample_data
        
        # Create ablation files
        evaluator.generate_report(
            "Ablation Study",
            {'accuracy': 0.96},
            save_name="ablation/ablation_results.txt"
        )
        
        ablation_dir = Path(temp_results_dir) / "ablation"
        assert ablation_dir.exists()
        assert (ablation_dir / "ablation_results.txt").exists()
    
    def test_prints_directory_structure(self, temp_results_dir, sample_data, capsys):
        """Test helper to visualize the created directory structure"""
        evaluator = IDSEvaluator(temp_results_dir)
        y_true, y_pred, y_score = sample_data
        
        # Create the expected structure
        subfolders = {
            "voltage": ["voltage_report.txt"],
            "cnn": ["report.txt", "attack_type_report.txt", "model.h5"],
            "lstm": ["report.txt", "attack_type_report.txt", "model.h5"],
            "fusion": ["report.txt", "attack_type_report.txt"],
            "comparison": ["model_comparison_table.txt"],
            "ablation": ["ablation_results.txt"]
        }
        
        for subfolder, files in subfolders.items():
            for filename in files:
                if filename.endswith('.txt'):
                    evaluator.generate_report(
                        f"{subfolder} model",
                        {'accuracy': 0.95},
                        save_name=f"{subfolder}/{filename}"
                    )
                elif filename.endswith('.h5'):
                    # Create dummy model file
                    model_path = Path(temp_results_dir) / subfolder / filename
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    model_path.touch()
        
        # Print structure
        print("\nCreated directory structure:")
        for root, dirs, files in sorted(os.walk(temp_results_dir)):
            level = root.replace(temp_results_dir, '').count(os.sep)
            indent = '\t' * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = '\t' * (level + 1)
            for file in sorted(files):
                print(f'{subindent}{file}')
        
        # Verify structure
        results_path = Path(temp_results_dir)
        for subfolder in subfolders.keys():
            assert (results_path / subfolder).exists(), f"{subfolder}/ not created"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
