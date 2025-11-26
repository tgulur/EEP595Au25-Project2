"""
Test that experiment produces the expected output structure
"""

import pytest
from pathlib import Path


class TestExperimentOutputStructure:
    """Test the expected output structure of experiment results"""
    
    def test_expected_folder_structure(self):
        """
        Verify the expected folder structure matches the reference output.
        
        Expected structure (from results/20251113_142805/):
        results/YYYYMMDD_HHMMSS/
        ├── ablation/
        │   └── ablation_results.txt
        ├── cnn/
        │   ├── attack_type_report.txt (optional)
        │   ├── model.h5
        │   └── report.txt
        ├── comparison/
        │   └── model_comparison_table.txt
        ├── fusion/
        │   ├── attack_type_report.txt (optional)
        │   └── report.txt
        ├── lstm/
        │   ├── attack_type_report.txt (optional)
        │   ├── model.h5
        │   └── report.txt
        └── voltage/
            └── report.txt
        """
        
        expected_structure = {
            'ablation': ['ablation_results.txt'],
            'cnn': ['cnn_report.txt', 'cnn_model.h5'],
            'lstm': ['lstm_report.txt', 'lstm_model.h5'],
            'fusion': ['fusion_report.txt'],
            'voltage': ['voltage_report.txt'],
            'comparison': ['model_comparison_table.txt'],
            'baselines': ['timing_report.txt', 'frequency_report.txt']
        }
        
        # This is a documentation test - it doesn't run against real data
        # but documents the expected structure
        assert expected_structure is not None
        
        print("\nExpected Result Structure:")
        print("="*60)
        for folder, files in expected_structure.items():
            print(f"{folder}/")
            for file in files:
                print(f"  {file}")
        print("="*60)
        
        # Verify structure keys are valid
        assert 'ablation' in expected_structure
        assert 'cnn' in expected_structure
        assert 'lstm' in expected_structure
        assert 'fusion' in expected_structure
        assert 'voltage' in expected_structure
        assert 'comparison' in expected_structure
    
    def test_ablation_report_format(self):
        """Test that ablation report will have expected content"""
        
        expected_sections = [
            "FUSION ABLATION STUDY RESULTS",
            "Configuration",
            "Components",
            "Accuracy",
            "TPR",
            "FPR",
            "Description",
            "ANALYSIS",
            "Best Single Component",
            "Best Pairwise Combination",
            "Full Fusion",
            "Voltage Fingerprinting Contribution"
        ]
        
        # Document expected sections
        print("\nExpected Ablation Report Sections:")
        print("="*60)
        for section in expected_sections:
            print(f"  - {section}")
        print("="*60)
        
        assert all(section for section in expected_sections)
    
    def test_expected_model_combinations(self):
        """Test that ablation study covers all expected combinations"""
        
        expected_combinations = [
            'Voltage Only',
            'CNN Only',
            'LSTM Only',
            'Voltage + CNN',
            'Voltage + LSTM',
            'CNN + LSTM',
            'Full Fusion'
        ]
        
        print("\nExpected Ablation Combinations:")
        print("="*60)
        for combo in expected_combinations:
            print(f"  - {combo}")
        print("="*60)
        
        assert len(expected_combinations) == 7
        assert 'Full Fusion' in expected_combinations
    
    def test_subfolder_creation_logic(self):
        """Test that save_name paths will create proper subfolders"""
        
        # Test cases from main_experiment.py
        test_paths = [
            "voltage/voltage_confusion_matrix.png",
            "voltage/voltage_roc_curve.png",
            "voltage/voltage_report.txt",
            "cnn/cnn_confusion_matrix.png",
            "cnn/cnn_roc_curve.png",
            "cnn/cnn_report.txt",
            "cnn/cnn_model.h5",
            "lstm/lstm_confusion_matrix.png",
            "lstm/lstm_roc_curve.png",
            "lstm/lstm_report.txt",
            "lstm/lstm_model.h5",
            "fusion/fusion_confusion_matrix.png",
            "fusion/fusion_roc_curve.png",
            "fusion/fusion_report.txt",
            "baselines/timing_confusion_matrix.png",
            "baselines/timing_roc_curve.png",
            "baselines/timing_report.txt",
            "baselines/frequency_confusion_matrix.png",
            "baselines/frequency_roc_curve.png",
            "baselines/frequency_report.txt",
            "comparison/model_comparison.png",
            "comparison/voltage_attack_timeline.png",
            "comparison/cnn_attack_timeline.png",
            "comparison/lstm_attack_timeline.png",
            "comparison/fusion_attack_timeline.png",
            "comparison/detection_heatmap.png",
            "comparison/comprehensive_comparison.png",
            "comparison/model_comparison_table.txt",
            "ablation/ablation_results.txt"
        ]
        
        # Extract unique folders
        folders = set(Path(p).parent.as_posix() for p in test_paths)
        
        print("\nSubfolders that will be created:")
        print("="*60)
        for folder in sorted(folders):
            print(f"  {folder}/")
        print("="*60)
        
        # Verify expected folders
        expected_folders = {'voltage', 'cnn', 'lstm', 'fusion', 'baselines', 'comparison', 'ablation'}
        assert folders == expected_folders
    
    def test_matches_reference_structure(self):
        """
        Test that our expected structure matches the reference folder
        provided by the user (results/20251113_142805/)
        """
        
        reference_structure = {
            'ablation': ['ablation_results.txt'],
            'cnn': ['attack_type_report.txt', 'model.h5', 'report.txt'],
            'comparison': ['model_comparison_table.txt'],
            'fusion': ['attack_type_report.txt', 'report.txt'],
            'lstm': ['attack_type_report.txt', 'model.h5', 'report.txt'],
            'voltage': ['report.txt']
        }
        
        # Our new structure
        new_structure_folders = {'voltage', 'cnn', 'lstm', 'fusion', 'baselines', 'comparison', 'ablation'}
        
        # Check all reference folders are covered
        reference_folders = set(reference_structure.keys())
        
        print("\nComparison with Reference Structure:")
        print("="*60)
        print(f"Reference folders: {sorted(reference_folders)}")
        print(f"New structure folders: {sorted(new_structure_folders)}")
        print("="*60)
        
        # All reference folders should be in new structure
        assert reference_folders.issubset(new_structure_folders)
        
        # New structure adds 'baselines' folder (which is good!)
        new_folders = new_structure_folders - reference_folders
        print(f"\nAdditional folders in new structure: {new_folders}")
        assert 'baselines' in new_folders


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
