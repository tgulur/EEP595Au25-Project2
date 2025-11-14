#!/usr/bin/env python3
"""
Data alignment validation tests

These tests catch common bugs like:
- Size mismatches between arrays
- Attack types not aligned with labels
- Fusion layer data misalignment
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from evaluation_metrics import IDSEvaluator


def assert_array_sizes_match(*arrays, names=None):
    """
    Helper function to assert all arrays have the same length.
    
    Args:
        *arrays: Variable number of arrays to check
        names: Optional list of names for error messages
    """
    if not arrays:
        return
    
    sizes = [len(arr) for arr in arrays]
    if names is None:
        names = [f"array_{i}" for i in range(len(arrays))]
    
    first_size = sizes[0]
    for size, name in zip(sizes[1:], names[1:]):
        assert size == first_size, \
            f"Size mismatch: {names[0]}={first_size}, {name}={size}"


class TestArraySizeValidation:
    """Test array size validation helpers"""
    
    def test_assert_array_sizes_match_success(self):
        """Test that matching sizes pass"""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        arr3 = np.array([7, 8, 9])
        
        # Should not raise
        assert_array_sizes_match(arr1, arr2, arr3)
    
    def test_assert_array_sizes_match_failure(self):
        """Test that mismatched sizes are caught"""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5])  # Wrong size!
        
        with pytest.raises(AssertionError, match="Size mismatch"):
            assert_array_sizes_match(arr1, arr2, names=['arr1', 'arr2'])


class TestEvaluationDataAlignment:
    """Test that evaluation functions receive correctly aligned data"""
    
    @pytest.fixture
    def evaluator(self, tmp_path):
        return IDSEvaluator(save_dir=str(tmp_path))
    
    def test_calculate_metrics_size_validation(self, evaluator):
        """Test that calculate_metrics validates sizes"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_score = np.array([0.1, 0.9, 0.2, 0.8])
        
        # Should work with matching sizes
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_score)
        assert 'accuracy' in metrics
    
    def test_evaluate_by_attack_type_requires_matching_sizes(self, evaluator):
        """Test that evaluate_by_attack_type requires all arrays to match"""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        attack_types = np.array(['normal', 'dos', 'normal', 'dos', 'normal'])
        
        # Should work
        results = evaluator.evaluate_by_attack_type(y_true, y_pred, attack_types)
        assert isinstance(results, dict)
    
    def test_evaluate_by_attack_type_catches_size_mismatch(self, evaluator):
        """Test that size mismatches are caught"""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        attack_types = np.array(['normal', 'dos'])  # Wrong size!
        
        # Should raise IndexError
        with pytest.raises(IndexError):
            evaluator.evaluate_by_attack_type(y_true, y_pred, attack_types)
    
    def test_evaluate_by_attack_type_with_scores_size_validation(self, evaluator):
        """Test that scores array must also match size"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        attack_types = np.array(['normal', 'dos', 'normal', 'dos'])
        y_score = np.array([0.1, 0.9, 0.2, 0.8])
        
        # Should work with matching sizes
        results = evaluator.evaluate_by_attack_type(y_true, y_pred, attack_types, y_score)
        assert isinstance(results, dict)
        
        # Should fail with mismatched scores
        y_score_wrong = np.array([0.1, 0.9])  # Wrong size!
        with pytest.raises(IndexError):
            evaluator.evaluate_by_attack_type(y_true, y_pred, attack_types, y_score_wrong)


class TestFusionLayerDataAlignment:
    """Test fusion layer data alignment scenarios"""
    
    def test_fusion_min_samples_alignment(self):
        """Test that min_samples alignment works correctly"""
        # Simulate voltage and CAN test data
        n_voltage = 1000
        n_can = 1200
        
        y_voltage = np.random.randint(0, 2, n_voltage)
        y_can = np.random.randint(0, 2, n_can)
        attack_types_can = np.random.choice(['normal', 'dos', 'fuzzing'], n_can)
        
        # Align to min_samples
        min_samples = min(n_voltage, n_can)
        y_can_aligned = y_can[:min_samples]
        attack_types_aligned = attack_types_can[:min_samples]
        
        # Sizes must match
        assert len(y_can_aligned) == min_samples
        assert len(attack_types_aligned) == min_samples
        assert len(y_can_aligned) == len(attack_types_aligned)
    
    def test_fusion_split_alignment(self):
        """Test that split_idx alignment works correctly"""
        n_samples = 1000
        split_idx = int(0.6 * n_samples)  # 600
        
        y_test = np.random.randint(0, 2, n_samples)
        attack_types = np.random.choice(['normal', 'dos'], n_samples)
        
        # After split
        y_test_fusion = y_test[split_idx:]
        attack_types_fusion = attack_types[split_idx:]
        
        # Sizes must match
        expected_size = n_samples - split_idx
        assert len(y_test_fusion) == expected_size
        assert len(attack_types_fusion) == expected_size
        assert len(y_test_fusion) == len(attack_types_fusion)
    
    def test_fusion_full_pipeline_alignment(self):
        """Test complete fusion pipeline alignment"""
        # Simulate full pipeline
        n_can_test = 1500
        n_voltage_test = 1200
        
        y_can_test = np.random.randint(0, 2, n_can_test)
        attack_types_test = np.random.choice(['normal', 'dos', 'fuzzing', 'replay'], n_can_test)
        
        # Step 1: min_samples alignment
        min_samples = min(n_voltage_test, n_can_test)
        y_can_aligned = y_can_test[:min_samples]
        attack_types_aligned = attack_types_test[:min_samples]
        
        # Step 2: split_idx
        split_idx = int(0.6 * min_samples)
        y_test_fusion = y_can_aligned[split_idx:]
        attack_types_fusion = attack_types_aligned[split_idx:]
        
        # Final validation
        assert len(y_test_fusion) == len(attack_types_fusion), \
            f"Final size mismatch: y_test={len(y_test_fusion)}, attack_types={len(attack_types_fusion)}"
        
        # Should be able to use in evaluation
        evaluator = IDSEvaluator()
        predictions = np.random.randint(0, 2, len(y_test_fusion))
        
        # Should not raise
        results = evaluator.evaluate_by_attack_type(y_test_fusion, predictions, attack_types_fusion)
        assert isinstance(results, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

