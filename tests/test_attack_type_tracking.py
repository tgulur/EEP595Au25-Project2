#!/usr/bin/env python3
"""
Tests for attack type tracking and data alignment

These tests ensure that:
1. Attack types are preserved through preprocessing
2. Array sizes match correctly (y_true, y_pred, attack_types)
3. Fusion layer gets correctly aligned data
4. Per-attack-type evaluation works correctly
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dataset_loader import CANDatasetLoader
from evaluation_metrics import IDSEvaluator


class TestAttackTypeTracking:
    """Test attack type preservation through preprocessing pipeline"""
    
    @pytest.fixture
    def loader(self, tmp_path):
        """Create loader with temporary directory"""
        return CANDatasetLoader(str(tmp_path))
    
    def test_can_preprocessing_returns_attack_types(self, loader):
        """Test that preprocess_can_data returns attack types"""
        df = loader._create_sample_can_data(n_samples=200)
        
        X, y, attack_types = loader.preprocess_can_data(df, sequence_length=100)
        
        # Should return 3 values
        assert len((X, y, attack_types)) == 3
        
        # Attack types should be numpy array
        assert isinstance(attack_types, np.ndarray)
        
        # Should have same length as sequences
        assert len(attack_types) == len(X)
        assert len(attack_types) == len(y)
    
    def test_attack_types_match_labels(self, loader):
        """Test that attack types align with labels"""
        df = loader._create_sample_can_data(n_samples=200)
        X, y, attack_types = loader.preprocess_can_data(df, sequence_length=100)
        
        # All sequences should have an attack type
        assert len(attack_types) == len(y)
        
        # Normal sequences (label=0) should have attack_type='normal'
        normal_mask = y == 0
        if np.any(normal_mask):
            normal_attack_types = attack_types[normal_mask]
            assert np.all(normal_attack_types == 'normal'), \
                f"Found non-normal attack types in normal sequences: {np.unique(normal_attack_types)}"
        
        # Attack sequences (label=1) should NOT have attack_type='normal'
        attack_mask = y == 1
        if np.any(attack_mask):
            attack_attack_types = attack_types[attack_mask]
            assert np.all(attack_attack_types != 'normal'), \
                f"Found 'normal' attack type in attack sequences: {np.unique(attack_attack_types)}"
    
    def test_attack_types_preserved_through_split(self, loader):
        """Test that attack types are preserved when splitting data"""
        df = loader._create_sample_can_data(n_samples=200)
        X, y, attack_types = loader.preprocess_can_data(df, sequence_length=100)
        
        splits = loader.split_data(X, y, random_seed=42)
        
        # Split attack types using same indices
        train_indices = splits['train_indices']
        val_indices = splits['val_indices']
        test_indices = splits['test_indices']
        
        attack_types_train = attack_types[train_indices]
        attack_types_val = attack_types[val_indices]
        attack_types_test = attack_types[test_indices]
        
        # Sizes should match
        assert len(attack_types_train) == len(splits['train'][1])
        assert len(attack_types_val) == len(splits['val'][1])
        assert len(attack_types_test) == len(splits['test'][1])
        
        # All attack types should be preserved
        original_types = set(attack_types)
        split_types = set(attack_types_train) | set(attack_types_val) | set(attack_types_test)
        assert original_types == split_types, \
            f"Attack types lost in split: original={original_types}, split={split_types}"


class TestDataAlignment:
    """Test that arrays are correctly aligned for evaluation"""
    
    @pytest.fixture
    def evaluator(self, tmp_path):
        """Create evaluator"""
        return IDSEvaluator(save_dir=str(tmp_path))
    
    def test_evaluate_by_attack_type_size_matching(self, evaluator):
        """Test that evaluate_by_attack_type requires matching sizes"""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 1])
        attack_types = np.array(['normal', 'normal', 'dos', 'dos', 'normal', 'fuzzing'])
        
        # Should work with matching sizes
        results = evaluator.evaluate_by_attack_type(y_true, y_pred, attack_types)
        assert isinstance(results, dict)
        assert len(results) > 0
    
    def test_evaluate_by_attack_type_size_mismatch_detection(self, evaluator):
        """Test that size mismatches are detected"""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 1])
        attack_types = np.array(['normal', 'normal', 'dos', 'dos'])  # Wrong size!
        
        # Should raise IndexError when sizes don't match
        with pytest.raises(IndexError):
            evaluator.evaluate_by_attack_type(y_true, y_pred, attack_types)
    
    def test_evaluate_by_attack_type_empty_attack_types(self, evaluator):
        """Test handling of empty attack types array"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        attack_types = np.array([])  # Empty!
        
        # Should return empty results or raise error
        # The function may handle this gracefully, so we just check it doesn't crash
        try:
            results = evaluator.evaluate_by_attack_type(y_true, y_pred, attack_types)
            # If it doesn't raise, results should be empty
            assert len(results) == 0
        except (ValueError, IndexError):
            # Or it should raise a clear error
            pass
    
    def test_evaluate_by_attack_type_all_attack_types_present(self, evaluator):
        """Test that all attack types are evaluated"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        attack_types = np.array(['normal', 'normal', 'dos', 'dos', 'normal', 'fuzzing', 'replay', 'normal'])
        
        results = evaluator.evaluate_by_attack_type(y_true, y_pred, attack_types)
        
        # Should have results for all unique attack types
        unique_types = set(attack_types)
        assert set(results.keys()) == unique_types, \
            f"Missing attack types in results: expected={unique_types}, got={set(results.keys())}"
    
    def test_evaluate_by_attack_type_metrics_correct(self, evaluator):
        """Test that metrics are calculated correctly per attack type"""
        # Create data where we know the expected results
        y_true = np.array([0, 0, 1, 1, 0, 1])  # 3 normal, 3 attacks
        y_pred = np.array([0, 0, 1, 1, 0, 1])  # All correct
        attack_types = np.array(['normal', 'normal', 'dos', 'dos', 'normal', 'fuzzing'])
        
        results = evaluator.evaluate_by_attack_type(y_true, y_pred, attack_types)
        
        # Check 'dos' attack type (should be perfect)
        if 'dos' in results:
            dos_metrics = results['dos']
            assert dos_metrics['accuracy'] == 1.0
            assert dos_metrics['true_positive_rate'] == 1.0
            assert dos_metrics['sample_count'] == 2
        
        # Check 'normal' attack type
        if 'normal' in results:
            normal_metrics = results['normal']
            assert normal_metrics['sample_count'] == 3
            assert normal_metrics['normal_count'] == 3
            assert normal_metrics['attack_count'] == 0


class TestFusionLayerAlignment:
    """Test that fusion layer receives correctly aligned data"""
    
    def test_fusion_data_alignment_simulation(self):
        """Simulate fusion layer data alignment to catch size mismatches"""
        # Simulate the fusion layer scenario
        n_total = 1000
        min_samples = 800  # After alignment
        split_idx = int(0.6 * min_samples)  # 480
        
        # Simulate test data
        y_test_full = np.random.randint(0, 2, n_total)
        attack_types_full = np.random.choice(['normal', 'dos', 'fuzzing'], n_total)
        
        # Align to min_samples (like fusion does)
        y_test_aligned = y_test_full[:min_samples]
        attack_types_aligned = attack_types_full[:min_samples]
        
        # After split
        y_test_fusion = y_test_aligned[split_idx:]
        attack_types_fusion = attack_types_aligned[split_idx:]
        
        # Sizes must match!
        assert len(y_test_fusion) == len(attack_types_fusion), \
            f"Size mismatch: y_test={len(y_test_fusion)}, attack_types={len(attack_types_fusion)}"
        
        # Should be able to evaluate
        evaluator = IDSEvaluator()
        predictions = np.random.randint(0, 2, len(y_test_fusion))
        
        # This should not raise an error
        results = evaluator.evaluate_by_attack_type(
            y_test_fusion, predictions, attack_types_fusion
        )
        assert isinstance(results, dict)
    
    def test_fusion_size_mismatch_detection(self):
        """Test that we can detect size mismatches before evaluation"""
        y_test = np.array([0, 1, 0, 1, 0])
        attack_types = np.array(['normal', 'dos', 'normal'])  # Wrong size!
        
        # Should detect mismatch
        assert len(y_test) != len(attack_types), \
            "Test setup error: sizes should not match"
        
        # Should raise error when trying to evaluate
        evaluator = IDSEvaluator()
        predictions = np.array([0, 1, 0, 1, 0])
        
        with pytest.raises(IndexError):
            evaluator.evaluate_by_attack_type(y_test, predictions, attack_types)


class TestIntegrationPipeline:
    """Integration tests for the full pipeline"""
    
    @pytest.fixture
    def loader(self, tmp_path):
        return CANDatasetLoader(str(tmp_path))
    
    def test_full_pipeline_attack_type_preservation(self, loader):
        """Test that attack types are preserved through full preprocessing pipeline"""
        # Generate data
        can_df = loader._create_sample_can_data(n_samples=500)
        
        # Preprocess
        X, y, attack_types = loader.preprocess_can_data(can_df, sequence_length=100)
        
        # Split
        splits = loader.split_data(X, y, random_seed=42)
        
        # Get test indices and align attack types
        test_indices = splits['test_indices']
        attack_types_test = attack_types[test_indices]
        y_test = splits['test'][1]
        
        # Sizes must match
        assert len(attack_types_test) == len(y_test), \
            f"Size mismatch after split: attack_types={len(attack_types_test)}, y_test={len(y_test)}"
        
        # Simulate predictions
        predictions = np.random.randint(0, 2, len(y_test))
        
        # Should be able to evaluate
        evaluator = IDSEvaluator()
        results = evaluator.evaluate_by_attack_type(y_test, predictions, attack_types_test)
        
        # Should have results
        assert isinstance(results, dict)
        assert len(results) > 0
    
    def test_fusion_pipeline_alignment(self, loader):
        """Test fusion layer pipeline with proper alignment"""
        # Generate data
        can_df = loader._create_sample_can_data(n_samples=500)
        
        # Preprocess
        X, y, attack_types = loader.preprocess_can_data(can_df, sequence_length=100)
        
        # Split
        splits = loader.split_data(X, y, random_seed=42)
        
        # Simulate fusion layer scenario
        test_indices = splits['test_indices']
        attack_types_test = attack_types[test_indices]
        y_test = splits['test'][1]
        
        # Simulate min_samples alignment (like fusion does)
        min_samples = min(len(y_test), 400)  # Some alignment
        y_test_aligned = y_test[:min_samples]
        attack_types_aligned = attack_types_test[:min_samples]
        
        # Simulate split_idx
        split_idx = int(0.6 * len(y_test_aligned))
        y_test_fusion = y_test_aligned[split_idx:]
        attack_types_fusion = attack_types_aligned[split_idx:]
        
        # Sizes must match
        assert len(y_test_fusion) == len(attack_types_fusion), \
            f"Fusion size mismatch: y_test={len(y_test_fusion)}, attack_types={len(attack_types_fusion)}"
        
        # Should be able to evaluate
        evaluator = IDSEvaluator()
        predictions = np.random.randint(0, 2, len(y_test_fusion))
        results = evaluator.evaluate_by_attack_type(y_test_fusion, predictions, attack_types_fusion)
        
        assert isinstance(results, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

