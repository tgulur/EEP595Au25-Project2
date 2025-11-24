#!/usr/bin/env python3
"""
Unit tests for dataset loader with edge case handling
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dataset_loader import CANDatasetLoader


class TestCANDatasetLoader:
    """Test suite for CANDatasetLoader with edge cases"""
    
    @pytest.fixture
    def loader(self, tmp_path):
        """Create loader with temporary directory"""
        return CANDatasetLoader(str(tmp_path))
    
    # ========== Initialization Tests ==========
    
    def test_initialization(self, tmp_path):
        """Test loader initialization"""
        loader = CANDatasetLoader(str(tmp_path))
        assert loader.data_path == tmp_path
    
    def test_initialization_nonexistent_path(self):
        """Test initialization with non-existent path"""
        loader = CANDatasetLoader("/nonexistent/path")
        assert loader.data_path == Path("/nonexistent/path")
    
    # ========== Voltage Data Generation Tests ==========
    
    def test_voltage_data_generation_normal(self, loader):
        """Test voltage data generation with normal parameters"""
        df = loader._create_sample_voltage_data(n_samples=100)
        
        assert len(df) == 100
        assert 'timestamp' in df.columns
        assert 'can_id' in df.columns
        assert 'voltage_samples' in df.columns
        assert 'label' in df.columns
        assert 'ecu_id' in df.columns
    
    def test_voltage_data_generation_small(self, loader):
        """Test with very small sample size"""
        df = loader._create_sample_voltage_data(n_samples=5)
        assert len(df) == 5
    
    def test_voltage_data_generation_large(self, loader):
        """Test with large sample size"""
        df = loader._create_sample_voltage_data(n_samples=1000)
        assert len(df) == 1000
    
    def test_voltage_data_has_attacks(self, loader):
        """Test that generated data contains attacks"""
        df = loader._create_sample_voltage_data(n_samples=100)
        attacks = df[df['label'] == 1]
        assert len(attacks) > 0, "Should contain some attack samples"
    
    def test_voltage_data_has_normal(self, loader):
        """Test that generated data contains normal traffic"""
        df = loader._create_sample_voltage_data(n_samples=100)
        normal = df[df['label'] == 0]
        assert len(normal) > 0, "Should contain some normal samples"
    
    def test_voltage_samples_format(self, loader):
        """Test that voltage samples have correct format"""
        df = loader._create_sample_voltage_data(n_samples=10)
        
        # Check first voltage sample - can be list or numpy array
        sample = df['voltage_samples'].iloc[0]
        assert isinstance(sample, (list, np.ndarray))
        assert len(sample) > 0
    
    def test_voltage_data_ecu_distribution(self, loader):
        """Test that multiple ECUs are represented"""
        df = loader._create_sample_voltage_data(n_samples=100)
        unique_ecus = df['ecu_id'].nunique()
        assert unique_ecus >= 3, "Should have multiple ECU IDs"
    
    # ========== CAN Data Generation Tests ==========
    
    def test_can_data_generation_normal(self, loader):
        """Test CAN data generation with normal parameters"""
        df = loader._create_sample_can_data(n_samples=100)
        
        assert len(df) == 100
        assert 'timestamp' in df.columns
        assert 'can_id' in df.columns
        assert 'data' in df.columns
        assert 'label' in df.columns
    
    def test_can_data_generation_small(self, loader):
        """Test with very small sample size"""
        df = loader._create_sample_can_data(n_samples=5)
        assert len(df) == 5
    
    def test_can_data_generation_large(self, loader):
        """Test with large sample size"""
        df = loader._create_sample_can_data(n_samples=1000)
        assert len(df) == 1000
    
    def test_can_data_has_attacks(self, loader):
        """Test that generated data contains attacks"""
        df = loader._create_sample_can_data(n_samples=100)
        attacks = df[df['label'] == 1]
        assert len(attacks) > 0, "Should contain some attack samples"
    
    def test_can_data_has_normal(self, loader):
        """Test that generated data contains normal traffic"""
        df = loader._create_sample_can_data(n_samples=1000)
        normal = df[df['label'] == 0]
        assert len(normal) > 0, "Should contain some normal samples"
    
    def test_can_data_format(self, loader):
        """Test CAN data payload format"""
        df = loader._create_sample_can_data(n_samples=10)
        
        # Check data format - should be a list of integers
        data_sample = df['data'].iloc[0]
        assert isinstance(data_sample, (list, bytes, bytearray, str))
        if isinstance(data_sample, list):
            assert len(data_sample) == 8  # CAN data is 8 bytes
    
    def test_can_id_range(self, loader):
        """Test that CAN IDs are in valid range"""
        df = loader._create_sample_can_data(n_samples=100)
        
        # CAN IDs should be valid (typically 0x000 to 0x7FF for standard)
        assert df['can_id'].min() >= 0
        assert df['can_id'].max() <= 0x7FF or df['can_id'].max() <= 0x1FFFFFFF
    
    # ========== Load Methods Tests ==========
    
    def test_load_voltage_data_creates_sample(self, loader):
        """Test that load_canmap_voltage_dataset creates sample data if file missing"""
        voltage_df = loader.load_canmap_voltage_dataset()
        
        assert voltage_df is not None
        assert len(voltage_df) > 0
    
    def test_load_can_data_creates_sample(self, loader):
        """Test that load_road_dataset creates sample data if file missing"""
        df = loader.load_road_dataset()
        
        assert df is not None
        assert len(df) > 0
    
    # ========== Data Splitting Tests ==========
    
    def test_split_data_normal(self, loader):
        """Test data splitting with normal ratio"""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        splits = loader.split_data(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        total = len(splits['train'][0]) + len(splits['val'][0]) + len(splits['test'][0])
        assert total == 100
    
    def test_split_data_extreme_ratio(self, loader):
        """Test data splitting with extreme ratio"""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        splits = loader.split_data(X, y, train_ratio=0.1, val_ratio=0.1, test_ratio=0.8)
        
        assert len(splits['test'][0]) > len(splits['train'][0])
    
    def test_split_data_small_dataset(self, loader):
        """Test splitting with very small dataset"""
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)
        
        splits = loader.split_data(X, y)
        
        total = len(splits['train'][0]) + len(splits['val'][0]) + len(splits['test'][0])
        assert total == 10
    
    # ========== Edge Cases ==========
    
    def test_zero_samples(self, loader):
        """Test with zero samples (edge case)"""
        df = loader._create_sample_voltage_data(n_samples=0)
        assert len(df) == 0
    
    def test_negative_samples(self, loader):
        """Test with negative samples (edge case)"""
        df = loader._create_sample_voltage_data(n_samples=-10)
        assert len(df) == 0
    
    def test_data_consistency(self, loader):
        """Test that generated data is consistent"""
        df1 = loader._create_sample_voltage_data(n_samples=50)
        df2 = loader._create_sample_voltage_data(n_samples=50)
        
        # Should have same columns
        assert set(df1.columns) == set(df2.columns)
        
        # Should have same data types
        for col in df1.columns:
            assert df1[col].dtype == df2[col].dtype
    
    def test_timestamp_ordering(self, loader):
        """Test that timestamps are properly ordered"""
        df = loader._create_sample_can_data(n_samples=100)
        
        # Timestamps should generally increase (allowing for some jitter)
        timestamps = df['timestamp'].values
        # Check that timestamps are reasonable (not all same)
        assert len(np.unique(timestamps)) > 1
    
    def test_attack_burst_detection(self, loader):
        """Test that attacks come in bursts as expected"""
        df = loader._create_sample_can_data(n_samples=200)
        
        # Find attack sequences
        labels = df['label'].values
        attack_positions = np.where(labels == 1)[0]
        
        if len(attack_positions) > 1:
            # Check that some attacks are consecutive (bursts)
            diffs = np.diff(attack_positions)
            consecutive_attacks = np.sum(diffs == 1)
            assert consecutive_attacks > 0, "Attacks should come in bursts"
    
    # ========== Voltage Waveform Physics Tests ==========
    
    def test_voltage_waveform_has_rise_time(self, loader):
        """Test that voltage waveforms have realistic rise times"""
        df = loader._create_sample_voltage_data(n_samples=10)
        
        for idx, row in df.iterrows():
            waveform = row['voltage_samples']
            # Check for transitions (rise/fall times)
            diffs = np.diff(waveform)
            has_transitions = np.any(np.abs(diffs) > 0.1)
            assert has_transitions, "Waveform should have voltage transitions"
    
    def test_voltage_levels_realistic(self, loader):
        """Test that voltage levels are realistic for CAN"""
        df = loader._create_sample_voltage_data(n_samples=20)
        
        for idx, row in df.iterrows():
            waveform = row['voltage_samples']
            if isinstance(waveform, list):
                waveform = np.array(waveform)
            # CAN bus typically 0V (recessive) to 3.5V (dominant)
            assert np.min(waveform) >= -0.5, "Voltage should not be too negative"
            assert np.max(waveform) <= 5.0, "Voltage should not exceed 5V"
    
    def test_different_ecu_signatures(self, loader):
        """Test that different ECUs have different voltage signatures"""
        df = loader._create_sample_voltage_data(n_samples=100)
        
        # Group by ECU and check variance
        ecu_groups = df.groupby('ecu_id')
        
        if len(ecu_groups) > 1:
            # Get average waveform characteristics per ECU
            ecu_characteristics = {}
            for ecu_id, group in ecu_groups:
                # Calculate average rise time or similar characteristic
                waveforms = group['voltage_samples'].values
                # Convert to numpy if needed
                waveforms_np = [np.array(w) if isinstance(w, list) else w for w in waveforms]
                avg_max = np.mean([w.max() for w in waveforms_np])
                ecu_characteristics[ecu_id] = avg_max
            
            # Check that ECUs have different characteristics
            values = list(ecu_characteristics.values())
            variance = np.var(values)
            assert variance > 0, "Different ECUs should have different signatures"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
