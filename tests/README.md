# Test Suite

## Overview

The test suite includes comprehensive tests to catch common bugs and ensure data integrity throughout the pipeline.

## Test Files

### Core Tests
- `test_basic.py` - Basic functionality tests
- `test_dataset_loader.py` - Dataset loading and preprocessing tests
- `test_evaluation_metrics.py` - Evaluation metrics tests
- `test_voltage_fingerprinting.py` - Voltage fingerprinting tests

### Data Alignment Tests (NEW)
- `test_attack_type_tracking.py` - Tests for attack type preservation and tracking
- `test_data_alignment.py` - Tests for array size validation and alignment

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Files
```bash
# Test attack type tracking
pytest tests/test_attack_type_tracking.py -v

# Test data alignment
pytest tests/test_data_alignment.py -v

# Run both new test files
pytest tests/test_attack_type_tracking.py tests/test_data_alignment.py -v
```

### Run Tests Before Committing
```bash
# Quick check - run data alignment tests
pytest tests/test_attack_type_tracking.py tests/test_data_alignment.py -v

# Full test suite
pytest tests/ -v
```

## What These Tests Catch

### 1. Size Mismatches
- Arrays (y_true, y_pred, attack_types) must have matching lengths
- Fusion layer data alignment issues
- Preprocessing pipeline size consistency

### 2. Attack Type Preservation
- Attack types are preserved through preprocessing
- Attack types align correctly with labels
- Attack types are preserved through train/val/test splits

### 3. Data Alignment
- Fusion layer receives correctly aligned data
- min_samples alignment works correctly
- split_idx alignment works correctly

### 4. Evaluation Integrity
- Per-attack-type evaluation receives matching arrays
- Metrics are calculated correctly per attack type
- All attack types are evaluated

## Example: Catching the Fusion Layer Bug

The fusion layer bug we encountered would have been caught by:

```python
def test_fusion_pipeline_alignment(self, loader):
    """Test fusion layer pipeline with proper alignment"""
    # ... test code ...
    
    # This assertion would have caught the bug:
    assert len(y_test_fusion) == len(attack_types_fusion), \
        f"Fusion size mismatch: y_test={len(y_test_fusion)}, attack_types={len(attack_types_fusion)}"
```

## Best Practices

1. **Run tests before committing code changes**
   ```bash
   pytest tests/test_attack_type_tracking.py tests/test_data_alignment.py -v
   ```

2. **Run full test suite before major changes**
   ```bash
   pytest tests/ -v
   ```

3. **Add new tests when adding new features**
   - If you add a new preprocessing step, add a test
   - If you add a new evaluation metric, add a test
   - If you modify data flow, add alignment tests

4. **Use tests to validate fixes**
   - After fixing a bug, add a test that would have caught it
   - This prevents regression

## Continuous Integration

These tests can be integrated into CI/CD pipelines to automatically catch issues before they reach production.

---

*Last updated: 2025-11-13*

