import numpy as np
from pathlib import Path

from src.evaluation import _align_predictions


class DummyModelResults:
    def __init__(self, indices=None, timestamps=None, y_test=None, predictions=None, scores=None):
        self.test_indices = indices
        self.test_timestamps = timestamps
        self.y_test = y_test
        self.predictions = predictions
        self.scores = scores


def make_results(voltage=None, cnn=None, lstm=None, fusion=None):
    results = {}
    if voltage is not None:
        results['voltage'] = {
            'test_indices': voltage.test_indices,
            'test_timestamps': voltage.test_timestamps,
            'y_test': voltage.y_test,
            'predictions': voltage.predictions,
            'scores': voltage.scores
        }
    if cnn is not None or lstm is not None:
        results['deep_learning'] = {}
        if cnn is not None:
            results['deep_learning']['CNN'] = {
                'test_indices': cnn.test_indices,
                'test_timestamps': cnn.test_timestamps,
                'y_test': cnn.y_test,
                'predictions': cnn.predictions,
                'scores': cnn.scores
            }
        if lstm is not None:
            results['deep_learning']['LSTM'] = {
                'test_indices': lstm.test_indices,
                'test_timestamps': lstm.test_timestamps,
                'y_test': lstm.y_test,
                'predictions': lstm.predictions,
                'scores': lstm.scores
            }
    if fusion is not None:
        results['fusion'] = {
            'test_indices': fusion.test_indices,
            'test_timestamps': fusion.test_timestamps,
            'y_test': fusion.y_test,
            'predictions': fusion.predictions,
            'scores': fusion.scores
        }
    return results


def test_timestamp_alignment_intersection():
    # Create three models that share some timestamps
    common_ts = np.array([1.000000, 2.000000, 3.000000])

    v = DummyModelResults(
        indices=np.array([10, 11, 12]),
        timestamps=np.array([1.0, 2.0, 3.0]),
        y_test=np.array([0, 1, 0]),
        predictions=np.array([0, 1, 1]),
        scores=np.array([0.1, 0.9, 0.2])
    )
    c = DummyModelResults(
        indices=np.array([100, 101, 102, 103]),
        timestamps=np.array([0.5, 1.0, 2.0, 4.0]),
        y_test=np.array([0, 0, 1, 0]),
        predictions=np.array([0, 0, 1, 0]),
        scores=np.array([0.2, 0.3, 0.8, 0.1])
    )
    l = DummyModelResults(
        indices=np.array([200, 201, 202]),
        timestamps=np.array([1.0, 2.0, 3.0]),
        y_test=np.array([0, 1, 0]),
        predictions=np.array([0, 1, 0]),
        scores=np.array([0.15, 0.85, 0.25])
    )

    results = make_results(voltage=v, cnn=c, lstm=l)

    aligned = _align_predictions(results)
    assert aligned is not None
    # common timestamps should be [1.0,2.0] because CNN lacks timestamp 3.0
    np.testing.assert_array_almost_equal(np.sort(aligned['common_indices']), np.array([1.0, 2.0]))

    # Check that aligned predictions have length 2
    assert 'Voltage' in aligned['aligned']
    assert 'CNN' in aligned['aligned']
    assert 'LSTM' in aligned['aligned']
    assert len(aligned['aligned']['Voltage']['predictions']) == 2
    assert len(aligned['aligned']['CNN']['predictions']) == 2
    assert len(aligned['aligned']['LSTM']['predictions']) == 2


def test_timestamp_alignment_no_overlap_fallback():
    # Two models with disjoint timestamps (no common ts)
    # Provide no numeric indices but disjoint timestamps -> alignment should be None
    v = DummyModelResults(
        indices=None,
        timestamps=np.array([10.0, 11.0]),
        y_test=np.array([0, 1]),
        predictions=np.array([0, 1]),
        scores=np.array([0.1, 0.9])
    )
    c = DummyModelResults(
        indices=None,
        timestamps=np.array([20.0, 21.0]),
        y_test=np.array([0, 1]),
        predictions=np.array([0, 0]),
        scores=np.array([0.2, 0.3])
    )

    results = make_results(voltage=v, cnn=c)
    aligned = _align_predictions(results)
    # Should return None because there's no timestamp intersection either
    assert aligned is None
