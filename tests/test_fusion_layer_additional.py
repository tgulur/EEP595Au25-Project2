import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from src.fusion_layer import FusionLayer


def test_create_combiner_models_types():
    f_mlp = FusionLayer(method='stacking', combiner_model='mlp')
    comb1 = f_mlp._create_combiner_model()
    assert isinstance(comb1, MLPClassifier)

    f_rf = FusionLayer(method='stacking', combiner_model='random_forest')
    comb2 = f_rf._create_combiner_model()
    assert isinstance(comb2, RandomForestClassifier)

    f_unknown = FusionLayer(method='stacking', combiner_model='not_a_real_one')
    comb3 = f_unknown._create_combiner_model()
    # Unknown combiner falls back to RandomForest
    assert isinstance(comb3, RandomForestClassifier)


def test_train_adaptive_weights_search():
    np.random.seed(1)
    n = 60
    voltage_scores = np.random.rand(n)
    dl_scores = np.random.rand(n)
    v_conf = np.random.uniform(0.6, 1.0, n)
    dl_conf = np.random.uniform(0.6, 1.0, n)
    labels = ((voltage_scores + dl_scores) > 1.0).astype(int)

    f = FusionLayer(method='adaptive')
    f.train(voltage_scores, dl_scores, v_conf, dl_conf, labels)

    # weights should be between 0 and 1 and sum to 1
    assert 0.0 <= f.weights['voltage'] <= 1.0
    assert 0.0 <= f.weights['dl'] <= 1.0
    assert pytest.approx(f.weights['voltage'] + f.weights['dl'], rel=1e-6) == 1.0


def test_train_single_class_switch_to_voting_and_predict():
    np.random.seed(2)
    n = 30
    vs = np.random.rand(n)
    ds = np.random.rand(n)
    v_conf = np.ones(n) * 0.8
    dl_conf = np.ones(n) * 0.9
    labels = np.zeros(n, dtype=int)  # single class only

    f = FusionLayer(method='stacking', combiner_model='random_forest')
    f.train(vs, ds, v_conf, dl_conf, labels)

    # Should have fallen back to voting mode
    assert f.method == 'voting'
    assert f.trained is True

    # Voting predict should work without errors
    p, c = f.predict(0.6, 0.7, 0.8, 0.85)
    assert p in (0, 1)
    assert 0.0 <= c <= 1.0


def test_predict_stacking_not_trained_raises_and_unknown_method():
    f = FusionLayer(method='stacking')
    # Not trained, should raise
    with pytest.raises(ValueError):
        f.predict(0.6, 0.6, 0.8, 0.8)

    f2 = FusionLayer(method='definitely_unknown')
    with pytest.raises(ValueError):
        f2.predict(0.6, 0.6, 0.8, 0.8)


def test_stacking_predict_without_predict_proba_uses_avg_confidence():
    # Train a normal stacking combiner to get shapes right
    np.random.seed(3)
    n = 50
    vs = np.random.rand(n)
    ds = np.random.rand(n)
    v_conf = np.random.uniform(0.6, 1.0, n)
    dl_conf = np.random.uniform(0.6, 1.0, n)
    labels = ((vs + ds) > 1.0).astype(int)

    f = FusionLayer(method='stacking', combiner_model='random_forest')
    f.train(vs, ds, v_conf, dl_conf, labels)

    # Replace combiner with a dummy that only has predict (no predict_proba)
    class DummyCombiner:
        def predict(self, X):
            # return alternating zeros/ones
            return np.array([int(x[0] > 0.5) for x in X])

    f.combiner = DummyCombiner()
    f.trained = True

    pred, conf = f.predict(0.9, 0.1, 0.7, 0.6)
    # confidence should equal average of the provided confidences when predict_proba absent
    assert pytest.approx(conf, rel=1e-6) == (0.7 + 0.6) / 2


def test_update_weights_adaptive_noop_for_non_weighted():
    f = FusionLayer(method='stacking')
    old = dict(f.weights)
    f.update_weights_adaptive(0.9, 0.1)
    # weights unchanged
    assert f.weights == old
