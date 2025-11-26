import numpy as np
import pytest

from src.fusion_layer import FusionLayer, AdaptiveFusion


def test_extract_fusion_features_and_trend():
    f = FusionLayer(method='stacking', combiner_model='random_forest')
    # call several times to populate history and hit trend branch
    for i in range(6):
        feats = f.extract_fusion_features(0.2 + i * 0.1, 0.3 + i * 0.05, 0.8, 0.9)
    # Expect feature vector length 14 (10 base + 4 trend)
    assert feats.shape[0] == 14


def test_stacking_train_predict_and_predict_batch():
    np.random.seed(0)
    n = 40
    voltage_scores = np.random.rand(n)
    dl_scores = np.random.rand(n)
    v_conf = np.random.uniform(0.6, 1.0, n)
    dl_conf = np.random.uniform(0.6, 1.0, n)
    # labels include both classes
    labels = (voltage_scores + dl_scores > 1.0).astype(int)

    f = FusionLayer(method='stacking', combiner_model='random_forest')
    f.train(voltage_scores, dl_scores, v_conf, dl_conf, labels)

    # single prediction
    pred, conf = f.predict(0.7, 0.3, 0.9, 0.8)
    assert pred in (0, 1)
    assert 0.0 <= conf <= 1.0

    # batch predict
    preds, confs = f.predict_batch(voltage_scores[:5], dl_scores[:5], v_conf[:5], dl_conf[:5])
    assert preds.shape == (5,)
    assert confs.shape == (5,)


def test_voting_and_weighted_average_and_adaptive():
    # Voting: when both agree
    fv = FusionLayer(method='voting')
    pred, conf = fv.predict(0.9, 0.8, 0.95, 0.92)
    assert pred == 1
    assert conf > 0.5

    # Voting: disagree pick more confident
    pred2, conf2 = fv.predict(0.9, 0.1, 0.6, 0.9)
    assert pred2 in (0, 1)

    # Weighted average adapt weights
    fwa = FusionLayer(method='weighted_average')
    fwa.update_weights_adaptive(0.8, 0.2)
    assert pytest.approx(fwa.weights['voltage'] + fwa.weights['dl'], rel=1e-6) == 1.0

    # Adaptive fusion: ensure weights update path runs
    adap = AdaptiveFusion(window_size=5, update_frequency=1)
    # warm weights
    for i in range(12):
        vs = 0.9 if i % 2 == 0 else 0.1
        ds = 0.1 if i % 2 == 0 else 0.9
        # true label matches alternating pattern
        true = 1 if i % 2 == 0 else 0
        _p, _c = adap.predict_adaptive(vs, ds, 0.9, 0.9, true_label=true)
    # After updates, weights should be set (sum to 1)
    assert pytest.approx(adap.weights['voltage'] + adap.weights['dl'], rel=1e-6) == 1.0
