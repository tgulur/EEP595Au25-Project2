import os
import tempfile
import numpy as np

from src.deep_learning_models import CNNModel, LSTMModel, HybridCNNLSTM


def _small_input(shape=(20, 4), n_samples=3):
    return np.random.randn(n_samples, *shape)


def test_cnn_build_predict_save_load():
    X = _small_input((20, 4), n_samples=4)

    cnn = CNNModel(filters=[8, 16], kernel_size=3, dropout=0.1)
    cnn.build_model(input_shape=(20, 4))

    # predict (will warm up if needed)
    preds, probs = cnn.predict(X)
    assert preds.shape[0] == X.shape[0]
    assert probs.shape[0] == X.shape[0]
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # latency measurement (small iterations for speed)
    lat = cnn.predict_with_latency(X[0], n_iterations=3)
    assert 'mean_latency_ms' in lat and lat['mean_latency_ms'] > 0

    # save and load
    with tempfile.TemporaryDirectory() as td:
        # Keras save requires a file extension like .keras or .h5
        path = os.path.join(td, 'cnn_saved.keras')
        cnn.save_model(path)

        # new instance loads the saved model
        cnn2 = CNNModel(filters=[8, 16])
        cnn2.load_model(path)
        preds2, probs2 = cnn2.predict(X)
        assert preds2.shape == preds.shape


def test_lstm_and_hybrid_predict_and_warmup():
    X = _small_input((15, 3), n_samples=2)

    lstm = LSTMModel(hidden_units=[16], dropout=0.1, bidirectional=False)
    lstm.build_model(input_shape=(15, 3))
    # explicit warmup
    lstm.warmup(input_shape=(15, 3), n_warmup=2)
    preds, probs = lstm.predict(X)
    assert preds.shape[0] == X.shape[0]

    hybrid = HybridCNNLSTM(cnn_filters=[8], lstm_units=[8], dropout=0.1)
    hybrid.build_model(input_shape=(15, 3))
    preds_h, probs_h = hybrid.predict(X)
    assert preds_h.shape[0] == X.shape[0]
