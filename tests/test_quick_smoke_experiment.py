import numpy as np
import tempfile
from pathlib import Path

from src.trainers.deep_learning import train_deep_learning_models
from src.evaluation_metrics import IDSEvaluator
import yaml


def test_quick_dl_smoke(tmp_path):
    # Load quick config
    cfg_path = Path('config/quick_config.yaml')
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    np.random.seed(0)

    # Small synthetic dataset
    seq_len = config['deep_learning']['sequence_length']
    features = 3
    n_train = 8
    n_val = 2
    n_test = 2

    X_train = np.random.randn(n_train, seq_len, features).astype('float32')
    y_train = np.random.randint(0, 2, size=(n_train,))
    X_val = np.random.randn(n_val, seq_len, features).astype('float32')
    y_val = np.random.randint(0, 2, size=(n_val,))
    X_test = np.random.randn(n_test, seq_len, features).astype('float32')
    y_test = np.random.randint(0, 2, size=(n_test,))

    data = {
        'can': {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        },
        # optional dataframes not required for this smoke test
    }

    evaluator = IDSEvaluator(save_dir=str(tmp_path / 'results'))

    results = train_deep_learning_models(config, data, evaluator, str(tmp_path))

    # Basic sanity checks
    assert isinstance(results, dict)
    assert 'CNN' in results and 'LSTM' in results
    for name, rr in results.items():
        assert 'metrics' in rr
        assert 'predictions' in rr
        assert 'scores' in rr
