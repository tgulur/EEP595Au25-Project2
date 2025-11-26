import numpy as np

from src.baseline_models import TimingBasedIDS, RuleBasedIDS, FrequencyBasedIDS


def test_timing_based_ids_train_predict():
    np.random.seed(0)
    n = 200
    # Simulate timestamps with fairly regular intervals for two CAN IDs
    base = np.cumsum(np.random.uniform(0.01, 0.02, n))
    can_ids = np.array([0x100 if i % 2 == 0 else 0x200 for i in range(n)])
    labels = np.zeros(n)
    # Introduce some attacks in second half
    labels[n//2::20] = 1

    timing = TimingBasedIDS(threshold=0.5)
    # train on first half
    timing.train(base[:100], can_ids[:100], labels[:100])
    preds, scores = timing.predict(base[100:], can_ids[100:])
    assert preds.shape[0] == base[100:].shape[0]
    assert scores.shape[0] == base[100:].shape[0]


def test_rule_based_ids_train_predict():
    np.random.seed(0)
    n = 50
    can_ids = np.array([0x100]*n)
    dlc = np.array([8]*n)
    data = [list(np.random.randint(0, 256, 8)) for _ in range(n)]
    labels = np.zeros(n)

    rule = RuleBasedIDS()
    rule.train(can_ids, dlc, labels)

    # Create a few suspicious entries: unknown id, wrong dlc, all zeros
    test_ids = np.array([0x100, 0x999, 0x100])
    test_dlc = np.array([8, 8, 4])
    test_data = [data[0], [0]*8, [0, 1, 2, 3]]

    preds, scores = rule.predict(test_ids, test_dlc, test_data)
    assert preds.shape[0] == 3
    assert scores.shape[0] == 3
    # Expect at least one anomaly flagged for unknown id or bad dlc
    assert preds.sum() >= 1


def test_frequency_based_ids_train_predict_small():
    np.random.seed(0)
    n = 300
    timestamps = np.cumsum(np.random.uniform(0.001, 0.01, n))
    can_ids = np.random.choice([0x100, 0x200], n)
    labels = np.zeros(n)

    freq = FrequencyBasedIDS(window_size=20, threshold=1.0)
    # train on slices (will likely learn some expected frequencies)
    freq.train(timestamps[:200], can_ids[:200], labels[:200])
    preds, scores = freq.predict(timestamps[200:], can_ids[200:])
    assert preds.shape[0] == timestamps[200:].shape[0]
    assert scores.shape[0] == timestamps[200:].shape[0]
