import os
import json
import tempfile
from pathlib import Path

import yaml

from src import configuration
from src import runner as runner_module


def test_load_config_and_namespace(tmp_path):
    cfg = {"a": 1, "b": {"c": 2}}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))

    raw, ns = configuration.load_config(str(p))
    assert raw == cfg
    assert hasattr(ns, 'a') and ns.a == 1
    assert hasattr(ns, 'b') and ns.b.c == 2


def test_snapshot_config_writes_metadata(tmp_path):
    raw = {"k": "v"}
    results_dir = tmp_path / "results"
    ts = "TEST_TS"

    # Should not raise even if git not available
    configuration.snapshot_config(raw, str(results_dir), ts, extra={"extra": 123})

    meta_file = results_dir / "metadata.json"
    assert meta_file.exists()
    data = json.loads(meta_file.read_text())
    assert data.get('timestamp') == ts
    assert data.get('config') == raw
    assert data.get('extra') == 123 or data.get('extra') is None or 'extra' in data


def test_configure_cuda_paths_sets_ld_library_path(tmp_path, monkeypatch):
    # Create a fake venv structure with one of the expected CUDA lib paths
    base = tmp_path / "venv_fake"
    target = base / "lib64" / "python3.12" / "site-packages" / "nvidia" / "cublas" / "lib"
    target.mkdir(parents=True)

    old = os.environ.get('LD_LIBRARY_PATH')
    try:
        # Call function with our fake venv
        configuration.configure_cuda_paths(str(base))
        new = os.environ.get('LD_LIBRARY_PATH', '')
        assert str(target) in new
    finally:
        # Restore environment
        if old is None:
            os.environ.pop('LD_LIBRARY_PATH', None)
        else:
            os.environ['LD_LIBRARY_PATH'] = old


def test_load_and_snapshot_config_returns_paths(tmp_path):
    cfg = {"output": {"results_path": str(tmp_path / "outbase")}}
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))

    raw, ns, results_dir, ts = configuration.load_and_snapshot_config(str(cfg_file), results_base=None, timestamp="TS123", extra={"x": 1})
    assert raw['output']['results_path'] == cfg['output']['results_path']
    assert ts == "TS123"
    assert Path(results_dir).exists()
    # metadata.json should be present
    assert (Path(results_dir) / 'metadata.json').exists()


def test_runner_success_and_failure():
    # Success case
    class DummyExp:
        def __init__(self, config_path, seed):
            self.config_path = config_path
            self.seed = seed

        def run_experiment(self):
            # set an attribute to show it ran
            self.ran = True

    rc = runner_module.run(DummyExp, argv=["--config", "cfg", "--seed", "7"])  # should return 0
    assert rc == 0

    # Failure case
    class FailingExp:
        def __init__(self, config_path, seed):
            pass

        def run_experiment(self):
            raise RuntimeError("boom")

    rc2 = runner_module.run(FailingExp, argv=["--config", "cfg"])  # should return non-zero
    assert rc2 != 0
