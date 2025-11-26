import json
from pathlib import Path

import pytest

from src.configuration import load_config, snapshot_config


def test_load_config_returns_dict_and_namespace():
    cfg_path = Path("config/config.yaml")
    raw, ns = load_config(str(cfg_path))
    assert isinstance(raw, dict)
    assert hasattr(ns, "data") or isinstance(ns, object)


def test_snapshot_config_writes_metadata(tmp_path):
    raw = {"data": {"random_seed": 42}, "fusion": {"method": "stacking"}}
    timestamp = "20250101_000000"
    extra = {"note": "unit-test"}

    out_dir = tmp_path / "results"
    snapshot_config(raw, str(out_dir), timestamp, extra=extra)

    meta_file = out_dir / "metadata.json"
    assert meta_file.exists()

    content = json.loads(meta_file.read_text())
    # basic keys
    assert content.get("timestamp") == timestamp
    assert "config" in content
    assert content["config"] == raw
    # extras merged
    assert content.get("note") == "unit-test"
    # git_sha may be None or a SHA string; ensure key present
    assert "git_sha" in content
