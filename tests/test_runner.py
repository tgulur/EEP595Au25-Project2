import sys
from types import SimpleNamespace

import pytest

from src import runner


class DummySuccess:
    def __init__(self, config_path=None, seed=None):
        self.config_path = config_path
        self.seed = seed

    def run_experiment(self):
        # simulate some work but don't crash
        self.result = "ok"


class DummyFail:
    def __init__(self, config_path=None, seed=None):
        pass

    def run_experiment(self):
        raise RuntimeError("expected failure")


def test_runner_run_success(monkeypatch):
    # Simulate CLI args
    monkeypatch.setattr(sys, "argv", ["prog", "--config", "config/config.yaml", "--seed", "7"])
    rc = runner.run(DummySuccess)
    assert rc == 0


def test_runner_run_failure(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--config", "config/config.yaml"])
    rc = runner.run(DummyFail)
    assert rc == 2
