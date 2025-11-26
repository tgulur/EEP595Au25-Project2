import json
import joblib
from pathlib import Path

import pytest

from src.artifacts import save_model, write_json, artifact_path, ensure_dir


class DummySaveModel:
    def __init__(self):
        self.saved = None

    def save_model(self, path):
        self.saved = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("saved_model")


class DummySave:
    def __init__(self):
        self.saved = None

    def save(self, path):
        self.saved = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("saved_save")


class DummyPlain:
    def __init__(self, value):
        self.value = value


def test_artifact_path_and_ensure_dir(tmp_path):
    base = str(tmp_path)
    sub = artifact_path(base, "models", "subdir")
    assert str(sub).startswith(str(tmp_path))

    d = ensure_dir(base, "models/testsub")
    assert d.exists()


def test_save_model_prefers_save_model(tmp_path):
    base = str(tmp_path)
    model = DummySaveModel()
    p = save_model(model, base, "mymodel", subdir="models", filename="myfile")
    assert p.exists()
    # default extension for save_model/save is .h5
    assert p.suffix == ".h5"
    assert p.read_text() == "saved_model"


def test_save_model_prefers_save(tmp_path):
    base = str(tmp_path)
    model = DummySave()
    p = save_model(model, base, "smodel", subdir="models", filename="sfile")
    assert p.exists()
    assert p.suffix == ".h5"
    assert p.read_text() == "saved_save"


def test_save_model_joblib_fallback(tmp_path):
    base = str(tmp_path)
    obj = DummyPlain({"a": 1})
    p = save_model(obj, base, "plain", subdir="models", filename="plainfile")
    # fallback uses .pkl when no save/save_model attribute
    assert p.exists()
    assert p.suffix == ".pkl"
    loaded = joblib.load(p)
    assert isinstance(loaded, DummyPlain)
    assert loaded.value == {"a": 1}


def test_write_json(tmp_path):
    base = str(tmp_path)
    data = {"x": 1, "y": [1, 2, 3]}
    p = write_json(data, base, filename="meta_test.json")
    assert p.exists()
    loaded = json.loads(p.read_text())
    assert loaded == data
