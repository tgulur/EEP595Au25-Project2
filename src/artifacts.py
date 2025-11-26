"""Artifacts helper utilities.

Centralizes common paths and model/artifact saving helpers so trainers and
other code produce consistent filenames and locations under the experiment
`results_dir`.

API (small):
- `artifact_path(results_dir, *parts) -> Path` : join path parts under results_dir
- `ensure_dir(results_dir, subdir) -> Path` : ensure a subdirectory exists
- `save_model(model, results_dir, model_name, subdir='models', filename=None) -> Path` : save model using a best-effort strategy
- `write_json(obj, results_dir, filename)` : write JSON metadata

This module makes conservative choices so it works with Keras-like model
objects (have `.save_model` or `.save`) and sklearn objects (saved via joblib).
"""
from pathlib import Path
from typing import Any, Optional
import json
import logging
import joblib

logger = logging.getLogger(__name__)


def artifact_path(results_dir: str, *parts: str) -> Path:
    """Return a Path under `results_dir` joined with provided parts."""
    base = Path(results_dir)
    return base.joinpath(*parts)


def ensure_dir(results_dir: str, subdir: Optional[str] = None) -> Path:
    """Ensure the directory exists and return the Path."""
    path = Path(results_dir) / (subdir or "")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model: Any, results_dir: str, model_name: str, subdir: str = "models", filename: Optional[str] = None) -> Path:
    """Save a model artifact to disk using a best-effort method.

    - If the model has `save_model(path)` it will be used (project-specific API).
    - Else if the model has `save(path)` it will be used (Keras/TensorFlow API).
    - Otherwise, fallback to `joblib.dump` to serialize the object.

    Returns the path written.
    """
    out_dir = ensure_dir(results_dir, subdir)

    if filename is None:
        # default filename
        filename = f"{model_name}"

    # Choose extension
    # If model has explicit save_model/save and user didn't provide extension,
    # prefer .h5 for Keras-style, else .pkl for joblib.
    ext = Path(filename).suffix
    if not ext:
        # heuristics
        if hasattr(model, "save_model") or hasattr(model, "save"):
            ext = ".h5"
        else:
            ext = ".pkl"

    out_path = out_dir / (Path(filename).stem + ext)

    try:
        if hasattr(model, "save_model"):
            # project uses `save_model(path)` API in some model wrappers
            model.save_model(str(out_path))
            logger.info(f"Saved model {model_name} via save_model() to {out_path}")
        elif hasattr(model, "save"):
            # Keras/TensorFlow style
            model.save(str(out_path))
            logger.info(f"Saved model {model_name} via save() to {out_path}")
        else:
            # fallback to joblib
            joblib.dump(model, str(out_path))
            logger.info(f"Serialized model {model_name} via joblib to {out_path}")
    except Exception as e:
        logger.warning(f"Failed to save model {model_name} to {out_path}: {e}")
        # As a last resort, try joblib with a .pkl
        try:
            out_path_fallback = out_dir / (Path(filename).stem + ".pkl")
            joblib.dump(model, str(out_path_fallback))
            logger.info(f"Serialized model {model_name} via joblib fallback to {out_path_fallback}")
            return out_path_fallback
        except Exception:
            logger.exception("Failed to serialize model with joblib fallback")
            raise

    return out_path


def write_json(obj: Any, results_dir: str, filename: str = "metadata.json") -> Path:
    """Write JSON `obj` to `results_dir/filename` and return the Path."""
    p = artifact_path(results_dir, filename)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(obj, f, indent=2)
        logger.info(f"Wrote JSON metadata to {p}")
    except Exception:
        logger.exception(f"Failed to write JSON to {p}")
        raise
    return p
