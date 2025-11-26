"""Configuration loader utilities.

Provides a small helper to load YAML configs and return both the raw dict
and a dot-accessible namespace. Keep this intentionally lightweight so it
can be used without pulling heavy dependencies like pydantic.
"""
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Union
import yaml


def _dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _dict_to_namespace(v)
        else:
            result[k] = v
    return SimpleNamespace(**result)


def load_config(path: Union[str, Path]) -> Tuple[Dict[str, Any], SimpleNamespace]:
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    ns = _dict_to_namespace(raw if raw is not None else {})
    return raw if raw is not None else {}, ns


def snapshot_config(raw_config: Dict[str, Any], results_dir: Union[str, Path], timestamp: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Write a metadata snapshot into `results_dir/metadata.json`.

    This captures the provided `raw_config`, a timestamp, optional extra
    metadata (git sha, python version, platform, seed overrides, etc.).
    """
    import json
    import subprocess
    import sys
    import platform as _platform

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": timestamp,
        "git_sha": None,
        "python_version": sys.version,
        "platform": _platform.platform(),
        "config": raw_config,
    }

    # merge extras if provided
    if extra:
        metadata.update(extra)

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        metadata["git_sha"] = git_sha
    except Exception:
        metadata["git_sha"] = None

    out_file = results_path / "metadata.json"
    try:
        with open(out_file, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        # non-fatal: callers should not fail if snapshot cannot be written
        pass


def configure_cuda_paths(venv_path: Optional[str] = None) -> None:
    """Attempt to set LD_LIBRARY_PATH to include common CUDA libs installed
    inside a project's virtual environment. This is best-effort and will not
    raise on failure.

    Args:
        venv_path: Optional path to the virtual environment root. If omitted
            the function will look for `./.venv` relative to the current
            working directory.
    """
    import os
    from pathlib import Path

    try:
        base = Path(venv_path) if venv_path else Path.cwd() / ".venv"
        nvidia_lib_paths = [
            base / "lib64/python3.12/site-packages/nvidia/cublas/lib",
            base / "lib64/python3.12/site-packages/nvidia/cudnn/lib",
            base / "lib64/python3.12/site-packages/nvidia/cuda_runtime/lib",
            base / "lib64/python3.12/site-packages/nvidia/cuda_cupti/lib",
            base / "lib64/python3.12/site-packages/nvidia/cuda_nvrtc/lib",
        ]

        existing = [str(p) for p in nvidia_lib_paths if p.exists()]
        if existing:
            current = os.environ.get("LD_LIBRARY_PATH", "")
            new_path = ":".join(existing)
            os.environ["LD_LIBRARY_PATH"] = f"{new_path}:{current}" if current else new_path
    except Exception:
        # Best-effort; do not fail on environment quirks
        pass


def load_and_snapshot_config(path: Union[str, Path], results_base: Optional[Union[str, Path]] = None, timestamp: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], SimpleNamespace, str, str]:
    """Load configuration and create a results directory with a metadata snapshot.

    Returns (raw_config, namespace, results_dir_path, timestamp_str).
    """
    from datetime import datetime
    raw, ns = load_config(path)

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve results directory base from provided arg or config
    if results_base:
        base = Path(results_base)
    else:
        # Fallback to config's output.results_path, or './results' if missing
        base = Path(raw.get('output', {}).get('results_path', 'results'))

    results_dir = base / ts
    results_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot config (non-fatal)
    try:
        snapshot_config(raw, str(results_dir), ts, extra=extra)
    except Exception:
        pass

    return raw, ns, str(results_dir), ts
