"""Config loader – reads config.yaml and exposes a nested dot-access object."""
from __future__ import annotations
import yaml
from pathlib import Path


class _Namespace(dict):
    """Dict with attribute access."""
    def __getattr__(self, key):
        try:
            val = self[key]
            return _Namespace(val) if isinstance(val, dict) else val
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def load_config(path: str | Path | None = None) -> _Namespace:
    if path is None:
        # Walk up from this file to find config.yaml
        here = Path(__file__).resolve()
        for parent in [here] + list(here.parents):
            candidate = parent / "config.yaml"
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError("config.yaml not found in any parent directory")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _to_ns(raw)


def _to_ns(obj):
    if isinstance(obj, dict):
        return _Namespace({k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj
