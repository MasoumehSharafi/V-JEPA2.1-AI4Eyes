"""YAML config loading: configs/supervised.yaml → the resolved config of one sign (defaults ⊕ sign overrides)."""
import copy

import yaml


def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def sign_config(cfg, sign):
    """Resolved config of `sign` from a supervised.yaml dict."""
    assert sign in cfg["signs"], f"{sign} not in config"
    return merge(cfg["defaults"], cfg["signs"][sign])
