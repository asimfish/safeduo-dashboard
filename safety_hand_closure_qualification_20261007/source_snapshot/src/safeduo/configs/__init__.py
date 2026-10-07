"""配置加载：YAML 是唯一阈值来源，代码里不许硬编码阈值。"""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(name: str = "duo_env.yaml") -> dict:
    """Load a config yaml; a top-level ``extends: <parent.yaml>`` key deep-
    merges the child over the parent (R17 S9: duo_env_v7_objects.yaml =
    duo_env_v7.yaml + assets.table_objects, no copy drift). Yamls without
    the key load exactly as before."""
    with open(CONFIG_DIR / name) as f:
        y = yaml.safe_load(f)
    parent = y.pop("extends", None)
    if parent:
        y = _deep_merge(load_config(str(parent)), y)
    return y


def repo_root() -> Path:
    # src/safeduo/configs -> src/safeduo -> src -> repo
    return CONFIG_DIR.parents[2]
