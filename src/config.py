import os
from pathlib import Path
import tomllib

CONFIG_PATH = Path.home() / "iwa-collections-analysis" / "config.toml"

_config_cache = None

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"compute": {"use_gpu": False}, "database": {"use_db": False}}
    try:
        with open(CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"Error loading config.toml: {e}")
        return {"compute": {"use_gpu": False}, "database": {"use_db": False}}

def get_config() -> dict:
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache

def is_gpu_enabled() -> bool:
    conf = get_config()
    return conf.get("compute", {}).get("use_gpu", False)

def is_db_enabled() -> bool:
    conf = get_config()
    return conf.get("database", {}).get("use_db", False)
