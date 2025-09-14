# File: src/utils.py
"""
Utility helpers used across scripts.
"""
import json
from pathlib import Path

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)
