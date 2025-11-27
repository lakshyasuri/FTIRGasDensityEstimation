import json
from types import SimpleNamespace
from pathlib import Path

directory = Path(__file__).resolve().parent
with open(directory / "config.json", "r") as f:
    CONFIG = json.load(f, object_hook=lambda d: SimpleNamespace(**d))