import json

import yaml


def load_features(path: str) -> list[str]:
    with open(path) as f:
        data = json.load(f)

    return data["features"]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
