import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.json"


def load_config(config_path=None):
    if config_path is None:
        config_path = CONFIG_PATH

    with open(config_path, "r") as f:
        config = json.load(f)

    paths = config["paths"]
    config["paths"] = {k: Path(v) for k, v in paths.items()}
    return config
