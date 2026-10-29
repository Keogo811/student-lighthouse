"""Load the project configuration from config.yaml."""
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read config.yaml and return it as a dictionary."""
    config_path = path if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)