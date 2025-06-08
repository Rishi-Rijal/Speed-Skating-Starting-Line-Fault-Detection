import json
import os

DEFAULT_CONFIG = {
    "preStartMin": 100,
    "preStartMax": 300,
    "startLine": 400,
    "threshold": 0.02,
    "goSound": "Sounds/goSound.mp3",
    "readySound": "Sounds/readySound.mp3",
    "falseStartSound": "Sounds/falseStartBuzzer.mp3",
    "gunSound": "Sounds/gunSound.mp3",
    "imaginaryOffset": 50,
    "readyHoldTime": 3,
    "readyToStartDelay": 5,
    "startOkMin": 3,
    "startOkMax": 4.1,
}


def load_config(path: str = "config.json") -> dict:
    """Load configuration from a JSON file.

    Parameters
    ----------
    path: str
        Path to the configuration file.

    Returns
    -------
    dict
        Configuration dictionary with defaults filled in.
    """
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                user_cfg = json.load(f)
            if isinstance(user_cfg, dict):
                config.update({k: user_cfg.get(k, v) for k, v in config.items()})
        except Exception:
            print(f"Warning: Failed to read {path}; using defaults.")
    else:
        print(f"Config file {path} not found. Using defaults.")
    return config
