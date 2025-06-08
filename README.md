# Speed Skating Starting Line Fault Detection

This project detects false starts in speed skating using computer vision.

## Installation

1. Create a Python virtual environment (optional but recommended).
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

- `main.py` runs the detection pipeline using your webcam.
- `app_ui.py` provides a Qt based interface to configure detection parameters.

Run a script with Python:

```bash
python main.py
```

Adjust sound files and thresholds in `config.json` or via the UI if needed.
