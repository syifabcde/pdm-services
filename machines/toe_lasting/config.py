"""
Toe-lasting paths and constants, read from machine.toml. Replaces the old app/config.py:
the credentials moved to core/settings.py (shared), and DEVICE_ID is gone -- the device is
an argument everywhere now.
"""
from pathlib import Path

from core import settings
from core.registry import read_devices, read_spec

PACKAGE_DIR = Path(__file__).resolve().parent
_SPEC = read_spec(PACKAGE_DIR)

LABEL = _SPEC["label"]
DB_TABLE = _SPEC["db_table"]
DB_COLUMNS = tuple(_SPEC["db_columns"])
DEVICES = tuple(_SPEC["devices"])


def device(iddev):
    """That device's settings from toe-lasting/config/devices.toml (cadence_s, session_break_min,
    trigger_readings, roll_windows_min, ...) -- the same file the training pipeline reads."""
    return read_devices(_SPEC["data_dir"])[f"iddev{iddev}"]

DATA_DIR = settings.REPO_ROOT / _SPEC["data_dir"]
MODEL_DIR = DATA_DIR / "models" / "forecasting"
RUL_MODEL_DIR = DATA_DIR / "models" / "rul"

# runtime audit logs: one folder per machine; every row carries its iddev
STATE_DIR = settings.STATE_DIR / _SPEC["data_dir"]
PRED_DB_PATH = STATE_DIR / "predictions.sqlite"
RUL_DB_PATH = STATE_DIR / "rul_predictions.sqlite"
DRIFT_DB_PATH = STATE_DIR / "band_drift.sqlite"
RUL_TRIGGER_DB_PATH = STATE_DIR / "rul_trigger_state.sqlite"
