"""
Shared settings for every machine: where things live and the Postgres credentials.
Nothing here is specific to a machine or a device.

The .env sits next to dashboard.py (pdm-services/.env) and holds credentials only --
the table name, columns and device list belong to each machine (machines/<name>/machine.toml),
and which device is being looked at is chosen in the UI, never by an environment variable.

Credentials are read when an engine is built, not at import, so modules that only need
paths (e.g. rul_bands.py) stay importable without a .env.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

SERVICES_DIR = Path(__file__).resolve().parent.parent   # pdm-services/
REPO_ROOT = SERVICES_DIR.parent                          # pdm/
MACHINES_DIR = SERVICES_DIR / "machines"
STATE_DIR = SERVICES_DIR / "state"                       # sqlite audit logs, one sub-folder per machine

load_dotenv(SERVICES_DIR / ".env")


def db_url():
    env = os.environ
    return (
        f"postgresql+psycopg2://{env['DB_USER']}:{env['DB_PASSWORD']}"
        f"@{env['DB_HOST']}:{env.get('DB_PORT', '5432')}/{env['DB_NAME']}"
    )
