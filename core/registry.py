"""
Finds the machines the dashboard can show. A machine is a folder under machines/ that has a
machine.toml; that file is the machine's specialised configuration (which DB table and columns
hold its readings, which devices are enabled, where its data/models live). The dashboard itself
is one page for all of them -- this registry is what fills its machine/device selectors.

Adding a machine: create machines/<name>/ with a machine.toml and a dashboard.py exposing
render(iddev). Nothing else in core/ or the top-level dashboard.py needs to change.
"""
import functools
import importlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from core import settings

_REQUIRED = ("label", "data_dir", "db_table", "db_columns", "devices")


@dataclass(frozen=True)
class Machine:
    key: str               # data_dir: also the machine's folder at the repo root and its state sub-folder
    label: str
    package: str           # importable path, e.g. "machines.toe_lasting"
    folder: Path
    db_table: str
    db_columns: tuple
    devices: tuple         # iddevs that are enabled (calibrated bands, models) -- others are not offered


def read_spec(folder):
    with open(Path(folder) / "machine.toml", "rb") as f:
        spec = tomllib.load(f)
    missing = [k for k in _REQUIRED if k not in spec]
    if missing:
        raise KeyError(f"{folder}/machine.toml is missing: {', '.join(missing)}")
    return spec


@functools.lru_cache(maxsize=None)
def read_devices(data_dir):
    """Per-device settings of a machine: <repo>/<data_dir>/config/devices.toml. The training
    pipeline reads the same file, so a device's polling cadence, session break, etc. are defined once."""
    with open(settings.REPO_ROOT / data_dir / "config" / "devices.toml", "rb") as f:
        return tomllib.load(f)


def refresh_seconds(machine, iddev):
    """The page refreshes once per sample: the device's own polling interval (iddev1 300 s, iddev2 120 s)."""
    return int(read_devices(machine.key)[f"iddev{iddev}"]["cadence_s"])


def load_machines():
    """{key: Machine} for every machines/*/machine.toml, sorted by label."""
    found = {}
    for folder in sorted(p for p in settings.MACHINES_DIR.iterdir() if (p / "machine.toml").is_file()):
        spec = read_spec(folder)
        found[spec["data_dir"]] = Machine(
            key=spec["data_dir"], label=spec["label"], package=f"machines.{folder.name}", folder=folder,
            db_table=spec["db_table"], db_columns=tuple(spec["db_columns"]),
            devices=tuple(spec["devices"]),
        )
    return dict(sorted(found.items(), key=lambda kv: kv[1].label))


def load_renderer(machine):
    """The machine's render(iddev) function, drawing that machine's sections for one device."""
    return importlib.import_module(f"{machine.package}.dashboard").render
