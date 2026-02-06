"""Resolve default paths for configs and data shipped with the package."""

import importlib.resources
from pathlib import Path


def get_default_config(name: str) -> Path:
    """Return the path to a bundled config file.

    Args:
        name: Config filename or relative path, e.g. ``"data.yaml"`` or
              ``"models/lstm.yaml"``. Passing ``"models"`` returns the
              models config directory itself.
    """
    ref = importlib.resources.files("pitch_sequencing") / "configs" / name
    # importlib.resources may return a Traversable; convert to a real Path
    # so callers can pass it to open() / os.path.exists() / etc.
    with importlib.resources.as_file(ref) as p:
        return Path(p)


def get_default_data_dir() -> Path:
    """Return the default data directory.

    Preference order:
    1. ``./data`` if it exists (repo checkout / development)
    2. ``~/pitch-sequencing-data/`` (pip-installed usage)
    """
    local = Path("data")
    if local.is_dir():
        return local.resolve()
    fallback = Path.home() / "pitch-sequencing-data"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback
