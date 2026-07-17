"""Shared configuration for the AURORA backend.

DATA_DIR is the single place the data root is resolved. It defaults to the
repo-level ``data/`` directory (a sibling of ``backend/``) and can be
overridden with the AURORA_DATA_DIR environment variable.
"""

import os
from pathlib import Path

# config.py lives at backend/src/config.py -> parents[2] is the repo root.
DATA_DIR = Path(
    os.environ.get("AURORA_DATA_DIR", Path(__file__).resolve().parents[2] / "data")
)
