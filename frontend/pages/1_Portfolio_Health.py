"""DO NOT EDIT — routing shim. The real page is frontend/views/portfolio_health.py (Developer 1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from views.portfolio_health import render

render()
