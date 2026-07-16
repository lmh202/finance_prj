"""DO NOT EDIT — routing shim. The real page is src/portfolio_health/page.py (Developer 1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.portfolio_health.page import render_performance

render_performance()
