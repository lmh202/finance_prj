"""DO NOT EDIT — routing shim. The real page is src/daily_strategy/page.py (Developer 2)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.daily_strategy.page import render

render()
