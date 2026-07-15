"""DO NOT EDIT — routing shim. The real page is src/recommendation/page.py (Developer 4)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.recommendation.page import render

render()
