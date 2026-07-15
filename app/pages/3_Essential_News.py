"""DO NOT EDIT — routing shim. The real page is src/news_intelligence/page.py (Developer 3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.news_intelligence.page import render

render()
