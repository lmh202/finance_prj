"""DO NOT EDIT — routing shim. The real page is frontend/views/fusion.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from views.fusion import render

render()
