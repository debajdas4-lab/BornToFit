"""
BornToFit Healthcare Agent: root-level entry point.
Run with:
    streamlit run app.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

runpy.run_path(str(PROJECT_ROOT / "app" / "streamlit_app.py"), run_name="__main__")
