"""Repository entry point for Streamlit Cloud and local checkout use."""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_module("conninfpy.app.streamlit_app", run_name="__main__")
