"""Former entry point, kept so existing deployments keep working.

The app moved to ``apps/streamlit_app.py`` when NiMARE decoding was replaced by
Brainnetome annotation; point deployments (e.g. the Streamlit Cloud "main file")
there. This file just runs it.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("streamlit_app.py")), run_name="__main__")
