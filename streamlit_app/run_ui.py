#!/usr/bin/env python3
"""
Run script for AgentMem Streamlit UI
"""
import sys
import os
from pathlib import Path

# Add project root to Python path (parent of streamlit_app)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variable for Streamlit to find the config
os.environ["STREAMLIT_CONFIG_DIR"] = str(project_root / ".streamlit")

# Ensure we're using the virtual environment
venv_python = project_root / ".venv" / "Scripts" / "python.exe"
if venv_python.exists():
    sys.executable = str(venv_python)

if __name__ == "__main__":
    import streamlit.web.cli as stcli

    # Run Streamlit app
    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).parent / "app.py"),
        "--server.port=8501",
        "--server.address=localhost",
    ]

    sys.exit(stcli.main())


