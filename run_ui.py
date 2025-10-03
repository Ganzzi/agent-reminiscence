#!/usr/bin/env python3
"""
Run script for AgentMem Streamlit UI
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variable for Streamlit to find the config
os.environ['STREAMLIT_CONFIG_DIR'] = str(project_root / '.streamlit')

if __name__ == "__main__":
    import streamlit.web.cli as stcli
    
    # Run Streamlit app
    sys.argv = [
        "streamlit",
        "run",
        str(project_root / "streamlit_app" / "app.py"),
        "--server.port=8501",
        "--server.address=localhost"
    ]
    
    sys.exit(stcli.main())
