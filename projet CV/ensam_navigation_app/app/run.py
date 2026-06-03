#!/usr/bin/env python
"""
Convenience launcher: runs the Streamlit app from the project root.
Usage: python run.py
"""
import subprocess
import sys
import os

app_path = os.path.join(os.path.dirname(__file__), "app", "main.py")
subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], cwd=os.path.dirname(__file__))
