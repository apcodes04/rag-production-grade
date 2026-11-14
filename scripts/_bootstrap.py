"""Put src/ on the import path for the CLI scripts.

Why this exists at all: the package lives in src/ (the "src layout"), which
stops Python from importing the half finished package from the current folder
by accident. Installing the project (pip install -e .) is the grown up fix;
this keeps the repo runnable with zero install steps.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
