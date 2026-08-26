"""Puts src/ on the path so tests and notebooks can import the modules directly.

The modules import their siblings by bare name because they are run as scripts
(python3 src/train.py), which already puts src/ on the path. This is the one
place that does it for every other caller.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
