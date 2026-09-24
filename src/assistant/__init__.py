"""Local-first macOS assistant foundation."""

import time
from importlib.metadata import version

STARTED = time.monotonic()  # when Jamie began loading; /stats reports startup from here

__version__ = version("j-cli")
