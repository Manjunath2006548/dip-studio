"""Vercel serverless entry point for DIP Studio.

Vercel imports this file as an ASGI handler.  We add the project root to
``sys.path`` so the regular app modules (app.py, dip_core.py, ...) and their
relative assets keep working exactly as they do for ``python app.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Matplotlib writes its font cache on first use; point it at the writable dir.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dip-studio-mpl")

from mangum import Mangum  # noqa: E402

from app import app  # noqa: E402

# Lifespan events are not needed here (no startup/shutdown hooks are used).
handler = Mangum(app, lifespan="off")

# Some Vercel Python runtimes look for `app` instead of `handler`.
__all__ = ["app", "handler"]
