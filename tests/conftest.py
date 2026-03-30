from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    load_dotenv(ROOT / ".env")
except PermissionError:
    pass  # Fallback to process environment or skip markers if API key missing

# Tests mock Gemini and Postgres interactions, so dummy env values keep CI self-contained.
os.environ.setdefault("POSTGRES_URL", "postgresql://test:test@localhost/test")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
