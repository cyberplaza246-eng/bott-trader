#!/usr/bin/env python3
"""Open http://127.0.0.1:5055 - MNQ action desk + Gemini chat."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.dashboard.mtf_actions import start_mtf_dashboard

if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", "5055"))
    print(f"MNQ desk: http://127.0.0.1:{port}")
    start_mtf_dashboard(port)
