#!/usr/bin/env python3
"""
Chess Companion - Floating HUD for live online chess board analysis.
Usage:
    python3 main.py
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.hud import launch

if __name__ == "__main__":
    launch()
