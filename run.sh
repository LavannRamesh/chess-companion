#!/usr/bin/env bash
# Chess Companion Launcher
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    ./.venv/bin/pip install --upgrade pip --quiet
fi

# Always ensure all packages are installed
echo "Checking dependencies..."
./.venv/bin/pip install pillow python-chess numpy opencv-python-headless mss customtkinter --quiet

# Ensure Stockfish binary has execute permissions
if [ -f "bin/stockfish/stockfish-macos-universal" ]; then
    chmod +x bin/stockfish/stockfish-macos-universal
    xattr -d com.apple.quarantine bin/stockfish/stockfish-macos-universal 2>/dev/null || true
fi

echo "Starting Chess Companion HUD..."
./.venv/bin/python3 main.py