# ♟ Chess Companion HUD

A sleek, modern desktop companion HUD that watches an online chess board (e.g. [Chess.com](https://www.chess.com) or [Lichess](https://lichess.org)), detects the board state in real-time using computer vision, and calculates the best move for your side using **Stockfish 19**.

---

## ✨ Features

- **Modern Redesigned HUD**: Deep obsidian dark theme (`#0f1117`), tournament-style mini-board, and glassmorphic card aesthetics.
- **🔍 1-Click Auto-Detect**: Instantly finds the Chess.com board on your screen with one click.
- **🎯 Desktop Snip Tool**: Freezes your screen in-place (no macOS space switching or black screens) so you can drag-select over your actual chessboard with an 8x8 alignment grid.
- **100% Accurate Computer Vision**: Uses corner background sampling and synthetic composite template matching (supports Chess.com Neo & Classic piece sets).
- **Stockfish 19 Engine**: Bundled Apple Silicon / macOS universal binary with NNUE support for master-level depth evaluation.
- **Perspective & Side Auto-Detection**:
  - Automatically identifies whether you are playing White or Black from board orientation.
  - Quick-switch segmented pill buttons (`Auto`, `White`, `Black`).
- **Live Mini-Board Preview with Move Arrow**:
  - Tournament wood styling (`#f0d9b5` / `#b58863`) with origin & destination highlights and a glowing directional move arrow.
- **Refined Advantage Meter**:
  - Smooth dual-tone balance capsule with White vs Black win probability percentages.
- **Alternative Lines (MultiPV)**:
  - Displays top 3 candidate moves and principal variations.
- **Auto-Scan Mode**:
  - Watches the board automatically every 0.8 seconds with 0% CPU impact when position hasn't changed.
- **Audio Announcements**:
  - Optional voice whispers the best move via macOS native speech synthesis (e.g., *"Knight to f3"*).

---

## 🚀 Quick Start

Open your terminal and run:

```bash
cd ~/Desktop/chess-companion
./run.sh
```

---

## 🎯 How to Calibrate & Use

1. **Open your Chess Game**:
   - Open Chess.com or Lichess in your web browser.
2. **Lock onto the Board**:
   - **Method A (Easiest)**: Click **`🔍 Auto-Detect`**. The app scans your display and locks onto the Chess.com board in one second!
   - **Method B (Manual Snip)**: Click **`🎯 Snip Board`**. A frozen capture of your screen will appear right in front of you. Click and drag a box across the 8x8 chessboard from top-left to bottom-right (green alignment grid lines guide you).
3. **Analyze**:
   - Press **`⚡ Analyze`** (or press <kbd>Space</kbd> / <kbd>Enter</kbd>) to calculate the best move immediately.
   - Or toggle **`[x] 🔄 Auto-Scan`** to let the HUD automatically track moves in the background.
4. **Choose Side / Perspective**:
   - Click the segmented pill buttons (`Auto`, `White`, `Black`) to toggle sides on the fly.

---

## ⌨️ Shortcuts

| Key | Action |
| --- | --- |
| `Space` or `Enter` | Scan and analyze the board immediately |
| `ESC` | Cancel board snip overlay |
