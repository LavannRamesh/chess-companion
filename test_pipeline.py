import sys
import os
import chess
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine import ChessEngine
from src.vision import BoardVision
from src.board_state import BoardStateManager


def test_full_pipeline():
    print("1. Initializing Stockfish Engine...")
    engine = ChessEngine()

    print("2. Initializing Board Vision (alpha theme)...")
    vision = BoardVision(theme="alpha")

    print("3. Generating test board image with Alpha pieces on 365chess-style board...")
    sq_size = 80
    board_pil = Image.new("RGB", (sq_size * 8, sq_size * 8))

    # 365chess-style blue/white board
    light_sq = (240, 217, 181)   # cream/ivory
    dark_sq  = (181, 136, 99)    # brown (approximating the 365chess blue with something distinct)

    for r in range(8):
        for c in range(8):
            color = light_sq if (r + c) % 2 == 0 else dark_sq
            sq_img = Image.new("RGB", (sq_size, sq_size), color)
            board_pil.paste(sq_img, (c * sq_size, r * sq_size))

    # Paste pieces — position: simplified starting formation
    pieces_to_paste = {
        (0, 4): "bk", (0, 3): "bq", (1, 4): "bp",
        (7, 4): "wk", (7, 3): "wq", (6, 4): "wp"
    }
    for (r, c), name in pieces_to_paste.items():
        p_img = vision.templates[name].resize((sq_size, sq_size), Image.Resampling.LANCZOS)
        board_pil.paste(p_img, (c * sq_size, r * sq_size), p_img)

    board_np = np.array(board_pil)[:, :, ::-1]  # RGB to BGR

    print("4. Testing Vision Recognition...")
    grid = vision.recognize_board(board_np)

    # Print grid for debugging
    piece_chars = {
        "wp": "wP", "wn": "wN", "wb": "wB", "wr": "wR", "wq": "wQ", "wk": "wK",
        "bp": "bP", "bn": "bN", "bb": "bB", "br": "bR", "bq": "bQ", "bk": "bK",
        None: " ."
    }
    print("   Recognized board:")
    for r in range(8):
        row_str = "  " + " ".join(piece_chars.get(grid[r][c], "??") for c in range(8))
        print(row_str)

    ok = True
    for (r, c), name in pieces_to_paste.items():
        got = grid[r][c]
        status = "✓" if got == name else "✗"
        print(f"   {status} ({r},{c}): expected={name}, got={got}")
        if got != name:
            ok = False

    assert ok, "Vision recognition failed for one or more squares!"
    print("   Vision recognized all pieces accurately!")

    print("5. Testing Orientation Detection...")
    orient = BoardStateManager.detect_orientation(grid)
    assert orient == "white", f"Expected 'white', got '{orient}'"
    print(f"   Orientation detected correctly as: {orient}")

    print("6. Testing FEN generation & validation...")
    fen, board = BoardStateManager.grid_to_fen(grid, orientation=orient, turn="w")
    print(f"   Generated FEN: {fen}")
    assert board is not None, "Board validation failed"

    print("7. Testing Stockfish evaluation...")
    analysis = engine.analyze(board, depth=12)
    print(f"   Best Move: {analysis['best_san']} ({analysis['best_uci']})")
    print(f"   Score: {analysis['score_str']}")
    print(f"   Win Chance: {analysis['win_chance']}%")
    print(f"   Candidate lines: {len(analysis['candidates'])}")
    assert analysis["best_move"] is not None

    engine.close()
    print("\n✅ Full pipeline test passed successfully!")


if __name__ == "__main__":
    test_full_pipeline()
