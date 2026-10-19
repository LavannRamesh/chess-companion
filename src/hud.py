"""
Chess Companion HUD
-------------------
A live-tracking companion HUD for online chess (Chess.com, Lichess, etc.).
Features:
  - 1-Snip Live Recording: Snip your board once, and the HUD automatically
    records live and updates after every move without needing manual resets.
  - Plain-English 1-Line Best Move display (e.g. "♞ Knight: G1 ➔ F3 (Nf3)  |  +0.45")
  - Automatic piece theme detection (Neo, Alpha, Classic, Wikipedia)
  - Perspective selection (Auto / White / Black) with Stockfish evaluation
  - Live mini-board preview with glowing directional arrow & square highlights
  - Win-rate advantage meter + top candidate lines
  - Audio move announcer via macOS native speech
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import tkinter as tk
from typing import Optional

import chess
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from src.audio import MoveAnnouncer
from src.board_state import BoardStateManager
from src.calibrator import BoardCalibrator
from src.engine import ChessEngine
from src.vision import BoardVision

# ── Appearance Configuration ───────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BOARD_PX = 360
SQ_PX = BOARD_PX // 8  # 45 px per square

LIGHT_SQ = "#f0d9b5"
DARK_SQ = "#b58863"

CONFIG_PATH = "config.json"
DEFAULT_CONFIG = {
    "board_bbox": None,
    "theme": "auto",
    "side": "auto",
    "depth": 14,
    "audio": False,
}

PIECE_ICONS = {
    "P": "♟", "N": "♞", "B": "♝", "R": "♜", "Q": "♛", "K": "♚",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}


def _load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
            return {**DEFAULT_CONFIG, **data}
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def _save_config(cfg: dict) -> None:
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (alpha,)


def _sq_to_canvas(sq: chess.Square, flipped: bool) -> tuple[int, int]:
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    if flipped:
        return 7 - f, r
    return f, 7 - r


def _draw_arrow(draw: ImageDraw.ImageDraw,
                x1: float, y1: float,
                x2: float, y2: float,
                color, width: int = 7, head_size: int = 18) -> None:
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    lx2, ly2 = x2 - ux * head_size, y2 - uy * head_size
    draw.line([(x1, y1), (lx2, ly2)], fill=color, width=width)
    px, py = -uy, ux
    draw.polygon([
        (x2, y2),
        (lx2 + px * head_size * 0.5, ly2 + py * head_size * 0.5),
        (lx2 - px * head_size * 0.5, ly2 - py * head_size * 0.5),
    ], fill=color)


class ChessHUD(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ── Window Setup ──────────────────────────────────────────────────
        self.title("♟ Chess Companion Live HUD")
        self.geometry("520x860")
        self.minsize(480, 800)
        self.configure(fg_color="#0f1117")

        # ── State ─────────────────────────────────────────────────────────
        self.cfg = _load_config()
        self.player_side: str = self.cfg.get("side", "auto")
        self.current_board: Optional[chess.Board] = None
        self.last_fen: Optional[str] = None
        self.last_screen_crop: Optional[np.ndarray] = None
        self.flipped: bool = False

        self.is_live: bool = False
        self.analysis_running: bool = False
        self._board_photo: Optional[ImageTk.PhotoImage] = None
        self._piece_images: dict[str, Image.Image] = {}

        # ── Backend Modules ───────────────────────────────────────────────
        self.vision = BoardVision(theme="neo")
        self.bsm = BoardStateManager()
        self.announcer = MoveAnnouncer(enabled=self.cfg.get("audio", False))
        
        try:
            self.engine = ChessEngine()
            self._engine_error: Optional[str] = None
        except Exception as exc:
            self.engine = None
            self._engine_error = str(exc)

        # ── Build GUI ─────────────────────────────────────────────────────
        self._load_piece_graphics()
        self._setup_ui()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Draw empty board initially
        self._redraw_board(None)

        # If a bbox was previously saved, start live recording immediately!
        if self.cfg.get("board_bbox"):
            self.after(600, self._start_live_tracking)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    #  Piece Graphics
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

    def _load_piece_graphics(self):
        """Loads and pre-scales piece PNGs for the mini-board display."""
        self._piece_images.clear()
        theme = self.vision.theme
        theme_dirs = [
            os.path.join("assets", "pieces", theme),
            os.path.join("assets", "pieces", "alpha"),
            os.path.join("assets", "pieces", "neo"),
            os.path.join("assets", "pieces", "wikipedia")
        ]
        for name in ["wp", "wn", "wb", "wr", "wq", "wk", "bp", "bn", "bb", "br", "bq", "bk"]:
            for d in theme_dirs:
                p = os.path.join(d, f"{name}.png")
                if os.path.exists(p):
                    try:
                        im = Image.open(p).convert("RGBA").resize((SQ_PX, SQ_PX), Image.Resampling.LANCZOS)
                        self._piece_images[name] = im
                        break
                    except Exception:
                        pass

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    #  UI Construction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

    def _setup_ui(self):
        # 1. Header Frame
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=20, pady=(14, 6))

        title_lbl = ctk.CTkLabel(
            hdr, text="♟ CHESS COMPANION",
            font=ctk.CTkFont("Helvetica", 20, "bold"),
            text_color="#10b981"
        )
        title_lbl.pack(side="left")

        self.audio_btn = ctk.CTkButton(
            hdr, text="🔊 Audio" if self.announcer.enabled else "🔇 Muted",
            width=75, height=28, corner_radius=6,
            fg_color="#1e293b", hover_color="#334155",
            font=ctk.CTkFont(size=12),
            command=self._toggle_audio
        )
        self.audio_btn.pack(side="right", padx=(8, 0))

        self.live_badge = ctk.CTkLabel(
            hdr, text="● READY", font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#94a3b8"
        )
        self.live_badge.pack(side="right", padx=6)

        # 2. Hero 1-Line Best Move Card
        self.hero_card = ctk.CTkFrame(self, fg_color="#1a1d24", corner_radius=12)
        self.hero_card.pack(fill="x", padx=20, pady=6)

        top_meta = ctk.CTkFrame(self.hero_card, fg_color="transparent")
        top_meta.pack(fill="x", padx=16, pady=(10, 2))

        ctk.CTkLabel(
            top_meta, text="BEST MOVE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#64748b"
        ).pack(side="left")

        self.eval_badge = ctk.CTkLabel(
            top_meta, text="+0.00",
            font=ctk.CTkFont("Courier", 13, "bold"),
            fg_color="#0f172a", corner_radius=6,
            text_color="#38bdf8", width=75, height=24
        )
        self.eval_badge.pack(side="right")

        # Plain-English 1-Line Move Display
        self.move_lbl = ctk.CTkLabel(
            self.hero_card, text="Snip board once to start live recording",
            font=ctk.CTkFont("Helvetica", 19, "bold"),
            text_color="#94a3b8"
        )
        self.move_lbl.pack(anchor="w", padx=16, pady=(2, 6))

        # Win probability advantage meter
        win_f = ctk.CTkFrame(self.hero_card, fg_color="transparent")
        win_f.pack(fill="x", padx=16, pady=(0, 10))

        self.w_pct_lbl = ctk.CTkLabel(win_f, text="W 50%", font=ctk.CTkFont(size=11, weight="bold"), text_color="#cbd5e1")
        self.w_pct_lbl.pack(side="left")

        self.win_bar = ctk.CTkProgressBar(win_f, height=8, corner_radius=4, fg_color="#1e293b", progress_color="#10b981")
        self.win_bar.pack(side="left", fill="x", expand=True, padx=8)
        self.win_bar.set(0.5)

        self.b_pct_lbl = ctk.CTkLabel(win_f, text="50% B", font=ctk.CTkFont(size=11, weight="bold"), text_color="#cbd5e1")
        self.b_pct_lbl.pack(side="right")

        # 3. Mini-Board Canvas
        board_container = ctk.CTkFrame(self, fg_color="#1a1d24", corner_radius=10)
        board_container.pack(padx=20, pady=6)

        self.canvas = tk.Canvas(
            board_container, width=BOARD_PX, height=BOARD_PX,
            bg="#1e293b", highlightthickness=0
        )
        self.canvas.pack(padx=6, pady=6)

        # 4. Top 3 Candidates
        cand_card = ctk.CTkFrame(self, fg_color="#1a1d24", corner_radius=8)
        cand_card.pack(fill="x", padx=20, pady=4)

        cand_row = ctk.CTkFrame(cand_card, fg_color="transparent")
        cand_row.pack(fill="x", padx=12, pady=6)

        ctk.CTkLabel(
            cand_row, text="LINES:",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#64748b"
        ).pack(side="left", padx=(0, 8))

        self.cand_labels: list[ctk.CTkLabel] = []
        for i in range(3):
            lbl = ctk.CTkLabel(
                cand_row, text=f"{i+1}. —",
                font=ctk.CTkFont("Courier", 12),
                text_color="#64748b"
            )
            lbl.pack(side="left", expand=True)
            self.cand_labels.append(lbl)

        # 5. Perspective Toggle (Segmented)
        side_row = ctk.CTkFrame(self, fg_color="transparent")
        side_row.pack(fill="x", padx=20, pady=4)

        ctk.CTkLabel(side_row, text="Perspective:", font=ctk.CTkFont(size=13), text_color="#94a3b8").pack(side="left", padx=(4, 8))

        self.side_seg = ctk.CTkSegmentedButton(
            side_row, values=["Auto", "White", "Black"],
            command=self._on_side_selected,
            selected_color="#3b82f6", selected_hover_color="#2563eb"
        )
        initial_side = self.player_side.capitalize() if self.player_side in ["white", "black"] else "Auto"
        self.side_seg.set(initial_side)
        self.side_seg.pack(side="left", expand=True, fill="x")

        # 6. Action Buttons Grid
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=6)

        # Primary action: Snip Board (starts live recording)
        self.snip_btn = ctk.CTkButton(
            btn_row, text="🎯 Snip Board (Start Live)",
            command=self._start_snip,
            fg_color="#10b981", hover_color="#059669",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=42, corner_radius=8
        )
        self.snip_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        # Pause / Resume Live Tracking
        self.pause_btn = ctk.CTkButton(
            btn_row, text="⏸ Pause Live",
            command=self._toggle_pause_live,
            fg_color="#374151", hover_color="#4b5563",
            font=ctk.CTkFont(size=14),
            height=42, corner_radius=8
        )
        self.pause_btn.pack(side="left", padx=4)

        # Reset button
        ctk.CTkButton(
            btn_row, text="🔄 Reset",
            command=self._reset_all,
            fg_color="#1e293b", hover_color="#334155",
            font=ctk.CTkFont(size=14),
            width=80, height=42, corner_radius=8
        ).pack(side="left", padx=(4, 0))

        # 7. Status bar
        self.status_lbl = ctk.CTkLabel(
            self, text="Ready — Click 🎯 Snip Board once. Live recording will take over.",
            font=ctk.CTkFont(size=12), text_color="#94a3b8"
        )
        self.status_lbl.pack(pady=(4, 10))

        if self._engine_error:
            self._set_status(f"⚠ Engine: {self._engine_error}", error=True)

    def _bind_shortcuts(self):
        self.bind("<space>", lambda e: self._manual_refresh())
        self.bind("<Return>", lambda e: self._manual_refresh())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    #  Mini-Board Rendering & Arrows
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

    def _render_board_image(
        self,
        board: Optional[chess.Board],
        best_move: Optional[chess.Move] = None
    ) -> Image.Image:
        """Draws the mini-board with piece icons, highlights, and glowing arrow."""
        img = Image.new("RGBA", (BOARD_PX, BOARD_PX))
        draw = ImageDraw.Draw(img)

        light = _hex_to_rgba(LIGHT_SQ)
        dark = _hex_to_rgba(DARK_SQ)

        for r in range(8):
            for c in range(8):
                color = light if (r + c) % 2 == 0 else dark
                draw.rectangle(
                    [c * SQ_PX, r * SQ_PX, (c + 1) * SQ_PX, (r + 1) * SQ_PX],
                    fill=color
                )

        if board is not None:
            # Highlight Move Squares
            if best_move:
                fc = _sq_to_canvas(best_move.from_square, self.flipped)
                tc = _sq_to_canvas(best_move.to_square, self.flipped)
                # Origin square highlight: amber gold
                draw.rectangle(
                    [fc[0] * SQ_PX, fc[1] * SQ_PX, (fc[0] + 1) * SQ_PX, (fc[1] + 1) * SQ_PX],
                    fill=(245, 158, 11, 150)
                )
                # Destination square highlight: emerald green
                draw.rectangle(
                    [tc[0] * SQ_PX, tc[1] * SQ_PX, (tc[0] + 1) * SQ_PX, (tc[1] + 1) * SQ_PX],
                    fill=(16, 185, 129, 170)
                )

            # Paste piece graphics
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece is None:
                    continue
                col, row = _sq_to_canvas(sq, self.flipped)
                sym = piece.symbol()
                code = ("w" if piece.color == chess.WHITE else "b") + sym.lower()

                if code in self._piece_images:
                    p_img = self._piece_images[code]
                    img.paste(p_img, (col * SQ_PX, row * SQ_PX), p_img)
                else:
                    # Fallback text symbol
                    txt = PIECE_ICONS.get(sym, sym)
                    clr = (255, 255, 255) if piece.color == chess.WHITE else (20, 20, 20)
                    draw.text((col * SQ_PX + 12, row * SQ_PX + 8), txt, fill=clr)

            # Draw directional move arrow
            if best_move:
                arrow_layer = Image.new("RGBA", (BOARD_PX, BOARD_PX), (0, 0, 0, 0))
                adraw = ImageDraw.Draw(arrow_layer)
                fc = _sq_to_canvas(best_move.from_square, self.flipped)
                tc = _sq_to_canvas(best_move.to_square, self.flipped)
                fx = fc[0] * SQ_PX + SQ_PX // 2
                fy = fc[1] * SQ_PX + SQ_PX // 2
                tx = tc[0] * SQ_PX + SQ_PX // 2
                ty = tc[1] * SQ_PX + SQ_PX // 2
                _draw_arrow(adraw, fx, fy, tx, ty, (239, 68, 68, 220), width=8, head_size=20)
                img = Image.alpha_composite(img, arrow_layer)

        return img

    def _redraw_board(
        self,
        board: Optional[chess.Board],
        best_move: Optional[chess.Move] = None
    ):
        board_img = self._render_board_image(board, best_move)
        self._board_photo = ImageTk.PhotoImage(board_img.convert("RGB"))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._board_photo, anchor="nw")

        # Ranks and Files notation labels
        files = "abcdefgh" if not self.flipped else "hgfedcba"
        ranks = "87654321" if not self.flipped else "12345678"
        for i in range(8):
            self.canvas.create_text(
                i * SQ_PX + SQ_PX - 3, BOARD_PX - 3,
                text=files[i], font=("Helvetica", 8), fill="#78716c", anchor="se"
            )
            self.canvas.create_text(
                3, i * SQ_PX + 3,
                text=ranks[i], font=("Helvetica", 8), fill="#78716c", anchor="nw"
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    #  Live Recording Engine (The Core Feature)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

    def _start_live_tracking(self):
        """Starts background live recording thread."""
        if not self.cfg.get("board_bbox"):
            return
        self.is_live = True
        self.live_badge.configure(text="● LIVE RECORDING", text_color="#10b981")
        self.pause_btn.configure(text="⏸ Pause Live", state="normal")
        threading.Thread(target=self._live_recorder_worker, daemon=True).start()

    def _toggle_pause_live(self):
        self.is_live = not self.is_live
        if self.is_live:
            self.live_badge.configure(text="● LIVE RECORDING", text_color="#10b981")
            self.pause_btn.configure(text="⏸ Pause Live")
            self._set_status("Live recording resumed.")
            threading.Thread(target=self._live_recorder_worker, daemon=True).start()
        else:
            self.live_badge.configure(text="● PAUSED", text_color="#f59e0b")
            self.pause_btn.configure(text="▶ Resume Live")
            self._set_status("Live recording paused.")

    def _live_recorder_worker(self):
        """
        Background daemon that continuously checks the snipped board.
        Uses near-zero CPU when the position has not changed.
        """
        while self.is_live:
            bbox = self.cfg.get("board_bbox")
            if not bbox:
                break

            try:
                # Fast capture of current frame
                current_crop = self.vision.capture_screen(bbox)

                # Check if board pixels changed from last analyzed frame
                if self.last_screen_crop is not None and self.last_screen_crop.shape == current_crop.shape:
                    diff = np.mean(np.abs(current_crop.astype(float) - self.last_screen_crop.astype(float)))
                    if diff < 2.5:
                        # Board hasn't changed at all; sleep and continue
                        time.sleep(0.4)
                        continue

                # Board pixels changed! Brief sleep to let drag/move animation complete
                time.sleep(0.2)
                current_crop = self.vision.capture_screen(bbox)

                # Recognize board
                grid = self.vision.recognize_board(current_crop, auto_detect_theme=True)
                orient = self.bsm.detect_orientation(grid)
                side = orient if self.player_side == "auto" else self.player_side
                turn = "w" if side == "white" else "b"

                fen, board = self.bsm.grid_to_fen(grid, orientation=orient, turn=turn, player_side=side)

                if board is not None and fen != self.last_fen:
                    # New position detected! Analyze immediately
                    self.last_screen_crop = current_crop.copy()
                    self._ui(self._process_board_analysis, board, fen, orient, side)

            except Exception:
                pass

            time.sleep(0.5)

    def _manual_refresh(self):
        """Manual trigger for re-analyzing the current live board."""
        if not self.cfg.get("board_bbox"):
            self._set_status("Please click 🎯 Snip Board first.", error=True)
            return

        def run():
            try:
                bbox = self.cfg["board_bbox"]
                img = self.vision.capture_screen(bbox)
                grid = self.vision.recognize_board(img, auto_detect_theme=True)
                orient = self.bsm.detect_orientation(grid)
                side = orient if self.player_side == "auto" else self.player_side
                turn = "w" if side == "white" else "b"
                fen, board = self.bsm.grid_to_fen(grid, orientation=orient, turn=turn, player_side=side)
                if board:
                    self._ui(self._process_board_analysis, board, fen, orient, side)
            except Exception as e:
                self._ui(self._set_status, f"Refresh error: {e}", error=True)

        threading.Thread(target=run, daemon=True).start()

    def _process_board_analysis(self, board: chess.Board, fen: str, orient: str, side: str):
        self.current_board = board
        self.last_fen = fen
        self.flipped = (orient == "black")

        # Reload piece graphics if theme was auto-adapted
        self._load_piece_graphics()

        result = self.engine.analyze(board, depth=self.cfg.get("depth", 14), player_side=side)
        if result.get("error"):
            self._show_error(result["error"])
            # Still redraw current board even if no move
            self._redraw_board(board, None)
            return

        best_move = result.get("best_move")
        self._redraw_board(board, best_move)

        best_readable = result.get("best_readable", "—")
        score = result.get("score_str", "+0.00")
        win_pct = result.get("win_pct", 50.0)
        cands = result.get("candidates", [])

        # Display plain-English Best Move on ONE line:
        # e.g.: "♞ Knight: G1 ➔ F3 (Nf3)"
        self.move_lbl.configure(text=best_readable, text_color="#4ade80")
        self.eval_badge.configure(text=score)

        # Update Win Probability Advantage meter
        w_pct = int(round(win_pct)) if side == "white" else int(round(100 - win_pct))
        b_pct = 100 - w_pct
        self.win_bar.set(w_pct / 100.0)
        self.w_pct_lbl.configure(text=f"W {w_pct}%")
        self.b_pct_lbl.configure(text=f"{b_pct}% B")

        # Update Candidate Lines
        for i, lbl in enumerate(self.cand_labels):
            if i < len(cands):
                prefix = "★" if i == 0 else f"{i+1}."
                color = "#e2e8f0" if i == 0 else "#94a3b8"
                lbl.configure(text=f"{prefix} {cands[i]}", text_color=color)
            else:
                lbl.configure(text=f"{i+1}. —", text_color="#64748b")

        self._set_status(f"✓ Live: {side.capitalize()} to move ➔ {best_readable}")

        # Audio announcement
        if self.announcer.enabled and result.get("best_san", "—") != "—":
            spoken = self.announcer.format_san_for_speech(result["best_san"])
            self.announcer.announce(spoken)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    #  Controls & Events
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

    def _start_snip(self):
        """Launches interactive desktop snip tool."""
        self.is_live = False
        self._set_status("Click & drag around the 8x8 chessboard on screen…")
        BoardCalibrator(self, on_select=self._on_snip_completed).start()

    def _on_snip_completed(self, rect: tuple[int, int, int, int]):
        x, y, w, h = rect
        self.cfg["board_bbox"] = [x, y, x + w, y + h]
        _save_config(self.cfg)
        self.last_screen_crop = None
        self.last_fen = None
        self._set_status(f"✓ Board locked: {w}×{h} px. Live recording started!")
        self._start_live_tracking()

    def _reset_all(self):
        """Resets the HUD to clean initial state."""
        self.is_live = False
        self.current_board = None
        self.last_fen = None
        self.last_screen_crop = None
        self._redraw_board(None)
        self.move_lbl.configure(text="Snip board once to start live recording", text_color="#94a3b8")
        self.eval_badge.configure(text="+0.00")
        self.win_bar.set(0.5)
        self.w_pct_lbl.configure(text="W 50%")
        self.b_pct_lbl.configure(text="50% B")
        for i, lbl in enumerate(self.cand_labels):
            lbl.configure(text=f"{i+1}. —", text_color="#64748b")
        self.cfg["board_bbox"] = None
        _save_config(self.cfg)
        self.live_badge.configure(text="● READY", text_color="#94a3b8")
        self.pause_btn.configure(text="⏸ Pause Live", state="disabled")
        self._set_status("Board reset. Click 🎯 Snip Board to begin.")

    def _on_side_selected(self, choice: str):
        self.player_side = choice.lower()
        self.cfg["side"] = self.player_side
        _save_config(self.cfg)
        self._set_status(f"Perspective set to {choice}.")
        self.last_fen = None
        self._manual_refresh()

    def _toggle_audio(self):
        self.announcer.enabled = not self.announcer.enabled
        self.cfg["audio"] = self.announcer.enabled
        _save_config(self.cfg)
        self.audio_btn.configure(text="🔊 Audio" if self.announcer.enabled else "🔇 Muted")

    def _set_status(self, msg: str, error: bool = False):
        color = "#ef4444" if error else "#94a3b8"
        self.status_lbl.configure(text=msg, text_color=color)

    def _show_error(self, msg: str):
        self.move_lbl.configure(text="Waiting for legal move...", text_color="#f59e0b")
        self._set_status(msg, error=True)

    def _ui(self, fn, *args):
        """Thread-safe UI dispatcher."""
        self.after(0, fn, *args)

    def _on_close(self):
        self.is_live = False
        if self.engine:
            self.engine.close()
        self.destroy()


def launch():
    app = ChessHUD()
    app.mainloop()