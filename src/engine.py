"""
ChessEngine: Stockfish engine integration with auto-recovery and plain-English move generation.
"""

import os
import chess
import chess.engine

PIECE_NAMES = {
    chess.PAWN: "Pawn",
    chess.KNIGHT: "Knight",
    chess.BISHOP: "Bishop",
    chess.ROOK: "Rook",
    chess.QUEEN: "Queen",
    chess.KING: "King",
}


class ChessEngine:
    def __init__(self, engine_path: str = "bin/stockfish/stockfish-macos-universal"):
        self.engine_path = engine_path
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(
                f"Stockfish binary not found at '{self.engine_path}'.\n"
                "Please ensure the Stockfish binary exists in bin/stockfish/."
            )
        if not os.access(self.engine_path, os.X_OK):
            os.chmod(self.engine_path, 0o755)

        self.engine = None
        self._init_engine()

    def _init_engine(self):
        """Initializes or restarts the Stockfish UCI subprocess."""
        if self.engine:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None

        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        except Exception as exc:
            self.engine = None
            raise RuntimeError(f"Failed to start Stockfish: {exc}")

    def _ensure_engine(self):
        """Ensures the Stockfish process is active and restarts if dead."""
        if self.engine is None:
            self._init_engine()
            return

        try:
            # Quick ping to check if process is alive
            self.engine.ping()
        except Exception:
            self._init_engine()

    def analyze(
        self,
        board: chess.Board,
        depth: int = 15,
        multipv: int = 3,
        player_side: str = "white",
    ) -> dict:
        """
        Analyses board and returns the best move with plain-English instructions.
        Safe against illegal boards or engine crashes.
        """
        empty_result = {
            "best_move": None,
            "best_san": "—",
            "best_uci": "—",
            "best_readable": "No move available",
            "from_sq": "",
            "to_sq": "",
            "score_str": "+0.00",
            "win_pct": 50.0,
            "win_chance": 50.0,
            "candidates": [],
            "error": None,
        }

        if board is None:
            return {**empty_result, "error": "No board position provided."}

        # Check kings existence
        if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            return {**empty_result, "error": "Kings missing — please re-snip the board."}

        # Sanitize board validity before sending to Stockfish
        working_board = board.copy()
        if not working_board.is_valid():
            # Try clearing castling rights
            working_board.castling_rights = 0
            if not working_board.is_valid():
                # Try flipping turn if opponent was in check
                working_board.turn = not working_board.turn
                if not working_board.is_valid():
                    return {**empty_result, "error": "Position is invalid or waiting for piece to land."}

        if not working_board.legal_moves:
            if working_board.is_checkmate():
                winner = "White" if working_board.turn == chess.BLACK else "Black"
                return {
                    **empty_result,
                    "score_str": f"{winner} wins by checkmate",
                    "best_readable": f"Checkmate ({winner} wins)",
                    "error": "Checkmate",
                }
            return {
                **empty_result,
                "score_str": "0.00",
                "best_readable": "Stalemate (Draw)",
                "error": "Stalemate or no legal moves.",
            }

        self._ensure_engine()

        try:
            num_lines = min(multipv, working_board.legal_moves.count())
            info = self.engine.analyse(
                working_board,
                chess.engine.Limit(depth=depth),
                multipv=num_lines,
            )
        except Exception as exc:
            # If engine died, restart it
            try:
                self._init_engine()
            except Exception:
                pass
            return {**empty_result, "error": f"Engine error: {exc}"}

        if not info:
            return {**empty_result, "error": "Stockfish returned no evaluation."}

        best_info = info[0]
        best_move = best_info["pv"][0] if best_info.get("pv") else None
        if best_move is None:
            return {**empty_result, "error": "No best move found."}

        # Format Plain-English move description on one line
        piece = working_board.piece_at(best_move.from_square)
        p_name = PIECE_NAMES.get(piece.piece_type, "Piece") if piece else "Piece"
        from_str = chess.square_name(best_move.from_square).upper()
        to_str = chess.square_name(best_move.to_square).upper()
        san = working_board.san(best_move)
        best_readable = f"{p_name} {from_str} ➔ {to_str} ({san})"

        # Score & Win Probability
        raw_score = best_info["score"].white()
        if raw_score.is_mate():
            mate_n = raw_score.mate()
            score_str = f"M{abs(mate_n)}" if mate_n > 0 else f"-M{abs(mate_n)}"
            win_pct = 100.0 if mate_n > 0 else 0.0
        else:
            cp = raw_score.score(mate_score=10000)
            score_str = f"{cp / 100:+.2f}"
            win_pct = 50 + 50 * (2 / (1 + 10 ** (-cp / 400)) - 1)

        # Flip score if evaluating for Black
        if player_side == "black":
            if score_str.startswith("+"):
                score_str = "-" + score_str[1:]
            elif score_str.startswith("-"):
                score_str = "+" + score_str[1:]
            win_pct = 100.0 - win_pct

        candidates = []
        for entry in info:
            pv = entry.get("pv")
            if pv:
                try:
                    candidates.append(working_board.san(pv[0]))
                except Exception:
                    pass

        return {
            "best_move": best_move,
            "best_san": san,
            "best_uci": best_move.uci(),
            "best_readable": best_readable,
            "from_sq": from_str,
            "to_sq": to_str,
            "score_str": score_str,
            "win_pct": max(0.0, min(100.0, win_pct)),
            "win_chance": round(max(0.0, min(100.0, win_pct)), 1),
            "candidates": candidates,
            "error": None,
        }

    def close(self):
        if self.engine:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None

    def __del__(self):
        self.close()