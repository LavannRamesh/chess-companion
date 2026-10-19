import chess
import re


class BoardStateManager:
    # ------------------------------------------------------------------ #
    #  Orientation detection
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_orientation(grid: list[list]) -> str:
        """
        Infer board orientation from king positions.

        If the white king is in the *bottom half* (rows 4-7) the board is
        displayed with White at the bottom (normal orientation).
        Returns 'white' or 'black'.
        """
        for r in range(8):
            for c in range(8):
                piece = grid[r][c]
                if piece == "wk":
                    return "white" if r >= 4 else "black"
                if piece == "bk":
                    return "black" if r >= 4 else "white"
        # No king found – fall back to white
        return "white"

    # ------------------------------------------------------------------ #
    #  Grid → FEN
    # ------------------------------------------------------------------ #

    @staticmethod
    def grid_to_fen(
        grid: list[list],
        orientation: str = "white",
        turn: str | None = None,
        player_side: str = "white",
    ) -> tuple[str, chess.Board | None]:
        """
        Convert the 8×8 recognised grid to a FEN string and a chess.Board.

        Parameters
        ----------
        grid        : 8×8 list of piece codes ('wp', 'bk', …) or None.
        orientation : 'white' (White at bottom) or 'black' (Black at bottom).
        turn        : 'w' or 'b' – whose turn it is.  If None, inferred from
                      player_side (we assume it's always the user's turn).
        player_side : 'white' or 'black' – the side the user is playing as.
        """
        piece_map = {
            "wp": "P", "wn": "N", "wb": "B", "wr": "R", "wq": "Q", "wk": "K",
            "bp": "p", "bn": "n", "bb": "b", "br": "r", "bq": "q", "bk": "k",
        }

        # When Black is at the bottom the captured grid is upside-down/mirrored
        working_grid = grid
        if orientation == "black":
            working_grid = [row[::-1] for row in reversed(grid)]

        # Build FEN piece placement (rank 8 first)
        fen_rows = []
        for r in range(8):
            empty = 0
            row_str = ""
            for c in range(8):
                piece = working_grid[r][c]
                if piece is None:
                    empty += 1
                else:
                    if empty:
                        row_str += str(empty)
                        empty = 0
                    row_str += piece_map.get(piece, "")
            if empty:
                row_str += str(empty)
            fen_rows.append(row_str)

        placement = "/".join(fen_rows)

        # Active colour – default to the user's side when not specified
        if turn is None:
            if player_side == "black":
                turn = "b"
            elif player_side == "white":
                turn = "w"
            else:
                # Auto perspective
                turn = "b" if orientation == "black" else "w"

        # Use generous castling rights (KQkq); Stockfish will figure it out
        fen = f"{placement} {turn} KQkq - 0 1"

        try:
            board = chess.Board(fen)
            # Validate: kings must exist for a playable position
            if not board.king(chess.WHITE) or not board.king(chess.BLACK):
                return fen, None
            return fen, board
        except (ValueError, Exception):
            # Try relaxed: strip castling rights if board is invalid
            fen_relaxed = f"{placement} {turn} - - 0 1"
            try:
                board = chess.Board(fen_relaxed)
                return fen_relaxed, board
            except Exception:
                return fen, None

    # ------------------------------------------------------------------ #
    #  Sanity check helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def count_pieces(grid: list[list]) -> dict:
        """Return a dict of piece_code → count from the grid."""
        counts: dict[str, int] = {}
        for row in grid:
            for piece in row:
                if piece:
                    counts[piece] = counts.get(piece, 0) + 1
        return counts

    @staticmethod
    def grid_looks_valid(grid: list[list]) -> bool:
        """
        Quick sanity check: both kings present, piece counts plausible.
        Returns True if the grid looks like a real chess position.
        """
        counts = BoardStateManager.count_pieces(grid)
        # Both kings must be present
        if counts.get("wk", 0) != 1 or counts.get("bk", 0) != 1:
            return False
        # No piece should exceed 10 (generous limit)
        for v in counts.values():
            if v > 10:
                return False
        return True