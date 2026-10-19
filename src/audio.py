"""Audio announcer for chess moves using macOS native speech synthesis."""

import subprocess
import threading
from typing import Optional

class MoveAnnouncer:
    def __init__(self, enabled: bool = False, voice: str = "Samantha"):
        self.enabled = enabled
        self.voice = voice

    def announce(self, text: str):
        """Speaks the text asynchronously without blocking the UI."""
        if not self.enabled or not text:
            return

        def _speak():
            try:
                # Use macOS native 'say' command with clean rate and voice
                subprocess.run(
                    ["say", "-v", self.voice, "-r", "200", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception:
                try:
                    subprocess.run(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass

        threading.Thread(target=_speak, daemon=True).start()

    @staticmethod
    def format_san_for_speech(san: str) -> str:
        """Translates chess SAN notation into natural speech."""
        if not san:
            return ""
        
        # Replace piece abbreviations
        replacements = {
            "N": "Knight to ",
            "B": "Bishop to ",
            "R": "Rook to ",
            "Q": "Queen to ",
            "K": "King to ",
            "O-O-O": "Castles queenside",
            "O-O": "Castles kingside",
            "x": " takes ",
            "+": " check",
            "#": " checkmate"
        }
        
        spoken = san
        if spoken == "O-O-O":
            return "Castles queenside"
        if spoken == "O-O":
            return "Castles kingside"

        # Check if first char is a piece
        if spoken[0] in "NBRQK":
            piece = replacements[spoken[0]]
            rest = spoken[1:]
            for k, v in replacements.items():
                if k in ["x", "+", "#"]:
                    rest = rest.replace(k, v)
            return piece + rest
        else:
            # Pawn move
            rest = spoken
            for k, v in replacements.items():
                if k in ["x", "+", "#"]:
                    rest = rest.replace(k, v)
            return "Pawn " + rest
