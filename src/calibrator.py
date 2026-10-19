"""Interactive screen snipping tool for calibrating chessboard coordinates without switching desktop spaces."""

import os
import tkinter as tk
from typing import Callable, Optional, Tuple
from PIL import Image, ImageTk, ImageEnhance
import mss

class BoardCalibrator:
    def __init__(self, root: tk.Tk, on_select: Callable[[Tuple[int, int, int, int]], None]):
        self.root = root
        self.on_select = on_select
        self.overlay: Optional[tk.Toplevel] = None
        self.canvas: Optional[tk.Canvas] = None
        self.bg_photo: Optional[ImageTk.PhotoImage] = None

        self.start_x = 0
        self.start_y = 0
        self.cur_x = 0
        self.cur_y = 0

        self.rect_id = None
        self.grid_lines = []
        self.dim_rect_id = None
        self.dim_text_id = None

    def start(self):
        """Captures the screen and displays the frozen desktop snip overlay."""
        if self.overlay is not None:
            return

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        screen_img = None
        try:
            with mss.mss() as sct:
                raw = sct.grab({"left": 0, "top": 0, "width": screen_w, "height": screen_h})
                screen_img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        except Exception:
            try:
                from PIL import ImageGrab
                screen_img = ImageGrab.grab()
            except Exception:
                screen_img = None

        # Borderless window covering screen without macOS fullscreen space transition
        self.overlay = tk.Toplevel(self.root)
        self.overlay.overrideredirect(True)
        self.overlay.geometry(f"{screen_w}x{screen_h}+0+0")
        self.overlay.attributes("-topmost", True)
        self.overlay.configure(bg="#000000", cursor="crosshair")

        self.canvas = tk.Canvas(
            self.overlay,
            width=screen_w,
            height=screen_h,
            bg="#000000",
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Dim frozen background slightly so the bright green selection box stands out
        if screen_img is not None:
            dimmed = screen_img.resize((screen_w, screen_h), Image.Resampling.BILINEAR)
            enhancer = ImageEnhance.Brightness(dimmed)
            dimmed = enhancer.enhance(0.75)
            self.bg_photo = ImageTk.PhotoImage(dimmed)
            self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")

        # Instruction Pill
        banner_w = 480
        banner_h = 42
        bx1 = (screen_w - banner_w) // 2
        by1 = 25
        bx2 = bx1 + banner_w
        by2 = by1 + banner_h

        self.canvas.create_rectangle(
            bx1, by1, bx2, by2,
            fill="#0f172a", outline="#10b981", width=2
        )
        self.canvas.create_text(
            (bx1 + bx2) // 2, (by1 + by2) // 2,
            text="🎯 Click & Drag around the 8x8 Chessboard  |  ESC to cancel",
            fill="#f8fafc",
            font=("Helvetica", 13, "bold")
        )

        self.overlay.bind("<ButtonPress-1>", self._on_press)
        self.overlay.bind("<B1-Motion>", self._on_motion)
        self.overlay.bind("<ButtonRelease-1>", self._on_release)
        self.overlay.bind("<Escape>", lambda e: self.close())
        self.overlay.focus_force()

    def _on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.cur_x = event.x
        self.cur_y = event.y

    def _on_motion(self, event):
        self.cur_x = event.x
        self.cur_y = event.y
        self._redraw_selection()

    def _redraw_selection(self):
        if not self.canvas:
            return

        if self.rect_id:
            self.canvas.delete(self.rect_id)
        if self.dim_rect_id:
            self.canvas.delete(self.dim_rect_id)
        if self.dim_text_id:
            self.canvas.delete(self.dim_text_id)
        for line_id in self.grid_lines:
            self.canvas.delete(line_id)
        self.grid_lines.clear()

        x1 = min(self.start_x, self.cur_x)
        y1 = min(self.start_y, self.cur_y)
        x2 = max(self.start_x, self.cur_x)
        y2 = max(self.start_y, self.cur_y)
        w = x2 - x1
        h = y2 - y1

        if w < 5 or h < 5:
            return

        # Outer green selection boundary
        self.rect_id = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="#10b981", width=3, fill=""
        )

        # 8x8 Grid lines
        for i in range(1, 8):
            lx = x1 + (w * i) // 8
            self.grid_lines.append(
                self.canvas.create_line(lx, y1, lx, y2, fill="#34d399", dash=(4, 4), width=1)
            )
            ly = y1 + (h * i) // 8
            self.grid_lines.append(
                self.canvas.create_line(x1, ly, x2, ly, fill="#34d399", dash=(4, 4), width=1)
            )

        # Size badge
        dim_text = f"Chessboard: {w} × {h} px"
        tx = x1 + 10
        ty = max(15, y1 - 22)

        self.dim_rect_id = self.canvas.create_rectangle(
            tx - 6, ty - 4, tx + 160, ty + 18,
            fill="#0f172a", outline="#10b981", width=1
        )
        self.dim_text_id = self.canvas.create_text(
            tx, ty + 7,
            text=dim_text,
            anchor="w",
            fill="#34d399",
            font=("Helvetica", 11, "bold")
        )

    def _on_release(self, event):
        x1 = min(self.start_x, self.cur_x)
        y1 = min(self.start_y, self.cur_y)
        x2 = max(self.start_x, self.cur_x)
        y2 = max(self.start_y, self.cur_y)
        w = x2 - x1
        h = y2 - y1

        self.close()

        if w >= 80 and h >= 80:
            self.on_select((x1, y1, w, h))

    def close(self):
        if self.overlay:
            try:
                self.overlay.destroy()
            except Exception:
                pass
            self.overlay = None
            self.canvas = None
            self.bg_photo = None
