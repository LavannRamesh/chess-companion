"""
BoardVision: High-accuracy screen capture, board detection, and piece recognition.
Features automatic piece theme detection (Neo, Alpha, Classic, Wikipedia) so pieces
are recognized accurately on any chess website without manual configuration.
"""

import os
import cv2
import numpy as np
import mss
from PIL import Image

PIECE_NAMES = ["wp", "wn", "wb", "wr", "wq", "wk", "bp", "bn", "bb", "br", "bq", "bk"]
SUPPORTED_THEMES = ["neo", "alpha", "classic", "wikipedia"]


class BoardVision:
    def __init__(self, theme: str = "neo"):
        self.theme = theme
        self.templates: dict[str, Image.Image] = {}
        self.all_theme_templates: dict[str, dict[str, Image.Image]] = {}
        self.scaled_templates: dict[str, tuple] = {}
        self.cached_size = (0, 0)
        self._preload_all_themes()
        self.set_theme(theme)

    def _preload_all_themes(self):
        """Pre-loads RGBA template images for all supported themes."""
        for t in SUPPORTED_THEMES:
            t_dir = os.path.join("assets", "pieces", t)
            if not os.path.isdir(t_dir):
                continue
            theme_dict = {}
            for name in PIECE_NAMES:
                p = os.path.join(t_dir, f"{name}.png")
                if os.path.exists(p):
                    try:
                        theme_dict[name] = Image.open(p).convert("RGBA")
                    except Exception:
                        pass
            self.all_theme_templates[t] = theme_dict

    def set_theme(self, theme: str):
        """Switch piece template theme."""
        self.theme = theme
        self.scaled_templates.clear()
        self.cached_size = (0, 0)
        
        if theme in self.all_theme_templates and self.all_theme_templates[theme]:
            self.templates = dict(self.all_theme_templates[theme])
        else:
            # Fallback to neo or first available
            fallback = "neo" if "neo" in self.all_theme_templates else list(self.all_theme_templates.keys())[0]
            self.templates = dict(self.all_theme_templates.get(fallback, {}))

    def auto_detect_theme(self, board_crop: np.ndarray) -> str:
        """
        Samples key piece squares to automatically determine which piece
        theme is active on the user's board (e.g. alpha vs neo vs classic).
        """
        h, w = board_crop.shape[:2]
        sq_w = max(10, int(round(w / 8.0)))
        sq_h = max(10, int(round(h / 8.0)))

        scores = {}
        # Test 4 corner pieces that are present in almost all opening/midgame boards
        sample_pieces = [(0, 0, "br"), (7, 7, "wr"), (0, 4, "bk"), (7, 4, "wk")]

        for t, tmpl_dict in self.all_theme_templates.items():
            errs = []
            for r, c, code in sample_pieces:
                if code not in tmpl_dict:
                    continue
                y1 = int(round(r * h / 8.0))
                y2 = int(round((r + 1) * h / 8.0))
                x1 = int(round(c * w / 8.0))
                x2 = int(round((c + 1) * w / 8.0))
                sq = board_crop[y1:y2, x1:x2]
                if sq.size == 0:
                    continue

                cs = max(2, int(min(sq.shape[:2]) * 0.12))
                corners = [sq[:cs, :cs], sq[:cs, -cs:], sq[-cs:, :cs], sq[-cs:, -cs:]]
                bg = np.median(np.concatenate([k.reshape(-1, 3) for k in corners]), axis=0).reshape(1, 1, 3)

                im = tmpl_dict[code].resize((x2 - x1, y2 - y1), Image.Resampling.LANCZOS)
                arr = np.array(im)
                tmpl_bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
                tmpl_alpha = arr[:, :, 3]

                a = (tmpl_alpha.astype(float) / 255.0)[:, :, np.newaxis]
                synth = tmpl_bgr.astype(float) * a + bg * (1.0 - a)
                err = np.mean(np.abs(sq.astype(float) - synth))
                errs.append(err)

            if errs:
                scores[t] = np.mean(errs)

        if scores:
            best_theme = min(scores, key=scores.get)
            return best_theme
        return self.theme

    def capture_screen(self, bbox) -> np.ndarray:
        """
        Captures the screen region inside bbox = [left, top, right, bottom].
        Works reliably across standard and Retina displays in logical coordinates.
        """
        left, top, right, bottom = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        width = right - left
        height = bottom - top

        with mss.mss() as sct:
            monitor = {"left": left, "top": top, "width": width, "height": height}
            raw = sct.grab(monitor)
            img = np.array(raw)

        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if bgr.shape[1] != width or bgr.shape[0] != height:
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)

        return bgr

    def align_chessboard_grid(self, img: np.ndarray) -> tuple[int, int, int, int]:
        """
        Detects the exact 8x8 chessboard outer boundaries within img.
        Returns (top, bottom, left, right).
        """
        h_img, w_img = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Canny edge contour detection for outer border
        edges = cv2.Canny(gray, 40, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        min_dim = min(h_img, w_img)

        best_rect = None
        max_area = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h)
            area = w * h
            if 0.90 <= aspect <= 1.10 and w >= int(min_dim * 0.65):
                if area > max_area:
                    max_area = area
                    best_rect = (y, y + h, x, x + w)

        if best_rect is not None:
            top, bottom, left, right = best_rect
            return max(0, top), min(h_img, bottom), max(0, left), min(w_img, right)

        # 2. Variance profiling fallback
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        row_stds = np.std(binary.astype(float), axis=1)
        col_stds = np.std(binary.astype(float), axis=0)

        valid_rows = np.where(row_stds > 30)[0]
        valid_cols = np.where(col_stds > 30)[0]

        if len(valid_rows) > 0 and len(valid_cols) > 0:
            top, bottom = int(valid_rows[0]), int(valid_rows[-1])
            left, right = int(valid_cols[0]), int(valid_cols[-1])
            board_h, board_w = bottom - top, right - left
            if board_h > 100 and board_w > 100 and 0.85 <= (board_w / board_h) <= 1.15:
                return top, bottom, left, right

        return 0, h_img, 0, w_img

    def _prepare_scaled_templates(self, sq_w: int, sq_h: int):
        """Pre-resizes templates to square dimensions for fast matching."""
        if self.cached_size == (sq_w, sq_h) and self.scaled_templates:
            return

        self.scaled_templates.clear()
        for name, img in self.templates.items():
            resized = img.resize((sq_w, sq_h), Image.Resampling.LANCZOS)
            arr = np.array(resized)
            bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            alpha = arr[:, :, 3]
            self.scaled_templates[name] = (bgr, gray, edges, alpha)

        self.cached_size = (sq_w, sq_h)

    def _get_square_bg(self, sq: np.ndarray) -> np.ndarray:
        """Samples the 4 corners of a square to determine background color."""
        h, w = sq.shape[:2]
        cs = max(2, int(min(h, w) * 0.12))
        corners = [
            sq[0:cs, 0:cs],
            sq[0:cs, w - cs:w],
            sq[h - cs:h, 0:cs],
            sq[h - cs:h, w - cs:w]
        ]
        return np.median(np.concatenate([c.reshape(-1, 3) for c in corners]), axis=0)

    def classify_square(self, sq: np.ndarray, is_light_sq: bool) -> str | None:
        """
        Classifies an individual board square.
        Returns piece code ('wp', 'bn', etc.) or None if empty.
        Immune to move highlight borders and background tints.
        """
        h, w = sq.shape[:2]
        if h < 8 or w < 8:
            return None

        # 1. Check center 60% of square for piece presence (immune to highlight borders)
        cy1, cy2 = int(h * 0.20), int(h * 0.80)
        cx1, cx2 = int(w * 0.20), int(w * 0.80)
        center = sq[cy1:cy2, cx1:cx2]
        center_gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        center_edges = cv2.Canny(center_gray, 50, 150)

        # Empty squares have near-zero center variance and edge count
        if np.std(center_gray) < 10.0 and np.count_nonzero(center_edges) < 10:
            return None

        # 2. Background sampling
        bg_color = self._get_square_bg(sq)
        bg_bgr = bg_color.reshape(1, 1, 3)

        # 3. Determine White vs Black piece from foreground pixels
        diff = np.linalg.norm(sq.astype(float) - bg_color.astype(float), axis=2)
        fg_mask = (diff > 25)
        fg_pixels = sq[fg_mask]
        if len(fg_pixels) > 0:
            fg_gray = cv2.cvtColor(fg_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).flatten()
            dark_frac = np.count_nonzero(fg_gray < 80) / float(len(fg_gray))
            mean_fg = np.mean(fg_gray)
            is_white = (dark_frac < 0.50) or (mean_fg > 120)
        else:
            is_white = is_light_sq

        candidates = ["wp", "wn", "wb", "wr", "wq", "wk"] if is_white else ["bp", "bn", "bb", "br", "bq", "bk"]

        # 4. Multi-cue matching (Edge IoU + Mask IoU + Synthetic Composite Error)
        sq_gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
        sq_edges = cv2.Canny(sq_gray, 50, 150)

        sq_float = sq.astype(float)
        best_cand = None
        best_score = -1e9

        for cand in candidates:
            if cand not in self.scaled_templates:
                continue
            tmpl_bgr, tmpl_gray, tmpl_edges, tmpl_alpha = self.scaled_templates[cand]
            t_mask = (tmpl_alpha > 50)

            # Edge IoU
            inter_edge = np.sum((sq_edges > 0) & (tmpl_edges > 0))
            union_edge = np.sum((sq_edges > 0) | (tmpl_edges > 0))
            edge_iou = inter_edge / float(max(1, union_edge))

            # Mask IoU
            inter_mask = np.sum(fg_mask & t_mask)
            union_mask = np.sum(fg_mask | t_mask)
            mask_iou = inter_mask / float(max(1, union_mask))

            # Synthetic Composite Pixel Error
            a = (tmpl_alpha.astype(float) / 255.0)[:, :, np.newaxis]
            synth = tmpl_bgr.astype(float) * a + bg_bgr * (1.0 - a)
            err = np.mean(np.abs(sq_float - synth))

            # Weighted confidence score
            score = edge_iou * 120.0 + (mask_iou * 60.0 if not is_light_sq else 0.0) - err * 0.5
            if score > best_score:
                best_score = score
                best_cand = cand

        return best_cand

    def recognize_board(self, img_bgr: np.ndarray, auto_detect_theme: bool = True) -> list[list[str | None]]:
        """
        Recognizes the complete 8x8 chessboard from an image crop or capture.
        Auto-detects piece theme and uses floating-point subpixel slicing.
        Returns an 8x8 matrix (row 0 = rank 8, col 0 = file a).
        """
        top, bottom, left, right = self.align_chessboard_grid(img_bgr)
        board_crop = img_bgr[top:bottom, left:right]

        # Auto-detect piece theme if enabled
        if auto_detect_theme:
            detected_theme = self.auto_detect_theme(board_crop)
            if detected_theme != self.theme:
                self.set_theme(detected_theme)

        h, w = board_crop.shape[:2]
        avg_sq_w = int(round(w / 8.0))
        avg_sq_h = int(round(h / 8.0))
        self._prepare_scaled_templates(avg_sq_w, avg_sq_h)

        grid = [[None for _ in range(8)] for _ in range(8)]

        for r in range(8):
            y1 = int(round(r * h / 8.0))
            y2 = int(round((r + 1) * h / 8.0))
            for c in range(8):
                x1 = int(round(c * w / 8.0))
                x2 = int(round((c + 1) * w / 8.0))

                sq = board_crop[y1:y2, x1:x2]
                if sq.shape[0] != avg_sq_h or sq.shape[1] != avg_sq_w:
                    sq = cv2.resize(sq, (avg_sq_w, avg_sq_h), interpolation=cv2.INTER_LINEAR)
                is_light_sq = ((r + c) % 2 == 0)
                grid[r][c] = self.classify_square(sq, is_light_sq)

        return grid