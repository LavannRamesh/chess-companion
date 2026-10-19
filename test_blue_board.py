import os
import cv2
import numpy as np
from PIL import Image

def test_full_composite_classifier():
    theme_dir = "assets/pieces/alpha"
    piece_names = ["wp", "wn", "wb", "wr", "wq", "wk", "bp", "bn", "bb", "br", "bq", "bk"]
    templates = {}
    for name in piece_names:
        templates[name] = Image.open(f"{theme_dir}/{name}.png").convert("RGBA")

    sq_size = 84
    # Blue square and pure white square
    dark_sq = (179, 136, 112)
    light_sq = (255, 255, 255)

    initial_layout = [
        ["br", "bn", "bb", "bq", "bk", "bb", "bn", "br"],
        ["bp", "bp", "bp", "bp", "bp", "bp", "bp", "bp"],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        ["wp", "wp", "wp", "wp", "wp", "wp", "wp", "wp"],
        ["wr", "wn", "wb", "wq", "wk", "wb", "wn", "wr"]
    ]

    board_pil = Image.new("RGB", (sq_size * 8, sq_size * 8))
    for r in range(8):
        for c in range(8):
            color = light_sq if (r + c) % 2 == 0 else dark_sq
            color_rgb = (color[2], color[1], color[0])
            sq_img = Image.new("RGB", (sq_size, sq_size), color_rgb)
            p = initial_layout[r][c]
            if p:
                p_img = templates[p].resize((sq_size, sq_size), Image.Resampling.LANCZOS)
                sq_img.paste(p_img, (0, 0), p_img)
            board_pil.paste(sq_img, (c * sq_size, r * sq_size))

    board_bgr = cv2.cvtColor(np.array(board_pil), cv2.COLOR_RGB2BGR)

    scaled_templates = {}
    for name, img in templates.items():
        resized = img.resize((sq_size, sq_size), Image.Resampling.LANCZOS)
        arr = np.array(resized)
        scaled_templates[name] = (cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR), arr[:, :, 3])

    def get_bg(sq):
        h, w, _ = sq.shape
        cs = max(2, int(min(h, w) * 0.12))
        corners = [
            sq[0:cs, 0:cs], sq[0:cs, w-cs:w],
            sq[h-cs:h, 0:cs], sq[h-cs:h, w-cs:w]
        ]
        return np.median(np.concatenate([c.reshape(-1, 3) for c in corners]), axis=0)

    def classify_direct(sq):
        h, w, _ = sq.shape
        bg = get_bg(sq)
        bg_bgr = bg.reshape(1, 1, 3)

        # Empty square reference: solid bg
        empty_err = np.mean(np.abs(sq.astype(float) - bg_bgr))
        if empty_err < 4.0:
            return None

        # Compare against all 12 piece templates composited on bg
        best_cand = None
        min_err = 1e9

        for cand in piece_names:
            tmpl_bgr, tmpl_alpha = scaled_templates[cand]
            alpha_norm = (tmpl_alpha.astype(float) / 255.0)[:, :, np.newaxis]
            synthetic = tmpl_bgr.astype(float) * alpha_norm + bg_bgr * (1.0 - alpha_norm)
            err = np.mean(np.abs(sq.astype(float) - synthetic))

            if err < min_err:
                min_err = err
                best_cand = cand

        # If best match is barely better than empty, it's empty
        if min_err >= empty_err * 0.90:
            return None

        return best_cand

    correct = 0
    for r in range(8):
        for c in range(8):
            expected = initial_layout[r][c]
            det = classify_direct(board_bgr[r*sq_size:(r+1)*sq_size, c*sq_size:(c+1)*sq_size])
            if det == expected:
                correct += 1
            else:
                print(f"Mismatch at r={r}, c={c}: Expected {expected}, got {det}")

    print(f"Direct Composite Classifier accuracy: {correct}/64 ({correct/64*100:.1f}%)")

if __name__ == "__main__":
    test_full_composite_classifier()
