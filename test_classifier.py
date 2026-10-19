import os
import cv2
import numpy as np
from PIL import Image

def get_square_bg_color(sq):
    h, w, _ = sq.shape
    corner_size = max(2, int(min(h, w) * 0.12))
    corners = [
        sq[0:corner_size, 0:corner_size],
        sq[0:corner_size, w-corner_size:w],
        sq[h-corner_size:h, 0:corner_size],
        sq[h-corner_size:h, w-corner_size:w]
    ]
    all_corners = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    return np.median(all_corners, axis=0)

def classify_square(sq, scaled_templates):
    h, w, _ = sq.shape
    bg_color = get_square_bg_color(sq)
    
    # Color difference from background
    diff = np.linalg.norm(sq.astype(float) - bg_color.astype(float), axis=2)
    fg_mask = (diff > 25).astype(np.uint8) * 255
    
    pad = max(2, int(min(h, w) * 0.06))
    inner_fg = fg_mask[pad:h-pad, pad:w-pad]
    
    fg_pixel_count = np.count_nonzero(inner_fg)
    total_inner_pixels = inner_fg.shape[0] * inner_fg.shape[1]
    fg_ratio = fg_pixel_count / total_inner_pixels
    
    # If foreground is tiny, square is empty
    if fg_ratio < 0.06:
        return None
    
    sq_inner = sq[pad:h-pad, pad:w-pad]
    fg_pixels = sq_inner[inner_fg > 0]
    
    fg_gray = cv2.cvtColor(fg_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).flatten()
    median_fg_gray = np.median(fg_gray)
    
    is_white = median_fg_gray > 135
    candidates = ["wp", "wn", "wb", "wr", "wq", "wk"] if is_white else ["bp", "bn", "bb", "br", "bq", "bk"]
    
    best_candidate = None
    min_error = 1e9
    
    bg_bgr = bg_color.reshape(1, 1, 3)
    
    for cand in candidates:
        tmpl_bgr, tmpl_alpha = scaled_templates[cand]
        alpha_norm = (tmpl_alpha.astype(float) / 255.0)[:, :, np.newaxis]
        
        # Composite candidate template onto this square's exact background
        synthetic = (tmpl_bgr.astype(float) * alpha_norm + bg_bgr * (1.0 - alpha_norm))
        
        # Compute mean absolute pixel difference across the square
        error = np.mean(np.abs(sq.astype(float) - synthetic))
        
        if error < min_error:
            min_error = error
            best_candidate = cand
            
    return best_candidate

def test_board_classification():
    theme_dir = "assets/pieces/neo"
    piece_names = ["wp", "wn", "wb", "wr", "wq", "wk", "bp", "bn", "bb", "br", "bq", "bk"]
    templates = {}
    for name in piece_names:
        img_path = os.path.join(theme_dir, f"{name}.png")
        templates[name] = Image.open(img_path).convert("RGBA")

    sq_size = 100
    board_img = np.zeros((sq_size * 8, sq_size * 8, 3), dtype=np.uint8)
    
    light_sq = (210, 238, 238)
    dark_sq = (86, 150, 118)
    
    for r in range(8):
        for c in range(8):
            color = light_sq if (r + c) % 2 == 0 else dark_sq
            board_img[r*sq_size:(r+1)*sq_size, c*sq_size:(c+1)*sq_size] = color

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

    board_pil = Image.fromarray(cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB))
    for r in range(8):
        for c in range(8):
            p = initial_layout[r][c]
            if p:
                piece_pil = templates[p].resize((sq_size, sq_size), Image.Resampling.LANCZOS)
                board_pil.paste(piece_pil, (c * sq_size, r * sq_size), piece_pil)

    test_board = cv2.cvtColor(np.array(board_pil), cv2.COLOR_RGB2BGR)

    scaled_templates = {}
    for name, img in templates.items():
        resized = img.resize((sq_size, sq_size), Image.Resampling.LANCZOS)
        np_img = np.array(resized)
        rgb = np_img[:, :, :3]
        alpha = np_img[:, :, 3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        scaled_templates[name] = (bgr, alpha)

    correct = 0
    total = 64
    for r in range(8):
        for c in range(8):
            expected = initial_layout[r][c]
            sq = test_board[r*sq_size:(r+1)*sq_size, c*sq_size:(c+1)*sq_size]
            detected = classify_square(sq, scaled_templates)
            if detected == expected:
                correct += 1
            else:
                print(f"Mismatch at r={r}, c={c}: Expected {expected}, got {detected}")

    print(f"Classification accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    test_board_classification()
