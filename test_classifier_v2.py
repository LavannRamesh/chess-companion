import cv2
import numpy as np
from PIL import Image

def test_v2_classifier():
    # Load 365chess / alpha pieces
    theme_dir = "assets/pieces/alpha"
    piece_names = ["wp", "wn", "wb", "wr", "wq", "wk", "bp", "bn", "bb", "br", "bq", "bk"]
    templates = {}
    for name in piece_names:
        templates[name] = Image.open(f"{theme_dir}/{name}.png").convert("RGBA")

    # Read the exact board from user screenshot
    board_img = cv2.imread("/Users/lavannramesh/Desktop/chess-companion/sample_board.png")
    # From y=50 to 546, x=2 to 494
    board_squares = board_img[50:546, 2:494]
    h, w, _ = board_squares.shape
    sq_h = h // 8
    sq_w = w // 8

    # Expected board state (after 1. e4 e5):
    # White played e4 (from e2), Black played e5 (from e7)
    expected_layout = [
        ["br", "bn", "bb", "bq", "bk", "bb", "bn", "br"],
        ["bp", "bp", "bp", "bp", None, "bp", "bp", "bp"],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, "bp", None, None, None],
        [None, None, None, None, "wp", None, None, None],
        [None, None, None, None, None, None, None, None],
        ["wp", "wp", "wp", "wp", None, "wp", "wp", "wp"],
        ["wr", "wn", "wb", "wq", "wk", "wb", "wn", "wr"]
    ]

    # Pre-scale templates
    scaled_templates = {}
    for name, img in templates.items():
        resized = img.resize((sq_w, sq_h), Image.Resampling.LANCZOS)
        arr = np.array(resized)
        bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = arr[:, :, 3]
        scaled_templates[name] = (bgr, alpha)

    correct = 0
    total = 64

    for r in range(8):
        row_str = []
        for c in range(8):
            expected = expected_layout[r][c]
            sq = board_squares[r*sq_h:(r+1)*sq_h, c*sq_w:(c+1)*sq_w]
            
            # 1. Occupancy check: inner 70% variance
            py = int(sq_h * 0.15)
            px = int(sq_w * 0.15)
            inner = sq[py:sq_h-py, px:sq_w-px]
            gray_inner = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
            std = np.std(gray_inner)
            
            if std < 10.0:
                detected = None
            else:
                # 2. Color check: White vs Black
                is_light_sq = ((r + c) % 2 == 0)
                dark_pixels = np.count_nonzero(gray_inner < 80)
                white_pixels = np.count_nonzero(gray_inner > 190)
                
                if is_light_sq:
                    # Light square (white bg):
                    # Black pieces have solid dark body (> 400 dark pixels)
                    # White pieces on white bg have only dark outline (< 250 dark pixels)
                    is_white = dark_pixels < 280
                else:
                    # Dark square (blue bg):
                    # White pieces have bright white fill (> 100 white pixels)
                    # Black pieces have no white fill (< 20 white pixels)
                    is_white = white_pixels > 80
                    
                candidates = ["wp", "wn", "wb", "wr", "wq", "wk"] if is_white else ["bp", "bn", "bb", "br", "bq", "bk"]
                
                # 3. Match against candidate templates using edge and contour correlation
                best_cand = None
                best_score = -1e9
                
                sq_gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
                sq_edges = cv2.Canny(sq_gray, 50, 150)
                
                for cand in candidates:
                    tmpl_bgr, tmpl_alpha = scaled_templates[cand]
                    tmpl_gray = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY)
                    tmpl_edges = cv2.Canny(tmpl_gray, 50, 150)
                    
                    # Edge correlation
                    denom = (np.sqrt(np.sum(sq_edges.astype(float)**2) * np.sum(tmpl_edges.astype(float)**2)) + 1e-5)
                    edge_score = np.sum(sq_edges.astype(float) * tmpl_edges.astype(float)) / denom
                    
                    # Also normalized cross correlation of grayscale
                    res = cv2.matchTemplate(sq_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED, mask=tmpl_alpha)
                    norm_score = res[0, 0]
                    
                    score = edge_score * 0.6 + norm_score * 0.4
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                        
                detected = best_cand
                
            row_str.append(detected if detected else "..")
            if detected == expected:
                correct += 1
            else:
                print(f"Mismatch at ({r},{c}): Expected {expected}, got {detected}")
                
        print(f"Row {r}: {' '.join(row_str)}")
        
    print(f"\nV2 Classifier Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    test_v2_classifier()
