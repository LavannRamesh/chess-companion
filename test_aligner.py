import cv2
import numpy as np
from PIL import Image

def align_chessboard_grid(cropped_img):
    """
    Given an image of a chessboard that might include outer borders,
    rank numbers, file letters, or margins, detects the exact inner
    8x8 checkerboard boundaries [top, bottom, left, right].
    """
    h, w, _ = cropped_img.shape
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Compute horizontal and vertical gradients
    # Checkerboard boundaries have periodic strong transitions between light and dark squares
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    
    # Absolute gradient profile
    y_profile = np.mean(np.abs(grad_y), axis=1)
    x_profile = np.mean(np.abs(grad_x), axis=0)
    
    # 2. Smooth profile
    ksize = max(3, int(min(h, w) * 0.015))
    if ksize % 2 == 0: ksize += 1
    y_smooth = cv2.GaussianBlur(y_profile.reshape(-1, 1), (1, ksize), 0).flatten()
    x_smooth = cv2.GaussianBlur(x_profile.reshape(-1, 1), (1, ksize), 0).flatten()
    
    # The 8x8 board has 7 internal grid transitions and 2 outer borders
    # Find peaks with distance roughly (size / 8)
    expected_sq = min(h, w) / 8.0
    min_dist = int(expected_sq * 0.70)
    
    # Find active region of squares
    # Light and dark squares create high local variance across the board
    otsu_thresh, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Scan rows and cols for checkerboard variance
    row_stds = np.std(binary, axis=1)
    col_stds = np.std(binary, axis=0)
    
    # Rows and cols with checkerboard have std > 35
    valid_rows = np.where(row_stds > 35)[0]
    valid_cols = np.where(col_stds > 35)[0]
    
    if len(valid_rows) > 0 and len(valid_cols) > 0:
        top = valid_rows[0]
        bottom = valid_rows[-1]
        left = valid_cols[0]
        right = valid_cols[-1]
        
        # Check aspect ratio
        board_h = bottom - top
        board_w = right - left
        if board_h > 100 and board_w > 100 and 0.85 <= (board_w / board_h) <= 1.15:
            # Refine to exact square
            return top, bottom, left, right
            
    return 0, h, 0, w

if __name__ == "__main__":
    img = cv2.imread("/Users/lavannramesh/Desktop/chess-companion/sample_board.png")
    top, bottom, left, right = align_chessboard_grid(img)
    print(f"Aligned bounds: top={top}, bottom={bottom}, left={left}, right={right}")
    aligned = img[top:bottom, left:right]
    print("Aligned board shape:", aligned.shape)
