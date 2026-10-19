import cv2
import numpy as np

def detect_chessboard_on_screen(screen_bgr):
    """
    Automatically detects a Chess.com board on the screen image.
    Returns (x, y, w, h) in image coordinates, or None if not found.
    """
    h_img, w_img, _ = screen_bgr.shape
    
    # Chess.com standard green: #769656 (BGR: 86, 150, 118)
    # Chess.com standard cream: #eeeed2 (BGR: 210, 238, 238)
    # Lichess green: #8ca2ad / #dee3e6, Lichess brown: #b58863 / #f0d9b5
    
    # Method: Board squares have distinctive colors and form a large square grid.
    # Convert to HSV
    hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)
    
    # Mask for Chess.com green squares
    # Green in HSV: H: 30-65, S: 50-200, V: 90-200
    green_mask = cv2.inRange(hsv, np.array([28, 45, 80]), np.array([68, 210, 210]))
    
    # Mask for Chess.com brown/wood or Lichess brown squares:
    # Brown in HSV: H: 10-25, S: 60-220, V: 80-220
    brown_mask = cv2.inRange(hsv, np.array([8, 50, 70]), np.array([26, 230, 220]))
    
    # Mask for Lichess blue:
    blue_mask = cv2.inRange(hsv, np.array([95, 40, 80]), np.array([125, 200, 230]))
    
    combined_mask = cv2.bitwise_or(green_mask, brown_mask)
    combined_mask = cv2.bitwise_or(combined_mask, blue_mask)
    
    # Morphological closing to bridge small piece gaps inside squares
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_box = None
    max_area = 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = float(w) / float(h)
        
        # Board must be reasonably large (> 200px on retina, > 100px on standard)
        # and roughly square (aspect ratio between 0.9 and 1.1)
        if 0.90 <= aspect <= 1.10 and area > 40000:
            if area > max_area:
                max_area = area
                # Make it a perfect square
                side = max(w, h)
                best_box = (x, y, side, side)
                
    return best_box

if __name__ == "__main__":
    # Test on a synthetic screen containing a Chess.com board
    screen = np.full((1000, 1600, 3), 40, dtype=np.uint8) # Dark desktop background
    
    # Place a 640x640 Chess.com board at (300, 150)
    bx, by, b_size = 300, 150, 640
    sq_size = b_size // 8
    light_sq = (210, 238, 238)
    dark_sq = (86, 150, 118)
    
    for r in range(8):
        for c in range(8):
            color = light_sq if (r + c) % 2 == 0 else dark_sq
            screen[by + r*sq_size : by + (r+1)*sq_size, bx + c*sq_size : bx + (c+1)*sq_size] = color
            
    detected = detect_chessboard_on_screen(screen)
    print("Detected board at:", detected)
    assert detected is not None
    dx, dy, dw, dh = detected
    assert abs(dx - bx) <= 10 and abs(dy - by) <= 10, f"Location mismatch: {detected}"
    print("Auto-detector test PASSED!")
