import cv2
import numpy as np

def remove_black_border_chessboard(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pattern_size = (7, 7)
    ret, corners = cv2.findChessboardCorners(
        gray, 
        pattern_size, 
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ret:
        raise ValueError("Chessboard corners not found")
    
    corners = corners.reshape(pattern_size[0], pattern_size[1], 2)
    
    dxs = [np.linalg.norm(corners[i, j+1] - corners[i, j]) 
             for i in range(pattern_size[0]) 
             for j in range(pattern_size[1]-1)]
    avg_square_width = np.mean(dxs) if dxs else 0
    
    dys = [np.linalg.norm(corners[i+1, j] - corners[i, j]) 
             for j in range(pattern_size[1]) 
             for i in range(pattern_size[0]-1)]
    avg_square_height = np.mean(dys) if dys else 0
    
    if avg_square_width == 0 or avg_square_height == 0:
        raise ValueError("Failed to compute square size")
    
    board_width = 8 * avg_square_width
    board_height = 8 * avg_square_height
    
    x0 = corners[0, 0, 0] - avg_square_width
    y0 = corners[0, 0, 1] - avg_square_height
    
    x1 = max(0, int(round(x0)))
    y1 = max(0, int(round(y0)))
    x2 = min(img.shape[1], int(round(x0 + board_width)))
    y2 = min(img.shape[0], int(round(y0 + board_height)))
    
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Invalid cropping coordinates")
    
    cropped = img[y1:y2, x1:x2]
    return cropped

def trim_to_board(img):
    """Remove non-board rows/columns from each edge.

    Chess board rows have a bimodal color distribution (alternating light/dark
    squares), so most pixels are far from the row median. Headers, footers, and
    solid-color borders are roughly uniform — most pixels are near the median.
    We remove edge rows/columns until we reach actual board content.
    Works for any border color (black, gray, white, colored UI strips, etc.).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    def is_non_board(strip):
        """True if the row/column looks uniform (border), not like alternating chess squares."""
        median = float(np.median(strip))
        return float(np.mean(np.abs(strip.astype(float) - median) < 25)) > 0.60

    while gray.shape[0] > 50 and is_non_board(gray[0]):
        gray = gray[1:]; img = img[1:]
    while gray.shape[0] > 50 and is_non_board(gray[-1]):
        gray = gray[:-1]; img = img[:-1]
    while gray.shape[1] > 50 and is_non_board(gray[:, 0]):
        gray = gray[:, 1:]; img = img[:, 1:]
    while gray.shape[1] > 50 and is_non_board(gray[:, -1]):
        gray = gray[:, :-1]; img = img[:, :-1]

    return img

def remove_black_border_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    mask_inv = cv2.bitwise_not(mask)
    
    kernel = np.ones((5,5), np.uint8)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)
    
    points = cv2.findNonZero(mask_inv)
    if points is None:
        raise ValueError("No non-black pixels found")
    
    x, y, w, h = cv2.boundingRect(points)
    margin = 2
    cropped = img[
        max(0, y - margin):min(img.shape[0], y + h + margin),
        max(0, x - margin):min(img.shape[1], x + w + margin)
    ]
    return cropped

def remove_black_border_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 51, 10
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found")
    
    best_contour = None
    best_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = min(w, h) / max(w, h)
        if 0.8 <= aspect_ratio <= 1.2 and area > best_area:
            best_area = area
            best_contour = contour
    
    if best_contour is None:
        raise ValueError("No suitable contour found")
    
    x, y, w, h = cv2.boundingRect(best_contour)
    margin = 2
    cropped = img[
        max(0, y - margin):min(img.shape[0], y + h + margin),
        max(0, x - margin):min(img.shape[1], x + w + margin)
    ]
    return cropped

def remove_black_border(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    try:
        cropped = remove_black_border_chessboard(img)
        cropped = trim_to_board(cropped)
        return cropped
    except Exception as e:
        print(f"Chessboard method failed: {e}")
    
    try:
        cropped = remove_black_border_color(img)
        cropped = trim_to_board(cropped)
        return cropped
    except Exception as e:
        print(f"Color-based method failed: {e}")
    
    try:
        cropped = remove_black_border_edges(img)
        cropped = trim_to_board(cropped)
        return cropped
    except Exception as e:
        print(f"Edge-based method failed: {e}")
        raise ValueError("All methods failed to detect chessboard")

def process_board_image(image_path, output_size=750):
    cropped = remove_black_border(image_path)
    return cv2.resize(cropped, (output_size, output_size))