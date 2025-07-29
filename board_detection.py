import cv2

def process_board_image(image_path, output_size=400):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.resize(img, (output_size, output_size))