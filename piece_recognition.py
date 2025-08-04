import cv2
import os
import numpy as np

class PieceRecognizer:
    def __init__(self, pieces_dir="pieces", debug=False):
        self.templates = self.load_templates(pieces_dir)
        self.debug = debug
        self.debug_dir = "debug_squares"
        if debug and not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

    def load_templates(self, dir_path):
        templates = {}
        for file in os.listdir(dir_path):
            if file.endswith(".png"):
                code = file.split(".")[0]
                img = cv2.imread(os.path.join(dir_path, file), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    templates[code] = img
        return templates

    def recognize_piece(self, square_img, row, col):
        if square_img.size == 0:
            return None

        if self.debug:
            cv2.imwrite(f"{self.debug_dir}/square_{row}_{col}_input.png", square_img)

        best_match = None
        max_val = 0

        for code, template in self.templates.items():
            template_resized = cv2.resize(template, (square_img.shape[1], square_img.shape[0]))
            template_gray = self.normalize_gray(template_resized[:, :, :3])
            square_gray = self.normalize_gray(square_img)

            mask = None
            if template_resized.shape[2] == 4:
                mask = template_resized[:, :, 3]
                mask = cv2.resize(mask, (square_img.shape[1], square_img.shape[0]))
                mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

            try:
                res = cv2.matchTemplate(square_gray, template_gray, cv2.TM_CCOEFF_NORMED, mask=mask)
                _, local_max_val, _, _ = cv2.minMaxLoc(res)
            except Exception as e:
                if self.debug:
                    print(f"[{code}] Error: {e}")
                local_max_val = 0

            if self.debug:
                print(f"  {code}: {local_max_val:.3f}")

            if local_max_val > max_val:
                best_match = code
                max_val = local_max_val

        return best_match if max_val > 0.5 else None

    def normalize_gray(self, img):
        """Grayscale and contrast-enhanced"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        return clahe.apply(gray)
