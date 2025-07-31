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
        
        self.light_color = np.array([234, 233, 210])
        self.dark_color = np.array([75, 115, 153])
        self.color_threshold = 40
    
    def load_templates(self, dir_path):
        templates = {}
        for file in os.listdir(dir_path):
            if file.endswith(".png"):
                piece_code = file.split(".")[0]
                img = cv2.imread(os.path.join(dir_path, file), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    
                    img = cv2.copyMakeBorder(
                        img, 5, 5, 5, 5, 
                        cv2.BORDER_CONSTANT, 
                        value=[0, 0, 0, 0]
                    )
                    templates[piece_code] = img
        return templates
    
    def recognize_piece(self, square_img, row, col):
        if square_img.size == 0:
            return None
            
        processed_square = self.remove_background(square_img)
        
        if self.debug:
            cv2.imwrite(f"{self.debug_dir}/square_{row}_{col}.png", processed_square)
        
        best_match = None
        max_val = 0
        
        for code, template in self.templates.items():
            h, w = processed_square.shape[:2]
            resized_template = cv2.resize(template, (w, h))
            
            if resized_template.shape[2] == 4:
                mask = resized_template[:, :, 3]
            else:
                mask = None
            
            template_gray = cv2.cvtColor(resized_template[:, :, :3], cv2.COLOR_BGR2GRAY)
            square_gray = cv2.cvtColor(processed_square, cv2.COLOR_BGR2GRAY)
            
            template_gray = cv2.normalize(template_gray, None, 0, 255, cv2.NORM_MINMAX)
            square_gray = cv2.normalize(square_gray, None, 0, 255, cv2.NORM_MINMAX)
            
            try:
                res = cv2.matchTemplate(square_gray, template_gray, cv2.TM_CCOEFF_NORMED, mask=mask)
                _, local_max_val, _, _ = cv2.minMaxLoc(res)
            except Exception as e:
                if self.debug:
                    print(f"Match error: {e}")
                local_max_val = 0
                
            if self.debug:
                print(f"  {code}: {local_max_val:.2f}")
            
            if local_max_val > 0.5 and local_max_val > max_val:
                best_match = code
                max_val = local_max_val
        
        return best_match
    
    def remove_background(self, square_img):
        diff_light = np.linalg.norm(square_img - self.light_color, axis=2)
        diff_dark = np.linalg.norm(square_img - self.dark_color, axis=2)
        
        background_mask = np.logical_or(
            diff_light < self.color_threshold,
            diff_dark < self.color_threshold
        )
        
        result = np.zeros_like(square_img)
        result[~background_mask] = square_img[~background_mask]
        
        return result