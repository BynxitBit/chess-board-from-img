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
        """Load piece images with standardized processing"""
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
    
    def recognize_piece(self, square_img):
        """Simplified recognition with background removal"""
        if square_img.size == 0:
            return None
            
        if self.debug:
            cv2.imwrite(f"{self.debug_dir}/square_{row}_{col}.png", square_img)
        
        processed_square = self.remove_background(square_img)
        
        best_match = None
        max_val = 0
        
        for code, template in self.templates.items():
            h, w = processed_square.shape[:2]
            template = cv2.resize(template, (w, h))
            
            if template.shape[2] == 4:
                mask = template[:, :, 3]
            else:
                mask = None
            
            template_gray = cv2.cvtColor(template[:, :, :3], cv2.COLOR_BGR2GRAY)
            square_gray = cv2.cvtColor(processed_square, cv2.COLOR_BGR2GRAY)
            
            template_gray = cv2.normalize(template_gray, None, 0, 255, cv2.NORM_MINMAX)
            square_gray = cv2.normalize(square_gray, None, 0, 255, cv2.NORM_MINMAX)
            
            try:
                res = cv2.matchTemplate(square_gray, template_gray, cv2.TM_CCOEFF_NORMED, mask=mask)
                _, local_max_val, _, _ = cv2.minMaxLoc(res)
            except:
                local_max_val = 0
                
            if self.debug:
                print(f"  {code}: {local_max_val:.2f}")
            
            if local_max_val > 0.5 and local_max_val > max_val:
                best_match = code
                max_val = local_max_val
        
        return best_match
    
    def remove_background(self, square_img):
        """Remove board background to isolate pieces"""
        light_color = np.array([234, 233, 210])
        dark_color = np.array([75, 115, 153])
        
        diff_light = np.abs(square_img - light_color)
        diff_dark = np.abs(square_img - dark_color)
        
        mask_light = np.all(diff_light < 50, axis=2)
        mask_dark = np.all(diff_dark < 50, axis=2)
        background_mask = np.logical_or(mask_light, mask_dark)
        
        piece_mask = ~background_mask
        
        result = square_img.copy()
        result[piece_mask] = [0, 0, 0]
        
        return result