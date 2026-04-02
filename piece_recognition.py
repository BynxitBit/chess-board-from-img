import cv2
import os
import numpy as np

class PieceRecognizer:
    def __init__(self, pieces_dir="pieces", debug=False):
        self.templates = self.load_templates(pieces_dir)
        self.debug = debug
        self.debug_dir = "debug_squares"
        self.light_bg = None
        self.dark_bg = None
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

    def calibrate(self, board_img, square_size, border_size):
        """Compute median background brightness for light and dark squares.
        Uses center 60% of each square to avoid coordinate labels and borders."""
        light_vals, dark_vals = [], []
        for row in range(8):
            for col in range(8):
                y1 = border_size + row * square_size
                x1 = border_size + col * square_size
                sq = board_img[y1:y1+square_size, x1:x1+square_size]
                if sq.size == 0:
                    continue
                gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY) if len(sq.shape) == 3 else sq.copy()
                h, w = gray.shape
                m = max(2, h // 5)
                center_median = float(np.median(gray[m:h-m, m:w-m]))
                if (row + col) % 2 == 0:
                    light_vals.append(center_median)
                else:
                    dark_vals.append(center_median)
        if light_vals:
            self.light_bg = float(np.median(light_vals))
        if dark_vals:
            self.dark_bg = float(np.median(dark_vals))
        if self.debug:
            print(f"Calibrated: light_bg={self.light_bg:.1f}, dark_bg={self.dark_bg:.1f}")

    def recognize_piece(self, square_img, row, col):
        if square_img.size == 0:
            return None

        if self.debug:
            cv2.imwrite(f"{self.debug_dir}/square_{row}_{col}_input.png", square_img)

        if self.is_empty_square(square_img, row, col):
            if self.debug:
                print(f"  Empty square detected")
            return None

        piece_color = self.get_piece_color(square_img, row, col)

        best_match = None
        max_val = -1

        square_processed = self.preprocess_square(square_img)

        for code, template in self.templates.items():
            # Filter by detected piece color to avoid white/black confusion
            if piece_color is not None and not code.startswith(piece_color):
                continue

            match_scores = []

            score1 = self.template_match(square_processed, template, code)
            if score1 is not None:
                match_scores.append(score1)

            score2 = self.feature_match(square_processed, template, code)
            if score2 is not None:
                match_scores.append(score2)

            score3 = self.histogram_match(square_processed, template, code)
            if score3 is not None:
                match_scores.append(score3)

            score4 = self.edge_match(square_processed, template, code)
            if score4 is not None:
                match_scores.append(score4)

            if match_scores:
                local_max_val = min(max(match_scores), 1.0)
            else:
                local_max_val = 0

            if self.debug:
                print(f"  {code}: {local_max_val:.3f}")

            if local_max_val > max_val:
                best_match = code
                max_val = local_max_val

        return best_match if max_val > 0.45 else None

    def is_empty_square(self, square_img, row, col):
        try:
            if len(square_img.shape) == 3:
                gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = square_img.copy()

            h, w = gray.shape
            m = max(2, h // 5)
            center = gray[m:h-m, m:w-m]

            # Method 1: inner-corner background estimate
            # Skip outermost ~5px to avoid anti-aliased boundary bleed
            bd = max(5, h // 15)
            cs = max(4, h // 12)
            corner_pixels = np.concatenate([
                gray[bd:bd+cs,    bd:bd+cs   ].flatten(),
                gray[bd:bd+cs,    w-bd-cs:w-bd].flatten(),
                gray[h-bd-cs:h-bd, bd:bd+cs  ].flatten(),
                gray[h-bd-cs:h-bd, w-bd-cs:w-bd].flatten()
            ])
            bg_corner = np.median(corner_pixels)
            diff_corner = np.abs(center.astype(float) - bg_corner)
            ratio_corner = np.mean(diff_corner > 8)

            # Method 2: calibrated parity-based background
            ratio_calib = 1.0  # assume not empty if no calibration
            if self.light_bg is not None and self.dark_bg is not None:
                bg_calib = self.light_bg if (row + col) % 2 == 0 else self.dark_bg
                diff_calib = np.abs(center.astype(float) - bg_calib)
                ratio_calib = np.mean(diff_calib > 8)

            # Take min: a square is empty only if BOTH methods agree it's empty.
            # This handles coordinate labels (which inflate corner bg) and
            # last-move highlights (which shift calibrated bg).
            non_bg_ratio = min(ratio_corner, ratio_calib)
            is_empty = non_bg_ratio < 0.12

            if self.debug:
                print(f"    Empty check: corner_bg={bg_corner:.1f}, ratio_corner={ratio_corner:.3f}, "
                      f"ratio_calib={ratio_calib:.3f} -> {is_empty}")

            return is_empty

        except Exception as e:
            if self.debug:
                print(f"    Empty check error: {e}")
            return False

    def get_piece_color(self, square_img, row, col):
        """Returns 'w', 'b', or None if color cannot be determined.

        Uses center_mean vs corner_bg difference. corner_bg reflects the actual
        square background (including last-move highlights) rather than the
        calibrated value, so a white pawn on a highlighted blue square is still
        correctly identified as bright relative to its background.
        """
        try:
            if len(square_img.shape) == 3:
                gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = square_img.copy()

            h, w = gray.shape
            m = max(2, h // 5)
            center = gray[m:h-m, m:w-m].astype(float)

            bd = max(5, h // 15)
            cs = max(4, h // 12)
            corner_pixels = np.concatenate([
                gray[bd:bd+cs,    bd:bd+cs   ].flatten(),
                gray[bd:bd+cs,    w-bd-cs:w-bd].flatten(),
                gray[h-bd-cs:h-bd, bd:bd+cs  ].flatten(),
                gray[h-bd-cs:h-bd, w-bd-cs:w-bd].flatten()
            ])
            bg = float(np.median(corner_pixels))
            center_mean = float(np.mean(center))
            diff = center_mean - bg  # positive → piece brighter than bg (white piece)

            if bg > 150:
                # Light square (or highlighted light ~130–230):
                # white piece body ≈ bg → diff slightly negative (−25 to 0)
                # black piece body ≈ 40–80 → diff strongly negative (< −50)
                if diff > -25:
                    color = 'w'
                elif diff < -50:
                    color = 'b'
                else:
                    color = None
            else:
                # Dark square (or highlighted dark ~100–150):
                # white piece ≫ bg → diff strongly positive (> 20)
                # black piece ≈ bg or darker → diff slightly negative (< −10)
                if diff > 20:
                    color = 'w'
                elif diff < -10:
                    color = 'b'
                else:
                    color = None

            if self.debug:
                print(f"    Color: bg={bg:.1f}, center_mean={center_mean:.1f}, "
                      f"diff={diff:+.1f} -> {color}")

            return color

        except Exception as e:
            if self.debug:
                print(f"    Color detection error: {e}")
            return None

    def preprocess_square(self, square_img):
        if len(square_img.shape) == 3:
            gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = square_img.copy()

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

        return normalized

    def template_match(self, square_img, template, code):
        try:
            template_resized = cv2.resize(template, (square_img.shape[1], square_img.shape[0]))

            if len(template_resized.shape) == 3:
                template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template_resized.copy()

            template_processed = self.preprocess_square(template_gray)

            mask = None
            if template_resized.shape[2] == 4:
                mask = template_resized[:, :, 3]
                mask = cv2.resize(mask, (square_img.shape[1], square_img.shape[0]))
                mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

            # TM_CCORR_NORMED is excluded: it returns ~0.85 for any uniform
            # (empty) square vs a white-background template, causing false positives.
            methods = [
                (cv2.TM_CCOEFF_NORMED, 1.0),
                (cv2.TM_SQDIFF_NORMED, 0.6)
            ]

            scores = []
            for method, weight in methods:
                try:
                    res = cv2.matchTemplate(square_img, template_processed, method, mask=mask)
                    _, local_max_val, _, _ = cv2.minMaxLoc(res)

                    if method == cv2.TM_SQDIFF_NORMED:
                        local_max_val = 1.0 - local_max_val

                    if np.isfinite(local_max_val) and 0 <= local_max_val <= 1:
                        scores.append(local_max_val * weight)
                except:
                    continue

            return max(scores) if scores else None

        except Exception as e:
            if self.debug:
                print(f"[{code}] Template match error: {e}")
            return None

    def feature_match(self, square_img, template, code):
        try:
            template_resized = cv2.resize(template, (square_img.shape[1], square_img.shape[0]))
            if len(template_resized.shape) == 3:
                template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template_resized.copy()

            orb = cv2.ORB_create(nfeatures=100)

            kp1, des1 = orb.detectAndCompute(square_img, None)
            kp2, des2 = orb.detectAndCompute(template_gray, None)

            if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
                return None

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            good_matches = [m for m in matches if m.distance < 50]
            if len(matches) > 0:
                score = len(good_matches) / len(matches)
                return min(score, 1.0)
            return None

        except Exception as e:
            if self.debug:
                print(f"[{code}] Feature match error: {e}")
            return None

    def edge_match(self, square_img, template, code):
        """Edge-based matching: Canny-edges of a filled Lichess piece produce
        an outline that matches the outline-style templates directly."""
        try:
            template_resized = cv2.resize(template, (square_img.shape[1], square_img.shape[0]))
            if len(template_resized.shape) == 3:
                template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template_resized.copy()

            square_edges = cv2.Canny(square_img, 30, 100).astype(np.float32)
            template_edges = cv2.Canny(template_gray, 30, 100).astype(np.float32)

            # Dilate slightly to tolerate minor positional/scale differences
            k = np.ones((3, 3), np.uint8)
            square_edges = cv2.dilate(square_edges, k, iterations=1)
            template_edges = cv2.dilate(template_edges, k, iterations=1)

            res = cv2.matchTemplate(square_edges, template_edges, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if np.isfinite(max_val) and 0 <= max_val <= 1:
                return max_val
            return None

        except Exception as e:
            if self.debug:
                print(f"[{code}] Edge match error: {e}")
            return None

    def histogram_match(self, square_img, template, code):
        """Histogram-based matching"""
        try:
            template_resized = cv2.resize(template, (square_img.shape[1], square_img.shape[0]))
            if len(template_resized.shape) == 3:
                template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template_resized.copy()

            hist1 = cv2.calcHist([square_img], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([template_gray], [0], None, [256], [0, 256])

            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            if np.isfinite(correlation) and correlation > 0:
                return correlation
            return None

        except Exception as e:
            if self.debug:
                print(f"[{code}] Histogram match error: {e}")
            return None

    def detect_orientation(self, board_img, square_size, border_size):
        """Detect if the board image is from Black's perspective.

        Lichess renders rank labels (1-8) as small text in the top-left corner of
        h-file squares. From White's perspective h-file = col=7 (right); from
        Black's perspective h-file = col=0 (left). We compare how many label-like
        high-contrast pixels appear in the left column vs the right column.
        Returns True if the board is from Black's perspective (needs to be flipped).
        """
        score_left = 0.0
        score_right = 0.0
        label_h = max(2, square_size // 8)
        label_w = max(3, square_size // 6)

        for row in range(8):
            for col_idx, is_left in [(0, True), (7, False)]:
                y1 = border_size + row * square_size
                x1 = border_size + col_idx * square_size
                sq = board_img[y1:y1+square_size, x1:x1+square_size]
                if sq.size == 0:
                    continue
                gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY) if len(sq.shape) == 3 else sq.copy()

                # Check top-left corner region where Lichess renders rank labels
                label_region = gray[2:2+label_h, 2:2+label_w]
                parity = (row + col_idx) % 2
                bg = self.light_bg if parity == 0 else self.dark_bg
                if bg is None:
                    continue

                contrast = np.abs(label_region.astype(float) - bg)
                label_score = float(np.mean(contrast > 25))
                if is_left:
                    score_left += label_score
                else:
                    score_right += label_score

        is_flipped = score_left > score_right
        if self.debug:
            print(f"Orientation: label_score_left={score_left:.2f}, label_score_right={score_right:.2f} -> {'Black' if is_flipped else 'White'}'s perspective")
        return is_flipped

    def normalize_gray(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        return clahe.apply(gray)
