"""
ocr_engine.py
-------------
OCR recognition pipeline for vehicle license plates.
Logic based on the original script (perform_ocr_variants + fix_plate_by_position).
"""

import re
import cv2
import numpy as np
import easyocr


ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


# -----------------------------------------------------------------------
# Perspective rectification
# -----------------------------------------------------------------------
def rectify_plate(img):
    """Rotates and crops the plate using minAreaRect. Falls back to original image on failure."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype(np.int32)
        s    = box.sum(axis=1)
        tl   = box[np.argmin(s)]
        br   = box[np.argmax(s)]
        diff = np.diff(box, axis=1).reshape(-1)
        tr   = box[np.argmin(diff)]
        bl   = box[np.argmax(diff)]
        src  = np.array([tl, tr, br, bl], dtype="float32")
        widthA   = np.linalg.norm(br - bl)
        widthB   = np.linalg.norm(tr - tl)
        maxWidth  = max(int(widthA), int(widthB))
        heightA  = np.linalg.norm(tr - br)
        heightB  = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        if maxWidth < 50 or maxHeight < 20:
            return img
        dst = np.array(
            [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
            dtype="float32"
        )
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        if warped.shape[1] < warped.shape[0]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        return warped
    except Exception:
        return img


# -----------------------------------------------------------------------
# Optimized preprocessing for OCR
# -----------------------------------------------------------------------
def prepare_for_ocr_strong(crop_raw, target_w=400):
    """
    Generates 4 image variants to maximize OCR recognition rate.
    Variants:
      0 - gray_clahe_denoised      (local contrast + denoising)
      1 - adaptive_threshold       (adaptive threshold + morphology)
      2 - clean_fusion             (AND between grayscale and binary)
      3 - otsu_dilated             (Otsu + dilation, good for dark plates)
    """
    if crop_raw is None or crop_raw.size == 0:
        return [np.zeros((1, 1), np.uint8)]

    crop_raw = rectify_plate(crop_raw)

    h, w = crop_raw.shape[:2]
    if w < target_w:
        scale = target_w / float(w)
        crop_raw = cv2.resize(crop_raw, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop_raw, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    gray_denoised = cv2.fastNlMeansDenoising(gray_clahe, h=10,
                                              templateWindowSize=7,
                                              searchWindowSize=21)
    # Variant 0: enhanced grayscale
    var0 = gray_denoised

    # Variant 1: adaptive threshold
    blur = cv2.GaussianBlur(gray_denoised, (3, 3), 0)
    th_adapt = cv2.adaptiveThreshold(blur, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th_adapt = cv2.morphologyEx(th_adapt, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    if np.mean(th_adapt == 255) < 0.5:
        th_adapt = cv2.bitwise_not(th_adapt)
    var1 = th_adapt

    # Variant 2: grayscale + binary fusion
    var2 = cv2.bitwise_and(gray_denoised, th_adapt)

    # Variant 3: Otsu + dilation
    _, th_otsu = cv2.threshold(gray_denoised, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th_otsu == 255) < 0.5:
        th_otsu = cv2.bitwise_not(th_otsu)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    th_otsu = cv2.dilate(th_otsu, kernel_dilate, iterations=1)
    var3 = th_otsu

    return [var0, var1, var2, var3]


# -----------------------------------------------------------------------
# Multi-variant OCR with weighted voting
# -----------------------------------------------------------------------
def perform_ocr_variants(reader, crop_raw, allowlist=ALLOWLIST):
    """
    Applies OCR over 4 variants of the plate crop with weighted voting.
      - 4 image variants
      - 2x upscaling before sending to EasyOCR
      - text_threshold=0.6 and contrast_ths=0.4 (more permissive)
      - Weighted voting with ideal-length bonus (7 chars)
    Returns: (winning_text, average_confidence, candidate_dict)
    """
    variants_gray = prepare_for_ocr_strong(crop_raw, target_w=400)

    variants_rgb = []
    for img in variants_gray:
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img
        img_rgb = cv2.resize(img_rgb, (0, 0), fx=2.0, fy=2.0,
                             interpolation=cv2.INTER_CUBIC)
        variants_rgb.append(img_rgb)

    all_results = []
    for var in variants_rgb:
        try:
            res = reader.readtext(
                var,
                detail=1,
                paragraph=False,
                allowlist=allowlist,
                contrast_ths=0.4,
                adjust_contrast=0.6,
                text_threshold=0.6,
                low_text=0.3
            )
            for box, text, conf in res:
                text_clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(text_clean) < 5 or len(text_clean) > 9:
                    continue
                all_results.append((text_clean, float(conf)))
        except Exception:
            continue

    if not all_results:
        return "-", 0.0, {}

    grouped = {}
    for t, c in all_results:
        grouped.setdefault(t, 0.0)
        grouped[t] += c

    def score(item):
        text, acc_conf = item
        length_bonus = 1.0 - abs(len(text) - 7) * 0.1
        return acc_conf * max(0.5, length_bonus)

    best = max(grouped.items(), key=score)
    final_text = best[0]
    final_conf = best[1] / max(1, len(all_results))

    return final_text, final_conf, grouped


# -----------------------------------------------------------------------
# Post-processing: position-based correction (LLLLNNN)
# -----------------------------------------------------------------------
letters_map_letterspos = {
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'
}
digits_map_digitpos = {
    'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8'
}
PLATE_PATTERN = re.compile(r'^[A-Z]{4}\d{3}$')


def normalize_and_clean_raw(s):
    if not s:
        return ""
    s = s.upper()
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s


def _corr_letter_pos(ch):
    if ch.isalpha():
        return ch
    if ch in letters_map_letterspos:
        return letters_map_letterspos[ch]
    return 'X'


def _corr_digit_pos(ch):
    if ch.isdigit():
        return ch
    if ch in digits_map_digitpos:
        return digits_map_digitpos[ch]
    return '0'


def _try_fix_positions(s7):
    if len(s7) != 7:
        if len(s7) > 7:
            s7 = s7[:7]
        else:
            s7 = s7.ljust(7, 'X')
    s7 = s7.upper()
    res_chars = []
    for i, ch in enumerate(s7):
        if i <= 3:
            res_chars.append(_corr_letter_pos(ch))
        else:
            res_chars.append(_corr_digit_pos(ch))
    return ''.join(res_chars)


def fix_plate_by_position(raw):
    """
    Attempts to correct the OCR text to the LLLLNNN format (4 letters + 3 digits).
    Returns: (corrected_text, is_valid: int) -- 1 if it matches the exact pattern.
    """
    if not raw:
        return "-", 0
    s = normalize_and_clean_raw(raw)
    if PLATE_PATTERN.match(s):
        return s, 1
    if len(s) > 7:
        candidates = []
        for i in range(0, len(s) - 6):
            sub = s[i:i + 7]
            cand = _try_fix_positions(sub)
            if PLATE_PATTERN.match(cand):
                return cand, 1
            candidates.append(cand)
        best = max(candidates, key=lambda x: sum(ch.isalpha() for ch in x[:4]))
        return best, 0
    if len(s) < 7:
        padded = s.ljust(7, 'X')
        fixed = _try_fix_positions(padded)
        return fixed, 0
    fixed = _try_fix_positions(s)
    if PLATE_PATTERN.match(fixed):
        return fixed, 1
    return fixed, 0


# -----------------------------------------------------------------------
# High-level class
# -----------------------------------------------------------------------
class OCREngine:
    """
    OCR engine for vehicle license plates.
    Combines EasyOCR + multi-variant preprocessing + positional correction.

    Usage:
        engine = OCREngine(use_gpu=True)
        text, conf, valid = engine.read(crop_bgr)
    """

    def __init__(self, use_gpu=False, allowlist=ALLOWLIST):
        self.allowlist = allowlist
        try:
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        except Exception as e:
            print(f"[OCREngine] Error initializing EasyOCR: {e}")
            self.reader = None

    def read(self, crop_bgr):
        """
        Reads the text from a plate crop.
        Returns: (corrected_text, confidence, is_valid: int)
        """
        if self.reader is None:
            return "-", 0.0, 0
        raw_text, conf, _ = perform_ocr_variants(self.reader, crop_bgr, self.allowlist)
        fixed, valid = fix_plate_by_position(raw_text)
        return fixed, conf, valid
