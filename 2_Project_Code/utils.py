"""
utils.py
--------
Image and GUI utilities shared by the plate recognition pipeline.
"""

import cv2
import numpy as np


# -----------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------
def can_show_windows():
    """Checks whether the environment supports OpenCV windows."""
    try:
        test = np.zeros((2, 2, 3), dtype=np.uint8)
        cv2.namedWindow(".__test__", cv2.WINDOW_NORMAL)
        cv2.imshow(".__test__", test)
        cv2.waitKey(1)
        cv2.destroyWindow(".__test__")
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------
# Resize and padding
# -----------------------------------------------------------------------
def smart_resize_and_pad(img, target_size):
    """
    Resizes img to fit within target_size=(H, W) while preserving
    aspect ratio, filling the remaining space with white.
    """
    th, tw = target_size
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((th, tw, 3), dtype=np.uint8)
    scale = min(tw / w, th / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    canvas = np.full((th, tw, 3), 255, dtype=np.uint8)
    x_off = (tw - new_w) // 2
    y_off = (th - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


# -----------------------------------------------------------------------
# Visual enhancements
# -----------------------------------------------------------------------
def unsharp_mask(img, kernel_size=(3, 3), amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(img, kernel_size, 0)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    if threshold > 0:
        mask = np.absolute(img.astype(int) - blurred.astype(int)) < threshold
        np.copyto(sharpened, img, where=mask)
    return sharpened


def enhance_for_display(img, display_w=320):
    """Enhances contrast and sharpness of a crop for on-screen display."""
    if img is None or img.size == 0:
        return np.zeros((display_w // 4, display_w, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(display_w / float(w), (display_w // 2) / float(h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    small = cv2.resize(img, (new_w, new_h), interpolation=interp)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab2 = cv2.merge((clahe.apply(l), a, b))
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return unsharp_mask(enhanced, amount=0.6)


# -----------------------------------------------------------------------
# Gallery / mosaic of crops
# -----------------------------------------------------------------------
def create_gallery(crops, labels, crop_display_size=(200, 100)):
    """
    Creates a mosaic image with the detected plate crops
    and their OCR labels above each tile.
    """
    if not crops:
        th, tw = crop_display_size[1], crop_display_size[0]
        return np.zeros((th + 30, tw, 3), dtype=np.uint8)

    tiles = []
    for i, c in enumerate(crops):
        display = enhance_for_display(c, display_w=crop_display_size[0])
        padded = smart_resize_and_pad(display, (crop_display_size[1], crop_display_size[0]))
        band = np.full((30, crop_display_size[0], 3), 255, dtype=np.uint8)
        label = labels[i] if i < len(labels) else ""
        cv2.putText(band, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(np.vstack([band, padded]))

    return np.hstack(tiles)


# -----------------------------------------------------------------------
# High-quality crop with margin (original logic)
# -----------------------------------------------------------------------
def make_high_quality_crop(frame_cropped, x1, y1, x2, y2,
                           margin_ratio=0.12, max_margin_px=40):
    """
    Crops the plate region with an optional lateral margin and a vertical
    10% inward trim (negative margin) that adjusts the crop to the useful
    plate area.
    Returns: (crop_img, (nx1, ny1, nx2, ny2))
    """
    h, w = frame_cropped.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    mx = int(min(max_margin_px, bw * margin_ratio))
    my = int(min(max_margin_px, bh * -0.1))   # negative: trims ~10% top and bottom
    nx1 = max(0, x1 - mx)
    ny1 = max(0, y1 - my)
    nx2 = min(w, x2 + mx)
    ny2 = min(h, y2 + my)
    crop = frame_cropped[ny1:ny2, nx1:nx2].copy()
    return crop, (nx1, ny1, nx2, ny2)
