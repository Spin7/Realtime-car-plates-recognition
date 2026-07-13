"""
Plate_Recognition_test.py
--------------------------
Single-image inference pipeline for license plate recognition.

Same visual output as Plate_Recognition.py but:
  - Input  : one image file (instead of a video)
  - Output : saved result image (instead of a video file)
  - Stats  : total pipeline inference time in ms (instead of avg FPS)

Usage (entry point at bottom):
    python Plate_Recognition_test.py
"""

import os
import time
import cv2
import numpy as np
import onnxruntime as ort

from detector   import PlateDetector
from ocr_engine import OCREngine
from utils      import can_show_windows, make_high_quality_crop


# -----------------------------------------------------------------------
# Layout constants  (kept identical to Plate_Recognition.py)
# -----------------------------------------------------------------------
PANEL_W    = 560   # fixed width of the right-side plate panel (pixels)
PANEL_COLS = 2     # number of plate columns in the panel
TILE_H     = 160   # fixed height of each plate tile (label + crop)
LABEL_H    = 30    # height of the text label band at the top of each tile
TITLE_H    = 72    # height of the top title bar (title + subtitle rows)
TITLE_TEXT = "License Plate Recognition System"


# -----------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------

def build_title_bar(width, num_plates=0, inference_ms=0.0,
                    title=TITLE_TEXT, bar_h=TITLE_H):
    """
    Midnight-blue header bar.
    Subtitle shows detected plate count and total pipeline inference time (ms).
    """
    bar = np.full((bar_h, width, 3), 0, dtype=np.uint8)
    bar[:, :] = (112, 25, 25)   # midnight blue base

    # Subtle top-to-bottom gradient
    for row in range(bar_h):
        alpha = 1.0 - row / bar_h * 0.35
        bar[row, :] = np.clip(
            np.array([112, 25, 25]) * alpha, 0, 255
        ).astype(np.uint8)

    # Accent line at bottom: bright electric blue BGR(235, 130, 50)
    bar[bar_h - 3 : bar_h, :] = (235, 130, 50)

    # --- Main title ---
    t_font  = cv2.FONT_HERSHEY_DUPLEX
    t_scale = 1.0
    t_thick = 2
    (tw, th), _ = cv2.getTextSize(title, t_font, t_scale, t_thick)
    tx = (width - tw) // 2
    ty = 34
    cv2.putText(bar, title, (tx, ty), t_font, t_scale,
                (255, 220, 210), t_thick, cv2.LINE_AA)

    # --- Subtitle: plate count + inference time ---
    subtitle   = f"Detected Plates: {num_plates}     Inference Time: {inference_ms:.1f} ms"
    s_font     = cv2.FONT_HERSHEY_SIMPLEX
    s_scale    = 0.52
    s_thick    = 1
    (sw, sh), _ = cv2.getTextSize(subtitle, s_font, s_scale, s_thick)
    sx = (width - sw) // 2
    sy = ty + sh + 12
    cv2.putText(bar, subtitle, (sx, sy), s_font, s_scale,
                (230, 170, 150), s_thick, cv2.LINE_AA)

    return bar


def build_side_panel(crops, labels, panel_w, panel_h,
                     cols=PANEL_COLS, tile_h=TILE_H, label_h=LABEL_H):
    """
    Grid of fixed-size plate tiles (left→right, then down).
    Each tile: label band on top + crop image below.
    """
    tile_w = panel_w // cols
    panel  = np.full((panel_h, panel_w, 3), 28, dtype=np.uint8)   # dark bg

    for i, (crop, label) in enumerate(zip(crops, labels)):
        row = i // cols
        col = i % cols
        x0  = col * tile_w
        y0  = row * tile_h

        if y0 + tile_h > panel_h:
            break   # no room for more rows

        # Label band
        band = np.full((label_h, tile_w, 3), 52, dtype=np.uint8)
        cv2.putText(band, label, (6, label_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        panel[y0 : y0 + label_h, x0 : x0 + tile_w] = band

        # Crop image
        crop_area_h  = tile_h - label_h
        crop_resized = cv2.resize(crop, (tile_w, crop_area_h))
        panel[y0 + label_h : y0 + tile_h, x0 : x0 + tile_w] = crop_resized

        # Horizontal separator
        sep_y = y0 + tile_h - 1
        if sep_y < panel_h:
            panel[sep_y, x0 : x0 + tile_w] = [60, 60, 60]

    # Vertical dividers between columns
    for c in range(1, cols):
        panel[:, c * tile_w - 1 : c * tile_w + 1] = [50, 50, 50]

    return panel


# -----------------------------------------------------------------------
# Single-image pipeline
# -----------------------------------------------------------------------

def run_image(image_path, model_path, output_path=None, display=True,
              crop_top_ratio=0.25, target_height=640):
    """
    Process a single image through the full detection + OCR pipeline.

    Parameters
    ----------
    image_path      : path to the input image
    model_path      : path to the ONNX detection model
    output_path     : if given, saves the composite result image here
    display         : whether to show the result in an OpenCV window
    crop_top_ratio  : fraction of the top of the image to ignore
    target_height   : height (px) of the video pane; out_width is computed
                      automatically from the native aspect ratio so PANEL_W
                      (plate column) is always exactly PANEL_W px.
    """

    # --- Models ---
    detector = PlateDetector(model_path)
    print(f"Using device: {detector.device_label}")

    use_gpu_ocr = "CUDAExecutionProvider" in ort.get_available_providers()
    ocr = OCREngine(use_gpu=use_gpu_ocr)

    # --- Load image ---
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width = frame.shape[:2]
    print(f"Image resolution: {width}x{height}")

    # --- Output folder ---
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # --- Composite dimensions (PANEL_W always fixed) ---
    out_height = target_height
    out_width  = int(width * out_height / height)
    vsep_w     = 3
    comp_w     = out_width + vsep_w + PANEL_W
    comp_h     = TITLE_H + out_height
    print(f"Display: video={out_width}x{out_height} | panel={PANEL_W}px | total={comp_w}x{comp_h}")

    # ===================================================================
    # PIPELINE  (timed)
    # ===================================================================
    t_start = time.perf_counter()

    # 1) Crop top region of image
    crop_y        = int(height * crop_top_ratio)
    frame_cropped = frame[crop_y:, :].copy()

    # 2) ONNX detection
    boxes, scores = detector.detect(frame_cropped)

    crops  = []
    labels = []

    # 3) OCR per detection
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        crop_raw, (nx1, ny1, nx2, ny2) = make_high_quality_crop(
            frame_cropped, x1, y1, x2, y2, max_margin_px=20
        )

        text, conf_ocr, valid_ocr = ocr.read(crop_raw)
        display_text = text if text else "???"

        crops.append(crop_raw)
        labels.append(display_text)

        # Draw bounding box + label on the original frame
        color = (0, 255, 0) if valid_ocr else (0, 165, 255)
        cv2.rectangle(frame, (nx1, ny1 + crop_y), (nx2, ny2 + crop_y), color, 2)
        cv2.putText(frame, display_text, (nx1, ny1 + crop_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    t_end = time.perf_counter()
    inference_ms = (t_end - t_start) * 1000.0
    print(f"Pipeline inference time: {inference_ms:.1f} ms | Plates found: {len(crops)}")

    # ===================================================================
    # BUILD COMPOSITE
    # ===================================================================

    # Scale annotated frame to display size
    frame_scaled = cv2.resize(frame, (out_width, out_height))

    # Plate panel (same height as video pane)
    side_panel = build_side_panel(crops, labels,
                                  panel_w=PANEL_W, panel_h=out_height)

    # Vertical separator
    vsep = np.full((out_height, vsep_w, 3), 40, dtype=np.uint8)

    # Row: [video | separator | plate panel]
    body = np.hstack([frame_scaled, vsep, side_panel])

    # Title bar with inference stats
    title_bar = build_title_bar(comp_w,
                                num_plates=len(crops),
                                inference_ms=inference_ms)

    # Final composite
    composite = np.vstack([title_bar, body])

    # --- Save ---
    if output_path:
        cv2.imwrite(output_path, composite)
        print(f"Result saved to: {output_path}")

    # --- Display ---
    gui_ok = can_show_windows()
    if display and gui_ok:
        cv2.imshow("Plates - Test Image", composite)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif display and not gui_ok:
        print("Warning: OpenCV GUI not available. Display skipped.")

    return composite


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    image_path  = "resources/test_image.jpg"   # ← change to your image
    model_path  = "resources/chapitas.onnx"
    output_path = "output/test_result.jpg"

    run_image(image_path, model_path, output_path,
              display=True, crop_top_ratio=0.5, target_height=640)
