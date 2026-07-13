"""
Plate_Recognition.py
--------------------
Main pipeline for vehicle license plate recognition.
Orchestrates the following modules:
  - detector   : ONNX inference (bounding box detection)
  - ocr_engine : image preprocessing + EasyOCR + text correction
  - tracker    : temporal tracking with coasting and ghost tracks
  - utils      : image utilities and GUI helpers
"""

import os
import cv2
import numpy as np
import onnxruntime as ort

from detector   import PlateDetector
from ocr_engine import OCREngine
from tracker    import PlateTracker
from utils      import can_show_windows, make_high_quality_crop


# -----------------------------------------------------------------------
# Side-panel layout constants
# -----------------------------------------------------------------------
PANEL_W    = 560   # total width of the right-side plate panel (pixels)
PANEL_COLS = 2     # number of plate columns in the panel
TILE_H     = 160   # fixed height of each plate tile (label + crop)
LABEL_H    = 30    # height of the text label band at the top of each tile
TITLE_H    = 72    # height of the top title bar (title + subtitle rows)
TITLE_TEXT = "License Plate Recognition System"


def build_title_bar(width, num_plates=0, avg_fps=0.0,
                    title=TITLE_TEXT, bar_h=TITLE_H):
    """Midnight-blue header bar with large centered title and live stats subtitle."""
    # Background: midnight blue  #191970 → BGR(112, 25, 25)
    bar = np.full((bar_h, width, 3), 0, dtype=np.uint8)
    bar[:, :] = (112, 25, 25)   # midnight blue

    # Subtle gradient: slightly lighter at top
    for row in range(bar_h):
        alpha = 1.0 - row / bar_h * 0.35
        bar[row, :] = np.clip(
            np.array([112, 25, 25]) * alpha, 0, 255
        ).astype(np.uint8)

    # Accent line at bottom: bright electric blue  BGR(235, 130, 50)
    bar[bar_h - 3 : bar_h, :] = (235, 130, 50)

    # --- Main title ---
    t_font  = cv2.FONT_HERSHEY_DUPLEX
    t_scale = 1.0
    t_thick = 2
    (tw, th), _ = cv2.getTextSize(title, t_font, t_scale, t_thick)
    tx = (width - tw) // 2
    ty = 34
    # Soft white-blue  RGB(210,220,255) → BGR(255,220,210)
    cv2.putText(bar, title, (tx, ty), t_font, t_scale,
                (255, 220, 210), t_thick, cv2.LINE_AA)

    # --- Subtitle: detected plates + avg FPS ---
    subtitle   = f"Detected Plates: {num_plates}     Avg FPS: {avg_fps:.1f}"
    s_font     = cv2.FONT_HERSHEY_SIMPLEX
    s_scale    = 0.52
    s_thick    = 1
    (sw, sh), _ = cv2.getTextSize(subtitle, s_font, s_scale, s_thick)
    sx = (width - sw) // 2
    sy = ty + sh + 12
    # Periwinkle blue  RGB(150,170,230) → BGR(230,170,150)
    cv2.putText(bar, subtitle, (sx, sy), s_font, s_scale,
                (230, 170, 150), s_thick, cv2.LINE_AA)

    return bar


def build_side_panel(crops, labels, panel_w, panel_h,
                     cols=PANEL_COLS, tile_h=TILE_H, label_h=LABEL_H):
    """
    Build a grid of fixed-size plate tiles (left→right, then down).

    Layout per tile:
      ┌──────────────────┐  ← label_h px  (dark band + plate text)
      │  PLATE TEXT      │
      ├──────────────────┤
      │                  │  ← (tile_h - label_h) px  (plate crop)
      │   [crop image]   │
      └──────────────────┘
    Tiles fill columns first, then new rows.
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

        # --- Label band ---
        band = np.full((label_h, tile_w, 3), 52, dtype=np.uint8)
        cv2.putText(band, label, (6, label_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        panel[y0 : y0 + label_h, x0 : x0 + tile_w] = band

        # --- Crop image ---
        crop_area_h  = tile_h - label_h
        crop_resized = cv2.resize(crop, (tile_w, crop_area_h))
        panel[y0 + label_h : y0 + tile_h, x0 : x0 + tile_w] = crop_resized

        # --- Horizontal separator ---
        sep_y = y0 + tile_h - 1
        if sep_y < panel_h:
            panel[sep_y, x0 : x0 + tile_w] = [60, 60, 60]

    # Vertical dividers between columns
    for c in range(1, cols):
        panel[:, c * tile_w - 1 : c * tile_w + 1] = [50, 50, 50]

    return panel


# -----------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------
def main(video_path, model_path, output_path=None, display=True,
         crop_top_ratio=0.25, save_crops=False, crops_dir="crops",
         target_height=640):
    """
    target_height : height (px) of the video pane in the composite window.
                    out_width is computed automatically from the native aspect
                    ratio so PANEL_W (plate column) is always exactly PANEL_W px
                    regardless of the input resolution.
    """

    # --- ONNX Detector ---
    detector = PlateDetector(model_path)
    print(f"Using device: {detector.device_label}")

    # Create output folder if it does not exist
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # --- GUI ---
    gui_ok = can_show_windows()
    if display and not gui_ok:
        print("Warning: OpenCV does not support GUI. 'display' disabled.")
        display = False

    # --- OCR ---
    use_gpu_ocr = "CUDAExecutionProvider" in ort.get_available_providers()
    ocr = OCREngine(use_gpu=use_gpu_ocr)

    # --- Tracker with coasting ---
    tracker = PlateTracker(
        history_len=12,
        dist_threshold=100,
        min_votes_to_confirm=3,    # votes required to lock in confirmed_text
        max_missed_frames=45,      # ~1.5s at 30fps of coasting before expiry
        ghost_min_ratio=0.25,
    )

    if save_crops:
        os.makedirs(crops_dir, exist_ok=True)

    # --- Video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps         = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Resolution: {width}x{height} | FPS: {fps} | Frames: {frame_count}")

    out        = None
    # Video pane: scale to target_height, preserve native aspect ratio
    out_height = target_height
    out_width  = int(width * out_height / height)
    # PANEL_W is always fixed (plate column never changes size)
    vsep_w     = 3
    comp_w     = out_width + vsep_w + PANEL_W  # adapts to video width
    comp_h     = TITLE_H + out_height
    print(f"Display: video={out_width}x{out_height} | panel={PANEL_W}px | total={comp_w}x{comp_h}")
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (comp_w, comp_h))

    frame_idx        = 0
    cleanup_interval = max(1, fps * 3)   # clean up expired tracks every 3s

    # FPS rolling average
    import time
    _fps_times   = []
    _fps_window  = 30   # frames to average over
    _avg_fps     = 0.0
    _t_last      = time.perf_counter()

    # ===================================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # --- FPS measurement ---
        _t_now   = time.perf_counter()
        _fps_times.append(_t_now - _t_last)
        _t_last  = _t_now
        if len(_fps_times) > _fps_window:
            _fps_times.pop(0)
        _avg_fps = 1.0 / (sum(_fps_times) / len(_fps_times)) if _fps_times else 0.0

        # Crop region of interest (lower portion of the frame)
        crop_y        = int(height * crop_top_ratio)
        frame_cropped = frame[crop_y:, :].copy()

        # ---------------------------------------------------------------
        # 1) ONNX DETECTION
        # ---------------------------------------------------------------
        boxes, scores = detector.detect(frame_cropped)

        # Centroids of this frame for the tracker
        active_centers = [
            ((int(b[0]) + int(b[2])) // 2, (int(b[1]) + int(b[3])) // 2)
            for b in boxes
        ]
        tracker.tick_all(active_centers)   # mark unseen tracks

        crops      = []
        labels     = []

        # ---------------------------------------------------------------
        # 2) OCR + TRACKING per each detection
        # ---------------------------------------------------------------
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            crop_raw, (nx1, ny1, nx2, ny2) = make_high_quality_crop(
                frame_cropped, x1, y1, x2, y2, max_margin_px=20
            )

            # OCR
            text, conf_ocr, valid_ocr = ocr.read(crop_raw)

            # Tracker (majority vote + coasting)
            cx = (nx1 + nx2) // 2
            cy = (ny1 + ny2) // 2
            display_text, is_confident = tracker.update(
                cx, cy, text, box=(nx1, ny1, nx2, ny2)
            )
            confident = is_confident or bool(valid_ocr)

            crops.append(crop_raw)
            labels.append(display_text)

            # Draw active detection
            color = (0, 255, 0) if confident else (0, 165, 255)
            cv2.rectangle(frame, (nx1, ny1 + crop_y), (nx2, ny2 + crop_y), color, 2)
            cv2.putText(frame, display_text, (nx1, ny1 + crop_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # ---------------------------------------------------------------
        # 3) GHOST TRACKS (coasting)
        # ---------------------------------------------------------------
        for g in tracker.get_ghost_tracks():
            missed      = g["missed_frames"]
            fade        = max(0.15, 1.0 - missed / tracker.max_missed_frames)
            ghost_color = (0, int(165 * fade), int(255 * fade))
            gt          = g["confirmed_text"]
            gbox        = g["best_box"]

            if gbox is not None:
                gx1, gy1, gx2, gy2 = gbox
                cv2.rectangle(frame,
                              (gx1, gy1 + crop_y), (gx2, gy2 + crop_y),
                              ghost_color, 1)
                cv2.putText(frame, f"[{gt}]",
                            (gx1, gy1 + crop_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, ghost_color, 2, cv2.LINE_AA)
            else:
                gcx, gcy = g["center"]
                cv2.putText(frame, f"[{gt}]",
                            (gcx - 40, gcy + crop_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, ghost_color, 2, cv2.LINE_AA)

        # ---------------------------------------------------------------
        # 4) MAINTENANCE
        # ---------------------------------------------------------------
        if frame_idx % cleanup_interval == 0:
            tracker.cleanup_old_tracks()

        # ---------------------------------------------------------------
        # 5) OUTPUT / DISPLAY  — unified side-by-side composite
        # ---------------------------------------------------------------

        # Scale main frame to output resolution
        frame_scaled = cv2.resize(frame, (out_width, out_height))

        # Build right-side plate panel (same height as video)
        side_panel = build_side_panel(crops, labels,
                                      panel_w=PANEL_W, panel_h=out_height)

        # Thin vertical separator bar
        vsep = np.full((out_height, vsep_w, 3), 40, dtype=np.uint8)

        # Row: [video | separator | plate panel]
        body = np.hstack([frame_scaled, vsep, side_panel])

        # Title bar spanning full width, with live stats
        title_bar = build_title_bar(comp_w,
                                    num_plates=len(crops),
                                    avg_fps=_avg_fps)

        # Final composite: title on top, body below
        composite = np.vstack([title_bar, body])

        if display and gui_ok:
            try:
                cv2.imshow("Plates - Annotated Video", composite)
            except cv2.error:
                print("Error displaying windows. Disabling display.")
                display = False
                gui_ok  = False

        if out:
            out.write(composite)

        if display and gui_ok:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # ===================================================================

    cap.release()
    if out:
        out.release()
    if display and gui_ok:
        cv2.destroyAllWindows()
    print("Processing completed successfully.")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    video_path  = "resources/Tránsito en Asunción.mp4"
    model_path  = "resources/chapitas.onnx"
    output_path = "output/salididita.mp4"
    main(video_path, model_path, output_path,
         display=True, crop_top_ratio=0.5, save_crops=False,
         target_height=640)
