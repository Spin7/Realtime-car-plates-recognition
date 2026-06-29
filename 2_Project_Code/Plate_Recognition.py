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
import onnxruntime as ort

from detector   import PlateDetector
from ocr_engine import OCREngine
from tracker    import PlateTracker
from utils      import can_show_windows, make_high_quality_crop, create_gallery


# -----------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------
def main(video_path, model_path, output_path=None, display=True,
         crop_top_ratio=0.25, save_crops=False, crops_dir="crops"):

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

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx        = 0
    cleanup_interval = max(1, fps * 3)   # clean up expired tracks every 3s

    # ===================================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

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
        # 5) OUTPUT / DISPLAY
        # ---------------------------------------------------------------
        gallery = create_gallery(crops, labels, crop_display_size=(220, 120))

        if display and gui_ok:
            try:
                cv2.imshow("Plates - Annotated Video", frame)
                cv2.imshow("Detected Plates", gallery)
            except cv2.error:
                print("Error displaying windows. Disabling display.")
                display = False
                gui_ok  = False

        if out:
            out.write(frame)

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
         display=True, crop_top_ratio=0.5, save_crops=False)
