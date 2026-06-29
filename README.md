# Realtime Car Plate Recognition

A complete end-to-end pipeline for **real-time vehicle license plate detection and OCR** using YOLOv8 (exported to ONNX) and EasyOCR, with a temporal tracking system featuring ghost tracks and majority voting.

---

## Project Structure

```
car_plates_recognition/
│
├── 0_Dataset/
│   └── License_Plate_Recognition_yolov8.zip   # Raw Roboflow dataset (~515 MB)
│
├── 1_Notebook_for_train_the_model/
│   ├── Yolo_Trainer.ipynb                      # Full training + hyperparameter search notebook
│   └── yolo_onnx_convertion.py                 # Exports trained .pt model to ONNX format
│
└── 2_Project_Code/
    ├── Plate_Recognition.py                    # Main pipeline entry point
    ├── detector.py                             # ONNX-based plate detector (YOLOv8)
    ├── ocr_engine.py                           # EasyOCR + preprocessing + text correction
    ├── tracker.py                              # Temporal tracker with coasting / ghost tracks
    ├── utils.py                                # Image & GUI utilities
    └── resources/
        ├── chapitas.onnx                       # Exported ONNX model (~42 MB)
        ├── modelito_chapitas.pt                # Trained YOLOv8 weights (~21 MB)
        └── Tránsito en Asunción.mp4            # Demo video (~14 MB)
```

---

## Features

- **YOLOv8 ONNX Detection** — Fast inference via ONNX Runtime (CPU or CUDA), no PyTorch required at runtime.
- **Multi-variant OCR** — Each plate crop is preprocessed into 4 variants (CLAHE, adaptive threshold, Otsu, fusion) and fed to EasyOCR with weighted voting.
- **Perspective rectification** — `minAreaRect` corrects tilted plate crops before OCR.
- **Temporal tracking with coasting** — Plates keep their last confirmed text on screen for up to ~1.5 s after disappearing (ghost tracks with progressive fade-out).
- **Majority vote confirmation** — Text is only "confirmed" once it appears consistently across a rolling OCR window, reducing false reads.
- **LLLLNNN format correction** — Position-aware post-processing maps OCR errors to the expected 4-letter + 3-digit plate format.
- **Hyperparameter optimization** — Optuna-based search over 15 YOLO training hyperparameters (lr, momentum, augmentations, loss weights, …).

---

## Architecture

```
Video Frame
    │
    ▼
┌──────────────┐
│  PlateDetector│  detector.py — letterbox → ONNX → NMS → bounding boxes
└──────┬───────┘
       │  crops
       ▼
┌──────────────┐
│  OCREngine   │  ocr_engine.py — rectify → 4 variants → EasyOCR → vote → fix_plate
└──────┬───────┘
       │  text + confidence
       ▼
┌──────────────┐
│  PlateTracker│  tracker.py — centroid matching → history deque → majority vote → ghost tracks
└──────┬───────┘
       │  display_text + is_confident
       ▼
  Annotated Frame  (OpenCV overlay + gallery mosaic)
```

---

## Installation

### Requirements

```bash
pip install ultralytics roboflow optuna
pip install easyocr opencv-python onnxruntime
```

> **GPU support:** For CUDA inference install the GPU builds:
> ```bash
> pip install onnxruntime-gpu
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Usage

### Run the recognition pipeline

```bash
cd 2_Project_Code
python Plate_Recognition.py
```

The entry point is configured at the bottom of [Plate_Recognition.py](2_Project_Code/Plate_Recognition.py):

```python
video_path  = "resources/Tránsito en Asunción.mp4"
model_path  = "resources/chapitas.onnx"
output_path = "output/salididita.mp4"

main(video_path, model_path, output_path,
     display=True,
     crop_top_ratio=0.5,   # ignore the top 50% of the frame (sky / buildings)
     save_crops=False)
```

| Parameter | Description |
|---|---|
| `display` | Show live OpenCV windows |
| `crop_top_ratio` | Fraction of the frame height to skip from the top (reduces false detections) |
| `save_crops` | Save individual plate crop images to disk |

---

## 🏋️ Training

Open [Yolo_Trainer.ipynb](1_Notebook_for_train_the_model/Yolo_Trainer.ipynb) in Jupyter / VS Code.

The notebook covers:

1. **Library installation** — `ultralytics`, `optuna`, PyTorch with CUDA.
2. **GPU verification** — `nvidia-smi` + PyTorch CUDA diagnostic.
3. **Dataset unzip & inspection** — unpacks the Roboflow dataset and prints split statistics.
4. **Hyperparameter search** — Optuna study over 15 parameters, 10 trials, maximising `mAP50`.
5. **Final training** — 270 epochs with the best hyperparameters found.
6. **Metrics visualisation** — loss curves and precision/recall/mAP plots.

### Dataset

| Split | Images | % |
|---|---|---|
| Train | 7 057 | 69.70 % |
| Valid | 2 048 | 20.23 % |
| Test  | 1 020 | 10.07 % |

Source: [Roboflow — License Plate Recognition v11](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/11) (CC BY 4.0)

### Export to ONNX

After training, run [yolo_onnx_convertion.py](1_Notebook_for_train_the_model/yolo_onnx_convertion.py):

```bash
python yolo_onnx_convertion.py
```

Edit `MODEL_PATH` to point to your `.pt` checkpoint. The script exports with `imgsz=224`, `opset=12`, and `simplify=False` (required for correct ONNX Runtime decoding).

---

## Module Reference

| File | Class / Key functions | Role |
|---|---|---|
| `detector.py` | `PlateDetector` | Letterbox preprocessing, ONNX inference, NMS, coordinate de-padding |
| `ocr_engine.py` | `OCREngine`, `perform_ocr_variants`, `fix_plate_by_position` | 4-variant OCR, weighted voting, LLLLNNN correction |
| `tracker.py` | `PlateTracker` | Centroid-based association, majority vote, ghost tracks with fade-out |
| `utils.py` | `make_high_quality_crop`, `create_gallery`, `enhance_for_display` | Crop extraction, display gallery mosaic, CLAHE sharpening |

---

## License

Dataset: **CC BY 4.0** (Roboflow Universe).  
Code: feel free to use and adapt.
