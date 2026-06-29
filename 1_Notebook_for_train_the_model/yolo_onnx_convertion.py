import os
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = "chapitas.pt"       # your trained model
ONNX_DIR   = "onnx_export"   # output folder
IMGSZ      = 224

# =========================
# LOAD MODEL
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print("[INFO] Loading model...")
model = YOLO(MODEL_PATH)

# =========================
# EXPORT TO ONNX
# =========================
print("[INFO] Exporting to ONNX...")

export_path = model.export(
    format="onnx",
    imgsz=IMGSZ,
    dynamic=False,
    simplify=False,   # critical 
    opset=12
)

print(f"[SUCCESS] ONNX model saved at: {export_path}")