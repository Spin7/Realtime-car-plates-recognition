"""
detector.py
-----------
ONNX inference for license plate detection.
Encapsulates all pre/post-processing for a YOLO model exported to ONNX.
"""

import cv2
import numpy as np
import onnxruntime as ort


# -----------------------------------------------------------------------
# Default configuration for chapitas.onnx
# -----------------------------------------------------------------------
DEFAULT_IMGSZ = 224   # actual shape: [1, 3, 224, 224]
DEFAULT_CONF  = 0.25
DEFAULT_IOU   = 0.45


# -----------------------------------------------------------------------
# Low-level functions
# -----------------------------------------------------------------------
def letterbox(image, new_size=DEFAULT_IMGSZ, color=(114, 114, 114)):
    """Resizes the image while preserving aspect ratio, padding with a solid color."""
    h, w = image.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh))
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    pad_x = (new_size - nw) // 2
    pad_y = (new_size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def preprocess(frame_bgr, imgsz=DEFAULT_IMGSZ):
    """
    Converts a BGR frame to a normalized float32 NCHW tensor [0, 1].
    Returns: (tensor, scale, pad_x, pad_y)
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img, scale, pad_x, pad_y = letterbox(rgb, new_size=imgsz)
    tensor = img.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]   # [1, 3, H, W]
    return tensor, scale, pad_x, pad_y


def nms(boxes, scores, iou_threshold=DEFAULT_IOU):
    """Non-Maximum Suppression implemented in pure NumPy."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def decode_output(raw_output, frame_h, frame_w, scale, pad_x, pad_y,
                  conf_thresh=DEFAULT_CONF, iou_thresh=DEFAULT_IOU,
                  imgsz=DEFAULT_IMGSZ):
    """
    Decodes the ONNX output with shape [1, 5, N].
    Columns: [cx, cy, w, h, conf] in pixels of the letterboxed space.
    Returns: (boxes [N,4] float32, scores [N] float32) in original frame coords.
    """
    out = raw_output[0]          # [1, 5, N]
    if out.ndim == 3:
        out = out[0]             # remove batch dim → [5, N]
    if out.shape[0] == 5:        # [5, N] → [N, 5]
        out = out.T

    cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    scores = out[:, 4].astype(np.float32)

    # cx/cy/w/h in letterboxed pixels → original frame coordinates
    x1 = (cx - w / 2 - pad_x) / scale
    y1 = (cy - h / 2 - pad_y) / scale
    x2 = (cx + w / 2 - pad_x) / scale
    y2 = (cy + h / 2 - pad_y) / scale

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_h)

    mask = scores > conf_thresh
    boxes, scores = boxes[mask], scores[mask]
    if len(boxes) == 0:
        return boxes, scores

    keep = nms(boxes, scores, iou_thresh)
    boxes, scores = boxes[keep], scores[keep]
    order = scores.argsort()[::-1]
    return boxes[order], scores[order]


# -----------------------------------------------------------------------
# High-level class
# -----------------------------------------------------------------------
class PlateDetector:
    """
    License plate detector based on ONNX Runtime.
    Automatically selects GPU (CUDA) if available.

    Usage:
        detector = PlateDetector("chapitas.onnx")
        boxes, scores = detector.detect(frame_bgr)
    """

    def __init__(self, model_path,
                 conf_thresh=DEFAULT_CONF,
                 iou_thresh=DEFAULT_IOU,
                 imgsz=DEFAULT_IMGSZ):
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        self.imgsz       = imgsz

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            self.providers   = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.device_label = "cuda (ONNX)"
        else:
            self.providers   = ["CPUExecutionProvider"]
            self.device_label = "cpu (ONNX)"

        self.session    = ort.InferenceSession(model_path, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[PlateDetector] device={self.device_label} | "
              f"input='{self.input_name}' {self.session.get_inputs()[0].shape}")

    def detect(self, frame_bgr):
        """
        Runs inference on a BGR frame.
        Returns: (boxes np.ndarray[N,4], scores np.ndarray[N])
                  boxes in [x1, y1, x2, y2] format in frame pixels.
        """
        h, w = frame_bgr.shape[:2]
        tensor, scale, pad_x, pad_y = preprocess(frame_bgr, self.imgsz)
        raw = self.session.run(None, {self.input_name: tensor})
        return decode_output(raw, h, w, scale, pad_x, pad_y,
                             self.conf_thresh, self.iou_thresh, self.imgsz)
