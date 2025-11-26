from ultralytics import YOLO
import cv2
import numpy as np

CLASS_COLORS = {
    "frontcourt":        (255, 0, 0),
    "midcourt-up":       (255, 255, 0),
    "midcourt-down":     (255, 255, 0),
    "rearcourt-up":      (0, 165, 255),
    "rearcourt-down":    (0, 165, 255),
    "sideline-left":     (0, 255, 255),
    "sideline-right":    (0, 255, 255),
    "baseline":          (203, 192, 255),
}

FALLBACK_PALETTE = [
    (255, 0, 0), (255, 255, 0), (0, 165, 255),
    (0, 255, 255), (180, 105, 255), (147, 20, 255),
    (211, 0, 148), (128, 128, 128)
]

class CourtPartDetector:
    def __init__(self, model_path, conf=0.25):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect_frame(self, frame):
        res = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        ann = []
        names = getattr(self.model, "names", None)
        if res.boxes is not None and len(res.boxes) > 0:
            for i, box in enumerate(res.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                cls  = int(box.cls[0].item())
                name = names[cls] if names and cls in names else None
                mask_arr = None
                if hasattr(res, "masks") and res.masks is not None:
                    try:
                        m = res.masks.data[i].cpu().numpy()
                        H, W = frame.shape[:2]
                        mask_arr = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                        mask_arr = (mask_arr > 0.5).astype(np.uint8)
                    except Exception:
                        mask_arr = None
                ann.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": conf,
                    "cls": cls,
                    "name": name,
                    "mask": mask_arr
                })
        return ann

def choose_best_annotation_set(annotations_per_frame):
    best_idx = None
    best_ann = []
    best_key = (-1, -1.0)
    for idx, ann in enumerate(annotations_per_frame):
        count = len(ann)
        conf_sum = sum(a["conf"] for a in ann) if count > 0 else 0.0
        key = (count, conf_sum)
        if key > best_key:
            best_key = key
            best_idx = idx
            best_ann = ann
    return best_idx, best_ann

def _pick_color_for_annotation(a):
    name = a.get("name")
    if isinstance(name, str) and name in CLASS_COLORS:
        return CLASS_COLORS[name]
    cls = a.get("cls")
    if isinstance(cls, int) and 0 <= cls < len(FALLBACK_PALETTE):
        return FALLBACK_PALETTE[cls]
    return FALLBACK_PALETTE[a.get("cls", 0) % len(FALLBACK_PALETTE)]

def draw_annotations(frame, annotations, alpha_mask=0.35):
    out = frame.copy()
    H, W = out.shape[:2]
    any_mask = any(a.get("mask") is not None for a in annotations)
    if any_mask:
        overlay = out.copy()
        for a in annotations:
            m = a.get("mask")
            if m is None:
                continue
            m = np.asarray(m)
            if m.ndim == 3:
                m = m[..., 0]
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            if m.dtype != np.bool_:
                m = m > 0
            if not m.any():
                continue
            color = np.array(_pick_color_for_annotation(a), dtype=np.float32)
            idx = m
            src = overlay[idx].astype(np.float32)
            overlay[idx] = (src * (1.0 - alpha_mask) + color * alpha_mask).astype(np.uint8)
        return overlay
    if annotations:
        overlay = out.copy()
        for a in annotations:
            bbox = a.get("bbox")
            if not bbox:
                continue
            try:
                x1, y1, x2, y2 = map(int, bbox)
            except Exception:
                continue
            x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = overlay[y1:y2, x1:x2].astype(np.float32)
            color = np.array(_pick_color_for_annotation(a), dtype=np.float32)
            overlay[y1:y2, x1:x2] = (roi * (1.0 - alpha_mask) + color * alpha_mask).astype(np.uint8)
        return overlay
    return out

