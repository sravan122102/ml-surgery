"""
auto_label.py — Auto-annotate arthroscopic surgery frames for YOLOv8 training
================================================================================
Uses OpenCV to detect metallic surgical instruments in arthroscopic frames
and generates YOLO-format bounding box annotations (.txt files).

Strategy:
  - Detect the circular arthroscopic view region (left half of dual-view)
  - Within that region, find bright/metallic instrument contours
  - Classify as arthroscope (class 0, larger) vs probe (class 1, smaller)
  - Also detect instruments in the external view (right half)
  - Output YOLO format: class_id x_center y_center width height (normalized)

Usage:
    python auto_label.py
"""

import cv2
import numpy as np
import os
import sys
import random
import shutil
import yaml

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(PROJECT_DIR, "dataset")
IMAGES_DIR   = os.path.join(DATASET_DIR, "images")
LABELS_DIR   = os.path.join(DATASET_DIR, "labels")

# Classes matching analyzer.py
CLASS_ARTHROSCOPE = 0
CLASS_PROBE       = 1
CLASSES           = ["arthroscope", "probe"]

# Detection thresholds
MIN_CONTOUR_AREA   = 800    # Minimum pixels for a valid instrument
MAX_CONTOUR_AREA   = 80000  # Maximum pixels (avoid detecting background)
METALLIC_LOW_HSV   = np.array([0, 0, 140])     # Bright/metallic objects
METALLIC_HIGH_HSV  = np.array([180, 80, 255])
INSTRUMENT_LOW_HSV = np.array([0, 0, 100])      # Slightly darker instruments
INSTRUMENT_HIGH_HSV= np.array([180, 120, 240])


def detect_circular_view(frame):
    """Detect the arthroscopic circular view boundary."""
    h, w = frame.shape[:2]
    # The circular view is typically in the left portion
    left_half = frame[:, :w//2]
    gray = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=100, param1=80, param2=50,
        minRadius=min(h, w//2) // 4,
        maxRadius=min(h, w//2)
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Take the largest circle
        largest = max(circles[0], key=lambda c: c[2])
        return int(largest[0]), int(largest[1]), int(largest[2])
    
    # Fallback: assume center of left half
    return w//4, h//2, min(h, w//2) // 2 - 20


def detect_instruments_in_roi(roi, roi_offset_x=0, roi_offset_y=0):
    """
    Detect metallic instruments in an ROI using multi-method approach.
    Returns list of (class_id, x, y, w, h) in pixel coordinates relative to full frame.
    """
    detections = []
    h, w = roi.shape[:2]
    
    if h < 20 or w < 20:
        return detections
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Detect bright metallic objects (instrument tips/shafts)
    metallic_mask = cv2.inRange(hsv, METALLIC_LOW_HSV, METALLIC_HIGH_HSV)
    
    # Method 2: Edge-based detection for instrument outlines
    edges = cv2.Canny(gray, 50, 150)
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    
    # Method 3: Detect instruments by color anomaly (instruments vs tissue)
    # Tissue is typically pinkish/reddish, instruments are gray/silver
    instrument_mask = cv2.inRange(hsv, INSTRUMENT_LOW_HSV, INSTRUMENT_HIGH_HSV)
    
    # Combine masks
    combined = cv2.bitwise_or(metallic_mask, instrument_mask)
    combined = cv2.bitwise_and(combined, dilated_edges)
    
    # Also add pure metallic detections
    combined = cv2.bitwise_or(combined, metallic_mask)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area and aspect ratio
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            # Instruments are typically elongated (aspect > 1.5) or compact tips
            if aspect > 1.2 or area > 2000:
                valid_contours.append((cnt, area, x, y, bw, bh))
    
    if not valid_contours:
        return detections
    
    # Sort by area (largest first)
    valid_contours.sort(key=lambda x: x[1], reverse=True)
    
    for i, (cnt, area, x, y, bw, bh) in enumerate(valid_contours[:3]):
        # Largest = arthroscope, others = probe
        cls_id = CLASS_ARTHROSCOPE if i == 0 and area > 3000 else CLASS_PROBE
        
        # Add padding
        pad = 8
        x = max(0, x - pad)
        y = max(0, y - pad)
        bw = min(w - x, bw + 2 * pad)
        bh = min(h - y, bh + 2 * pad)
        
        detections.append((
            cls_id,
            x + roi_offset_x,
            y + roi_offset_y,
            bw,
            bh
        ))
    
    return detections


def detect_instruments_adaptive(frame):
    """
    Detect instruments using adaptive background modeling.
    Works on the full frame with awareness of the dual-view layout.
    """
    h, w = frame.shape[:2]
    all_detections = []
    
    # ── Arthroscopic view (left half, inside the circle) ──
    left_half = frame[:, :w//2]
    cx, cy, radius = detect_circular_view(frame)
    
    # Create circular mask
    mask = np.zeros(left_half.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius - 10, 255, -1)
    
    # Apply mask to get just the arthroscopic view
    arthro_view = cv2.bitwise_and(left_half, left_half, mask=mask)
    
    # Detect in arthroscopic view
    arthro_dets = detect_instruments_in_roi(arthro_view, roi_offset_x=0, roi_offset_y=0)
    
    # Filter detections that are inside the circle
    for det in arthro_dets:
        cls_id, x, y, bw, bh = det
        center_x = x + bw // 2
        center_y = y + bh // 2
        dist = np.hypot(center_x - cx, center_y - cy)
        if dist < radius - 5:
            all_detections.append(det)
    
    # ── External view (right half) — detect instruments there too ──
    right_half = frame[:, w//2:]
    ext_dets = detect_instruments_in_roi(right_half, roi_offset_x=w//2, roi_offset_y=0)
    all_detections.extend(ext_dets)
    
    return all_detections


def to_yolo_format(detections, img_w, img_h):
    """Convert pixel detections to YOLO normalized format."""
    yolo_lines = []
    for cls_id, x, y, bw, bh in detections:
        x_center = (x + bw / 2) / img_w
        y_center = (y + bh / 2) / img_h
        width    = bw / img_w
        height   = bh / img_h
        
        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width    = max(0.001, min(1.0, width))
        height   = max(0.001, min(1.0, height))
        
        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_lines


def auto_label_all():
    """Auto-label all frames in dataset/images/."""
    os.makedirs(LABELS_DIR, exist_ok=True)
    
    img_files = sorted([f for f in os.listdir(IMAGES_DIR) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not img_files:
        print(f"ERROR: No images found in {IMAGES_DIR}")
        sys.exit(1)
    
    print(f"Auto-labeling {len(img_files)} frames...")
    print(f"Classes: {CLASSES}")
    print("-" * 60)
    
    labeled = 0
    skipped = 0
    class_counts = {0: 0, 1: 0}
    
    for i, fname in enumerate(img_files):
        img_path = os.path.join(IMAGES_DIR, fname)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"  SKIP: Cannot read {fname}")
            skipped += 1
            continue
        
        h, w = frame.shape[:2]
        
        # Detect instruments
        detections = detect_instruments_adaptive(frame)
        
        if not detections:
            # Even with no detection, create an empty label file
            # (tells YOLO this is a negative example)
            label_name = os.path.splitext(fname)[0] + ".txt"
            label_path = os.path.join(LABELS_DIR, label_name)
            with open(label_path, "w") as f:
                pass  # empty file
            skipped += 1
            continue
        
        # Convert to YOLO format
        yolo_lines = to_yolo_format(detections, w, h)
        
        # Write label file
        label_name = os.path.splitext(fname)[0] + ".txt"
        label_path = os.path.join(LABELS_DIR, label_name)
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")
        
        labeled += 1
        for det in detections:
            class_counts[det[0]] = class_counts.get(det[0], 0) + 1
        
        if (i + 1) % 30 == 0:
            print(f"  Processed {i+1}/{len(img_files)} — labeled: {labeled}, skipped: {skipped}")
    
    print("-" * 60)
    print(f"Done! Labeled: {labeled} | Skipped/empty: {skipped}")
    print(f"Class distribution: arthroscope={class_counts.get(0,0)}, probe={class_counts.get(1,0)}")
    print(f"Labels saved to: {LABELS_DIR}")
    return labeled


def create_train_val_split(split_ratio=0.8):
    """Split labeled data into train/valid sets."""
    img_files = sorted([f for f in os.listdir(IMAGES_DIR)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Only include images that have labels
    labeled_files = []
    for fname in img_files:
        label_name = os.path.splitext(fname)[0] + ".txt"
        label_path = os.path.join(LABELS_DIR, label_name)
        if os.path.exists(label_path):
            labeled_files.append(fname)
    
    random.shuffle(labeled_files)
    split_idx = int(len(labeled_files) * split_ratio)
    train_files = labeled_files[:split_idx]
    val_files   = labeled_files[split_idx:]
    
    for split_name, files in [("train", train_files), ("valid", val_files)]:
        img_out = os.path.join(DATASET_DIR, split_name, "images")
        lbl_out = os.path.join(DATASET_DIR, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        
        for fname in files:
            # Copy image
            shutil.copy2(
                os.path.join(IMAGES_DIR, fname),
                os.path.join(img_out, fname)
            )
            # Copy label
            label_name = os.path.splitext(fname)[0] + ".txt"
            src_label = os.path.join(LABELS_DIR, label_name)
            if os.path.exists(src_label):
                shutil.copy2(src_label, os.path.join(lbl_out, label_name))
    
    print(f"\nSplit: {len(train_files)} train / {len(val_files)} valid")
    return len(train_files), len(val_files)


def create_data_yaml():
    """Create dataset/data.yaml for YOLOv8 training."""
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    
    config = {
        "path": DATASET_DIR.replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    print("=" * 60)
    print("SurgiScore Auto-Labeler — Arthroscopic Instrument Detection")
    print("=" * 60)
    
    # Step 1: Auto-label
    n_labeled = auto_label_all()
    
    if n_labeled < 5:
        print("\nWARNING: Very few frames labeled. Training may not converge.")
        print("Consider manually annotating with Roboflow for better results.")
    
    # Step 2: Split
    print("\n" + "=" * 60)
    print("Creating train/valid split...")
    n_train, n_val = create_train_val_split()
    
    # Step 3: Create data.yaml
    print("\n" + "=" * 60)
    create_data_yaml()
    
    print("\n" + "=" * 60)
    print("AUTO-LABELING COMPLETE!")
    print(f"  Train: {n_train} | Valid: {n_val}")
    print("\nNext step: Run training")
    print("  python train_yolo.py")
    print("=" * 60)
