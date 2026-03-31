"""
train_yolo.py  — Train YOLOv8 on annotated arthroscopic surgery frames
======================================================================
Prerequisites:
    pip install ultralytics

This script:
  1. Creates or uses a dataset.yaml config
  2. Fine-tunes YOLOv8n (nano) on your annotated dataset
  3. Copies the best weights to weights/best.pt for SurgiScore

Usage:
    python train_yolo.py

Dataset structure expected (after Roboflow export):
    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── data.yaml          (auto-created if not present)
"""

import os
import sys
import shutil
import yaml

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR   = os.path.join(PROJECT_DIR, "dataset")
WEIGHTS_DIR   = os.path.join(PROJECT_DIR, "weights")
RUNS_DIR      = os.path.join(PROJECT_DIR, "runs")

# Training parameters
MODEL_SIZE    = "yolov8n.pt"   # nano — fast training, good for small datasets
EPOCHS        = 25             # CPU-optimized: fewer epochs
BATCH_SIZE    = 8              # CPU-optimized: smaller batch
IMG_SIZE      = 320            # CPU-optimized: smaller images for faster training
PATIENCE      = 10             # Early stopping patience

# Classes — must match your Roboflow annotation classes
CLASSES       = ["arthroscope", "probe"]


def create_dataset_yaml():
    """Create dataset.yaml if it doesn't exist."""
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")

    # Check if Roboflow already created one
    if os.path.exists(yaml_path):
        print(f"Found existing data.yaml at {yaml_path}")
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        print(f"  Classes: {config.get('names', 'unknown')}")
        return yaml_path

    # Auto-create one
    print("Creating data.yaml...")

    train_img = os.path.join(DATASET_DIR, "train", "images")
    val_img   = os.path.join(DATASET_DIR, "valid", "images")

    # If no train/valid split exists, check if there's just an images folder
    if not os.path.exists(train_img):
        raw_images = os.path.join(DATASET_DIR, "images")
        if os.path.exists(raw_images):
            print("  No train/valid split found. Creating 80/20 split...")
            _create_split(raw_images)
        else:
            print(f"ERROR: No images found in {DATASET_DIR}")
            print("Make sure you've exported from Roboflow into the dataset/ folder.")
            sys.exit(1)

    config = {
        "path": DATASET_DIR,
        "train": "train/images",
        "val": "valid/images",
        "nc": len(CLASSES),
        "names": CLASSES,
    }

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"  Created: {yaml_path}")
    return yaml_path


def _create_split(images_dir, split_ratio=0.8):
    """Split raw images + labels into train/valid folders."""
    import random

    labels_dir = os.path.join(DATASET_DIR, "labels")

    # Get all image files
    img_files = [f for f in os.listdir(images_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(img_files)

    split_idx = int(len(img_files) * split_ratio)
    train_files = img_files[:split_idx]
    val_files   = img_files[split_idx:]

    for split_name, files in [("train", train_files), ("valid", val_files)]:
        img_out = os.path.join(DATASET_DIR, split_name, "images")
        lbl_out = os.path.join(DATASET_DIR, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for fname in files:
            # Copy image
            src_img = os.path.join(images_dir, fname)
            shutil.copy2(src_img, os.path.join(img_out, fname))

            # Copy label if exists
            label_name = os.path.splitext(fname)[0] + ".txt"
            src_lbl = os.path.join(labels_dir, label_name)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(lbl_out, label_name))

    print(f"  Split: {len(train_files)} train / {len(val_files)} valid")


def check_gpu():
    """Check if GPU is available for training."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            return True
        else:
            print("GPU: Not available — training on CPU (will be slower)")
            return False
    except ImportError:
        print("GPU: PyTorch not installed — will use CPU")
        return False


def train():
    """Run YOLOv8 training."""
    print("=" * 60)
    print("YOLOv8 Training — Arthroscopic Instrument Detection")
    print("=" * 60)

    # Check GPU
    has_gpu = check_gpu()

    # Setup dataset
    yaml_path = create_dataset_yaml()

    # Check dataset size
    train_imgs = os.path.join(DATASET_DIR, "train", "images")
    if os.path.exists(train_imgs):
        n_train = len([f for f in os.listdir(train_imgs)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Training images: {n_train}")
        if n_train < 20:
            print("WARNING: Very few training images. Annotate more for better results.")
    else:
        print("ERROR: No training images found. Run extract_frames.py and annotate first.")
        sys.exit(1)

    # Check labels exist
    train_labels = os.path.join(DATASET_DIR, "train", "labels")
    if os.path.exists(train_labels):
        n_labels = len([f for f in os.listdir(train_labels) if f.endswith('.txt')])
        print(f"Training labels: {n_labels}")
        if n_labels == 0:
            print("ERROR: No label files found. Did you annotate and export from Roboflow?")
            sys.exit(1)
    else:
        print("ERROR: No labels directory found. Export annotations from Roboflow first.")
        sys.exit(1)

    print("-" * 60)
    print(f"Model: {MODEL_SIZE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | ImgSize: {IMG_SIZE}")
    print("-" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run:")
        print("  pip install ultralytics")
        sys.exit(1)

    # Load pretrained model
    model = YOLO(MODEL_SIZE)

    # Train
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        patience=PATIENCE,
        project=RUNS_DIR,
        name="arthroscope_detector",
        exist_ok=True,
        pretrained=True,
        # Augmentation for small datasets
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=10,
        translate=0.1,
        scale=0.3,
        flipud=0.3,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.1,
        # Save settings
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    # Copy best weights to the app's weights directory
    best_weights = os.path.join(RUNS_DIR, "arthroscope_detector", "weights", "best.pt")
    if os.path.exists(best_weights):
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        dest = os.path.join(WEIGHTS_DIR, "best.pt")
        shutil.copy2(best_weights, dest)
        print("\n" + "=" * 60)
        print(f"SUCCESS! Best weights copied to: {dest}")
        print("SurgiScore will now use your trained model automatically.")
        print("Restart the app: python app.py")
        print("=" * 60)
    else:
        print("\nWARNING: best.pt not found. Check training logs for errors.")
        print(f"Expected at: {best_weights}")


if __name__ == "__main__":
    train()
