"""
extract_frames.py  — Extract training frames from surgery video
================================================================
Extracts every Nth frame from the video, applying quality filters
to ensure diverse, useful training images.

Usage:
    python extract_frames.py /mnt/FDrive/CIT1.mp4

Output goes to: /mnt/FDrive/cit/dataset/images/
"""

import cv2
import os
import sys
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
EVERY_NTH_FRAME  = 30       # Extract 1 frame per second (at 30fps)
MIN_BRIGHTNESS   = 20       # Skip very dark frames
MAX_BRIGHTNESS   = 245      # Skip overexposed frames
BLUR_THRESHOLD   = 30       # Skip very blurry frames (Laplacian variance)
OUTPUT_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "images")


def is_quality_frame(frame):
    """Check if frame is good quality for training."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Check brightness
    mean_brightness = np.mean(gray)
    if mean_brightness < MIN_BRIGHTNESS or mean_brightness > MAX_BRIGHTNESS:
        return False, f"brightness={mean_brightness:.0f}"

    # Check blur (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < BLUR_THRESHOLD:
        return False, f"blur={laplacian_var:.0f}"

    return True, "ok"


def extract_frames(video_path):
    """Extract quality frames from video."""
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"Resolution: {w}x{h} | FPS: {fps:.0f} | Frames: {total_frames} | Duration: {duration:.1f}s")
    print(f"Extracting every {EVERY_NTH_FRAME}th frame → ~{total_frames // EVERY_NTH_FRAME} candidates")
    print(f"Output: {OUTPUT_DIR}")
    print("-" * 60)

    saved = 0
    skipped_quality = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx % EVERY_NTH_FRAME != 0:
            continue

        # Quality check
        is_good, reason = is_quality_frame(frame)
        if not is_good:
            skipped_quality += 1
            continue

        # Save frame
        filename = f"frame_{frame_idx:06d}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

        if saved % 20 == 0:
            print(f"  Saved {saved} frames...")

    cap.release()

    print("-" * 60)
    print(f"Done! Saved {saved} frames | Skipped {skipped_quality} (quality filter)")
    print(f"Frames are in: {OUTPUT_DIR}")
    print()
    print("NEXT STEPS:")
    print("1. Go to https://app.roboflow.com → Create Project → Object Detection")
    print("2. Upload all images from the 'dataset/images/' folder")
    print("3. Annotate 2 classes: 'arthroscope' and 'probe'")
    print("   - Draw bounding boxes around each instrument tip")
    print("   - You only need to annotate ~100-150 images for decent results")
    print("4. Export as 'YOLOv8' format → download the zip")
    print("5. Extract the zip into /mnt/FDrive/cit/dataset/")
    print("6. Run: python train_yolo.py")


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/FDrive/CIT1.mp4"
    extract_frames(video_path)
