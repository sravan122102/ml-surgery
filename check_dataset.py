import os

labels = "dataset/labels"
l = os.listdir(labels) if os.path.exists(labels) else []
non_empty = [f for f in l if os.path.getsize(os.path.join(labels, f)) > 0]
print(f"Total labels: {len(l)}")
print(f"Non-empty (with detections): {len(non_empty)}")
print(f"Empty (negative examples): {len(l) - len(non_empty)}")

td = "dataset/train/images"
vd = "dataset/valid/images"
print(f"Train images: {len(os.listdir(td)) if os.path.exists(td) else 0}")
print(f"Valid images: {len(os.listdir(vd)) if os.path.exists(vd) else 0}")
print(f"data.yaml exists: {os.path.exists('dataset/data.yaml')}")

# Show a sample label
if non_empty:
    sample = os.path.join(labels, non_empty[0])
    print(f"\nSample label ({non_empty[0]}):")
    with open(sample) as f:
        print(f.read())
