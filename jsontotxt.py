import json
import os
from PIL import Image

# Paths
json_path = "/Users/mahin/Downloads/facial_keypoints/all_data.json"     # your JSON file
images_dir = "/Users/mahin/Downloads/facial_keypoints/images"            # folder with .png images
labels_dir = "/Users/mahin/Downloads/facial_keypoints/labels"            # where to save YOLO .txt files
os.makedirs(labels_dir, exist_ok=True)

# Load JSON
with open(json_path, 'r') as f:
    data = json.load(f)

for key, entry in data.items():
    file_name = entry["file_name"]
    landmarks = entry["face_landmarks"]

    # Load image to get size
    img_path = os.path.join(images_dir, file_name)
    with Image.open(img_path) as img:
        w, h = img.size

    # Compute bounding box
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = ((x_min + x_max) / 2) / w
    y_center = ((y_min + y_max) / 2) / h
    width = (x_max - x_min) / w
    height = (y_max - y_min) / h

    # Normalize keypoints
    keypoints = []
    for (x, y) in landmarks:
        keypoints.append(f"{x/w:.6f}")
        keypoints.append(f"{y/h:.6f}")
        keypoints.append("2")  # visibility = 2 (visible)

    # Build YOLO line
    yolo_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} " + " ".join(keypoints)

    # Save label file
    label_path = os.path.join(labels_dir, file_name.replace(".png", ".txt"))
    with open(label_path, "w") as f:
        f.write(yolo_line + "\n")

print("Conversion complete! YOLOv11-pose labels saved to:", labels_dir)
