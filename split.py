import os
import random
import shutil
from pathlib import Path

# -------------------------------
# User-defined paths
# -------------------------------
train_img_path = Path(r"C:\Users\Mahin\Desktop\yolo_images\data\images\train")
train_lbl_path = Path(r"C:\Users\Mahin\Desktop\yolo_images\data\labels\train")
val_img_path = Path(r"C:\Users\Mahin\Desktop\yolo_images\data\images\val")
val_lbl_path = Path(r"C:\Users\Mahin\Desktop\yolo_images\data\labels\val")

# Percentage for training set
train_pct = 0.9
val_pct = 1 - train_pct

# -------------------------------
# Create validation folders if they don't exist
# -------------------------------
for path in [val_img_path, val_lbl_path]:
    path.mkdir(parents=True, exist_ok=True)

# -------------------------------
# List all files and shuffle
# -------------------------------
img_files = list(train_img_path.glob('*'))
random.shuffle(img_files)

num_train = int(len(img_files) * train_pct)
num_val = len(img_files) - num_train

print(f"Total images: {len(img_files)}")
print(f"Images in train: {num_train}")
print(f"Images in val: {num_val}")

# -------------------------------
# Move files to validation set
# -------------------------------
for img_path in img_files[num_train:]:
    # Move image
    shutil.move(str(img_path), val_img_path / img_path.name)
    
    # Move corresponding label
    label_path = train_lbl_path / (img_path.stem + '.txt')
    if label_path.exists():
        shutil.move(str(label_path), val_lbl_path / label_path.name)

print("Dataset split complete.")
