from ultralytics import YOLO
import cv2
import numpy as np

# --- Load your bounding-box facial region model ---
bbox_model = YOLO("my_model/train/weights/best.pt")

# --- Check your model's class names/indices before running ---
print("Model classes:", bbox_model.names)
# Find the index that corresponds to "forehead" and set it below

FOREHEAD_CLASS_ID = 2  # <-- update this based on the printed class list above

video_path = "15secondtest.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

R_signal = []
G_signal = []
B_signal = []
skipped = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # end of video

    results = bbox_model(frame, verbose=False)
    result = results[0]
    boxes = result.boxes

    forehead_box = None
    if boxes is not None:
        for box in boxes:
            if int(box.cls[0]) == FOREHEAD_CLASS_ID:
                forehead_box = box.xyxy[0].cpu().numpy().astype(int)
                break

    if forehead_box is None:
        skipped += 1
        continue

    x1, y1, x2, y2 = forehead_box

    # Clamp coordinates to frame bounds just in case the box edges 
    # extend slightly outside the image
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        skipped += 1
        continue

    # frame is BGR (OpenCV convention)
    B = roi[:, :, 0].mean()
    G = roi[:, :, 1].mean()
    R = roi[:, :, 2].mean()

    R_signal.append(R)
    G_signal.append(G)
    B_signal.append(B)

cap.release()

print(f"Collected {len(G_signal)} frames, skipped {skipped}")

np.savez("rppg_signal_forehead.npz", R=R_signal, G=G_signal, B=B_signal, fps=fps)