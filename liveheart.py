from ultralytics import YOLO
import cv2
import numpy as np
from scipy import signal
from collections import deque
import time

bbox_model = YOLO("my_model/train/weights/best.pt")
pose_model = YOLO("runs/pose/train5/weights/best.pt")

FOREHEAD_CLASS_ID = 2  # confirmed from your model.names output

FPS_ASSUMED = 30
WINDOW_SECONDS = 8
UPDATE_EVERY_SECONDS = 1

LOW_HZ = 0.7
HIGH_HZ = 4.0

POSE_MODEL_INTERVAL = 3  # only run the pose model every Nth frame

# IMPORTANT: verify these indices against YOUR model's actual point layout
# before trusting them — plot indices on a test image if unsure.
# Standard 68-point layout assumed here:
# jawline 0-16, right eyebrow 17-21, left eyebrow 22-26,
# nose 27-35, right eye 36-41, left eye 42-47
RIGHT_CHEEK_IDX = [1, 2, 3, 31, 36]   # jaw + nose side + eye corner (right side)
LEFT_CHEEK_IDX = [13, 14, 15, 35, 45]  # mirror on left side


def remove_outliers(signal_array, threshold=3.0):
    signal_array = np.array(signal_array, dtype=float)
    if len(signal_array) < 5:
        return signal_array

    diffs = np.diff(signal_array)
    median_diff = np.median(diffs)
    mad = np.median(np.abs(diffs - median_diff))
    mad_std = mad * 1.4826

    is_outlier = np.zeros(len(signal_array), dtype=bool)
    for i in range(1, len(signal_array)):
        jump = signal_array[i] - signal_array[i - 1]
        score = abs(jump - median_diff) / mad_std if mad_std > 0 else 0
        if score > threshold:
            is_outlier[i] = True

    good_idx = np.where(~is_outlier)[0]
    bad_idx = np.where(is_outlier)[0]
    cleaned = signal_array.copy()
    if len(bad_idx) > 0 and len(good_idx) > 1:
        cleaned[bad_idx] = np.interp(bad_idx, good_idx, signal_array[good_idx])
    return cleaned


def compute_bpm(buffer, fps):
    # Need nyquist (fps/2) > HIGH_HZ, i.e. fps > HIGH_HZ * 2, with margin
    if fps <= HIGH_HZ * 2:
        return None

    G = remove_outliers(np.array(buffer))
    G = signal.detrend(G)

    nyquist = fps / 2
    b, a = signal.butter(N=3, Wn=[LOW_HZ / nyquist, HIGH_HZ / nyquist], btype="band")
    filtered = signal.filtfilt(b, a, G)

    n = len(filtered)
    freqs = np.fft.rfftfreq(n, d=1 / fps)
    fft_vals = np.abs(np.fft.rfft(filtered))

    valid = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    if not valid.any():
        return None

    peak_freq = freqs[valid][np.argmax(fft_vals[valid])]
    return peak_freq * 60


def get_forehead_roi(frame, box):
    x1, y1, x2, y2 = box
    box_h = y2 - y1
    box_w = x2 - x1

    y1_inset = y1 + int(box_h * 0.25)
    x1_inset = x1 + int(box_w * 0.10)
    x2_inset = x2 - int(box_w * 0.10)

    h, w = frame.shape[:2]
    y1_inset, y2c = max(0, y1_inset), min(h, y2)
    x1_inset, x2_inset = max(0, x1_inset), min(w, x2_inset)

    return frame[y1_inset:y2c, x1_inset:x2_inset]


def get_cheek_mean(frame, keypoints, indices):
    pts = np.array([keypoints[i] for i in indices], dtype=np.int32)
    hull = cv2.convexHull(pts.reshape((-1, 1, 2)))

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)

    pixels = frame[mask == 255]
    if len(pixels) == 0:
        return None, hull
    return pixels[:, 1].mean(), hull  # green channel


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

buffer_size = int(FPS_ASSUMED * WINDOW_SECONDS)
buffer = deque(maxlen=buffer_size)
timestamps = deque(maxlen=buffer_size)

current_bpm = None
last_update_time = time.time()
frame_idx = 0
last_keypoints = None

print("Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # --- Bbox model: forehead (run every frame) ---
    bbox_results = bbox_model(frame, verbose=False)
    boxes = bbox_results[0].boxes

    forehead_box = None
    if boxes is not None:
        for box in boxes:
            if int(box.cls[0]) == FOREHEAD_CLASS_ID:
                forehead_box = box.xyxy[0].cpu().numpy().astype(int)
                break

    # --- Pose model: cheeks (throttled — only every Nth frame) ---
    if frame_idx % POSE_MODEL_INTERVAL == 0:
        pose_results = pose_model(frame, verbose=False)
        pose_result = pose_results[0]
        if pose_result.keypoints is not None and pose_result.keypoints.xy.shape[0] > 0:
            last_keypoints = pose_result.keypoints.xy[0].cpu().tolist()

    keypoints = last_keypoints  # reuse most recent detection between throttled frames

    # --- Collect per-ROI means for this frame ---
    frame_values = []

    if forehead_box is not None:
        roi = get_forehead_roi(frame, forehead_box)
        if roi.size > 0:
            frame_values.append(roi[:, :, 1].mean())
        x1, y1, x2, y2 = forehead_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    right_hull = left_hull = None
    if keypoints is not None:
        right_val, right_hull = get_cheek_mean(frame, keypoints, RIGHT_CHEEK_IDX)
        left_val, left_hull = get_cheek_mean(frame, keypoints, LEFT_CHEEK_IDX)
        if right_val is not None:
            frame_values.append(right_val)
        if left_val is not None:
            frame_values.append(left_val)
        if right_hull is not None:
            cv2.polylines(frame, [right_hull], True, (255, 0, 0), 2)
        if left_hull is not None:
            cv2.polylines(frame, [left_hull], True, (255, 0, 0), 2)

    # --- Average across all ROIs found this frame, append one combined value ---
    if frame_values:
        buffer.append(np.mean(frame_values))
        timestamps.append(time.time())

    # --- Recompute BPM periodically ---
    now = time.time()
    if now - last_update_time >= UPDATE_EVERY_SECONDS and len(buffer) >= FPS_ASSUMED * 4:
        elapsed = timestamps[-1] - timestamps[0]
        actual_fps = len(timestamps) / elapsed if elapsed > 0 else FPS_ASSUMED
        print(f"Actual measured fps: {actual_fps:.2f}")

        bpm = compute_bpm(list(buffer), actual_fps)
        if bpm is not None:
            current_bpm = bpm
        last_update_time = now

    display_text = f"BPM: {current_bpm:.1f}" if current_bpm else "BPM: measuring..."
    cv2.putText(frame, display_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 2)

    cv2.imshow("rPPG Heart Rate Monitor (forehead + cheeks)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()