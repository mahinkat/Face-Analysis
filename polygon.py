from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/pose/train5/weights/best.pt")

results = model('data/images/1cc65bc3-_-17-_jpeg.rf.48e1fa229e6cd2d12868e72e61e95b04.jpg')
result = results[0]
keypoints = result.keypoints.xy[0].cpu().tolist()
print(keypoints)


annotated_image = result.plot()
cv2.imwrite("output.jpg", annotated_image)

image = cv2.imread('data/images/1cc65bc3-_-17-_jpeg.rf.48e1fa229e6cd2d12868e72e61e95b04.jpg')
selected_indices = (54, 35, 45, 16, 14)
resultArr = np.array(
    [keypoints[i] for i in selected_indices],
    dtype=np.int32
).reshape((-1, 1, 2))
resultArr = cv2.convexHull(resultArr)



print("Result array:" , resultArr)
#color = (255, 0, 0)

#cv2.fillPoly(
#    image,
#    [resultArr],
#    color=color
#)
mask = np.zeros(image.shape[:2], dtype=np.uint8)

cv2.fillPoly(mask, [resultArr], 255)



result = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("result", result)
cv2.waitKey(1)         
cv2.destroyAllWindows() 
pixels = image[mask == 255]

print("Pixels:")
print(pixels)
B = pixels[:, 0].mean()
G = pixels[:, 1].mean()
R = pixels[:, 2].mean()

rgbres = [R,G,B]

print(rgbres)




print("Number of pixels:", len(pixels))

video_path = "15secondtest.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

R_signal = []
G_signal = []
B_signal = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # end of video

    results = model(frame, verbose=False)
    result = results[0]

    if result.keypoints is None or len(result.keypoints.xy[0]) == 0:
        continue  # no face detected this frame, skip it

    keypoints = result.keypoints.xy[0].cpu().tolist()

    resultArr = np.array(
        [keypoints[i] for i in selected_indices],
        dtype=np.int32
    ).reshape((-1, 1, 2))
    resultArr = cv2.convexHull(resultArr)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [resultArr], 255)

    pixels = frame[mask == 255]

    if len(pixels) == 0:
        continue  # mask ended up empty, skip

    B = pixels[:, 0].mean()
    G = pixels[:, 1].mean()
    R = pixels[:, 2].mean()

    R_signal.append(R)
    G_signal.append(G)
    B_signal.append(B)

cap.release()

print(f"Collected {len(G_signal)} frames of signal")

# Save the raw signal for later processing (filtering, FFT, etc.)
np.savez("rppg_signal.npz", R=R_signal, G=G_signal, B=B_signal, fps=fps)


# for still images
#while(1):
#    
#    cv2.imshow('image', image)
#    if cv2.waitKey(20) & 0xFF == 27:
#        
#        break
#cv2.destroyAllWindows()

