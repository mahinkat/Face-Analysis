from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r'C:\Users\Mahin\Desktop\yolo_images\yolo11n-pose.pt')  # or your .pt file if downloaded

    model.train(
        data=r'C:/Users/Mahin/Desktop/yolo_images/config1.yaml',
        epochs=100,
        imgsz=640,
        device=0  # GPU
    )