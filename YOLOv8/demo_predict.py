from ultralytics import YOLO

yolo = YOLO("./YOLOv8/yolov8n.pt", task="detect")

result = yolo(source="./YOLOv8/ultralytics/assets/bus.jpg", save=True, conf=0.05)
