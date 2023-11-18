from ultralytics import YOLO

# Load a model
model = YOLO("./YOLOv8/yolov8n.pt")

# Train the model
model.train(data="./YOLOv8/yolo-cp.yaml", workers=0, epochs=50, batch=16)
