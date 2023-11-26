from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train10/weights/best.pt")

# Train the model
model.predict(source="../datasets/cp/images/val", save_txt=True, save_conf=True)
