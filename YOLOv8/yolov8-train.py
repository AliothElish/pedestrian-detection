from ultralytics import YOLO


batchs = [18]

for i in batchs:
    # Load a model
    model = YOLO("yolov8m.pt")

    # Train the model
    model.train(data="yolo-cp.yaml", workers=0, epochs=30, batch=i, optimizer="AdamW")
