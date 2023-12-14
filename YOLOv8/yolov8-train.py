from ultralytics import YOLO


# Load a model
model = YOLO("runs/detect/best (1).pt")

# Train the model
model.train(
    data="yolo-trans.yaml",
    workers=0,
    epochs=80,
    batch=16,
    optimizer="AdamW",
    imgsz=[1440, 736],
)
# ,
#     scale=0,
#     flipud=0.5,
#     mosaic=0,
#     copy_paste=0.3,
