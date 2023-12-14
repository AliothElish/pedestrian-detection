from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train22/weights/best.pt")

# Train the model
model.predict(source="../datasets/cp/images/val", save_txt=True, save_conf=True)

# import zipfile

# f = zipfile.ZipFile("", "r")
# for file in f.namelist():
#     f.extract(file, "")
# f.close()
