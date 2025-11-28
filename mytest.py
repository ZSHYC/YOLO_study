# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics import YOLO

model = YOLO(r"yolo11n.pt")
print(model.task)
print(model.names)
