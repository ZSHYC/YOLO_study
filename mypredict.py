# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics import YOLO

model = YOLO(r"yolo11n.pt")
model.predict(
    source=r"D:\Personal\Desktop\img\ac76c3cc7fcb813cd066f3cf3d4ea17c.mp4",  # 预测目标
    # source = 0,  # 打开摄像头
    save=False,  # 保存预测结果
    show=True,  # 立刻显示结果
)
