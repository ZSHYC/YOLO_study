from ultralytics import YOLO

if __name__ == "__main__":
    model =  YOLO(r"yolo11n.pt")
    model.train(
        data = r"D:\deeplearning\ultralytics-8.3.163\ultralytics\cfg\datasets\zshyc.yaml",
        epochs = 1,
        imgsz = 640,
        batch = 11,   # 若batch=-1，则自动找合适的batch大小（AutoBatch），找到后再设置具体的batch
        cache = "ram",
        workers = 1,
        patience=20,   # 新增：早停等待 20 个 epoch
        project = "results",   
        name = "zshyc"
    )
    

# # 如果要在一个数据集上训练好几个模型，比较结果
# ms = [
#     "yolov8n", "yolov8s", "yolov8m"
#     "yolov9n", "yolov9s", "yolov9m"
#     "yolov10n", "yolov10s", "yolov10m"
#     "yolo11n", "yolo11s", "yolo11m"
#     "yolo12n", "yolo12s", "yolo12m"
# ]

# if __name__ == "__main__":
#     for m in ms:
#         model = YOLO(m + ".pt")
#         model.train(
#             data = r"D:\deeplearning\ultralytics-8.3.163\ultralytics\cfg\datasets\zshyc.yaml",
#             epochs = 100,
#             imgsz = 640,
#             batch = -1,
#             cache = "ram",
#             workers = 1,
#             project = "results",
#             name = m,
#         )