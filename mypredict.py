from ultralytics import YOLO

model = YOLO(r"D:\deeplearning\ultralytics-8.3.163\runs\detect\train3\weights\best.pt")
model.predict(
    source=r"D:\Personal\Desktop\2",   # 预测目标
    # source = 0,  # 打开摄像头
    save = True,                    # 保存预测结果
    show = False, 
    save_txt = True,                  
)