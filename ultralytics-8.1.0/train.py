from ultralytics import YOLO

# 加载 YOLOv8m 配置文件
model = YOLO("yolov8s.yaml")  

# 开始训练
model.train(
    data="/home/xuke/box_prog/ultralytics-8.1.0/box_light_yolov8.yaml",  # 数据集配置文件的绝对路径
    epochs=200,                                 # 训练轮数
    imgsz=1280,                                  # 图像大小
    batch=16,                                   # 批量大小
    device=3                                    # GPU设备编号
)