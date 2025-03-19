from ultralytics import YOLO
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# 手动加载 YOLO 模型权重
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo_ckpt = torch.load('E:/hezi/v8s_e300_2stage/weights/best.pt', weights_only=False, map_location=device)
    # yolo_ckpt = torch.load('E:/hezi/v8s-p2_e300_2stage/weights/best.pt', weights_only=False, map_location=device)
except Exception as e:
    print(f"加载 YOLO 模型权重时出错: {e}")
    exit(1)

# 创建 YOLO 模型并传入加载的权重
model = YOLO('')
model.model = yolo_ckpt.get('model').to(device)

# 将模型数据类型统一为单精度浮点数
model.model = model.model.float()

# 定义需要分类的特定类别名称
specific_classes = [
    "right_sign",
    "left_sign",
    "circle_sign",
    "bicycle_sign",
    "pedestrian_sign",
    "straight_sign",
    "uturn_sign",
    "straight_left_arrow"
]

# 读取图片
image_path = 'E:/hezi/box/scene5.jpg'
image = cv2.imread(image_path)

# 加载自定义的ResNet50分类模型（使用指定设备）
resnet_model = models.resnet50()
num_classes = 3  # 红、绿、黄三种颜色
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, num_classes)
try:
    # 加载模型到指定设备
    resnet_model.load_state_dict(
        torch.load('E:/hezi/resNet50.pth', map_location=device)
    )
    resnet_model = resnet_model.to(device)
    resnet_model.eval()
    print("ResNet50模型加载成功")
except Exception as e:
    print(f"加载ResNet50模型时出错: {e}")

# 定义图像预处理函数
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 颜色分类映射字典
color_mapping = {
    0: 'red',
    1: 'green',
    2: 'yellow'
}

# 进行目标检测（使用指定设备）
detection_results = model(image, device=device)

# 遍历检测结果
for result in detection_results:
    boxes = result.boxes
    for box in boxes:
        class_index = int(box.cls.item())
        class_name = result.names[class_index]
        # 检查是否为特定类别
        if class_name in specific_classes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            # 裁剪图像
            cropped_image = image[y1:y2, x1:x2]

            # 预处理图像
            input_tensor = preprocess(cropped_image)
            input_batch = input_tensor.unsqueeze(0).to(device)

            # 进行分类（使用指定设备）
            with torch.no_grad():
                output = resnet_model(input_batch)

            # 获取预测结果
            _, predicted_idx = torch.max(output, 1)
            predicted_class = predicted_idx.item()
            color_label = color_mapping.get(predicted_class, 'unknown')

            # 在原检测图片上绘制边界框和分类标签
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, color_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 显示结果图像
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()