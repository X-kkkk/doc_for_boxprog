"""
显示推理时间
"""
import torch
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.file import download_from_url
from PIL import Image
import os
import time

# 自定义加载函数
def custom_load(file):
    return torch.load(file, map_location="cuda", weights_only=False), file


# 替换 ultralytics 中的加载函数
from ultralytics.nn.tasks import attempt_load_one_weight
import types

attempt_load_one_weight.__globals__['torch_safe_load'] = custom_load

# 加载训练好的模型
model_path = 'E:/hezi/model/v8s-p2_e300_imgsz640_b16/weights/best.pt'
model = YOLO(model_path)

# 使用 SAHI 加载模型
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.3,
    device="cuda:0",  # 可根据实际情况修改为 'cpu'
)

# 输入图片路径
image_path = 'E:/hezi/predict_data/picture/scene5.jpg'

# 获取图片文件名（不包含扩展名）
image_name = os.path.splitext(os.path.basename(image_path))[0]

# 确保保存目录存在
save_dir = "yolo_sahi_result"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 记录开始时间
start_time = time.time()

# 执行切片推理
result_sliced = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

# 记录结束时间
end_time = time.time()

# 计算推理时间
inference_time = end_time - start_time
print(f"推理时间: {inference_time} 秒")

# 对切片推理结果进行可视化
visualization_file_name = image_name
visualization_path = os.path.join(save_dir, visualization_file_name + ".jpg")
try:
    temp_file_name_sliced = f"{image_name}_sliced"
    temp_path_sliced = os.path.join(save_dir, temp_file_name_sliced + ".png")
    result_sliced.export_visuals(export_dir=save_dir, file_name=temp_file_name_sliced)
    # 转换为 JPG 并显示
    img_sliced = Image.open(temp_path_sliced)
    img_sliced.convert('RGB').save(visualization_path, 'JPEG')
    img_sliced.show()
    os.remove(temp_path_sliced)
except Exception as e:
    print(f"切片结果可视化错误: {e}")

# 处理预测结果（原打印逻辑保留）
object_prediction_list = result_sliced.object_prediction_list
for prediction in object_prediction_list:
    class_id = prediction.category.id
    conf = prediction.score.value
    bbox = prediction.bbox.to_xyxy()
    print(f"Class ID: {class_id}, Confidence: {conf}, Bbox: {bbox}")