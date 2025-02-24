import os
import json
import cv2  # 用于读取图像获取shape
import shutil  # 用于复制文件

# 需要的类别
required_categories = {
    "Arrow_signal_light",
    "Non_motorized_vehiclesignal_light",
    "Circle_signal_light",
    "Pedestrian_signal_lights",
    "Other_signal_light",
    "right_sign",
    "left_sign",
    "circle_sign",
    "bicycle_sign",
    "pedestrian_sign",
    "straight_sign",
    "uturn_sign",
    "straight_left_arrow"
}

# 颜色属性限制
color_limits = {
    "pedestrian_sign": ["red", "green"],
    "bicycle_sign": ["red", "green"],
    "right_sign": ["red", "green", "yellow"],
    "left_sign": ["red", "green", "yellow"],
    "circle_sign": ["red", "green", "yellow"],
    "straight_sign": ["red", "green", "yellow"],
    "uturn_sign": ["red", "green", "yellow"],
    "straight_left_arrow": ["red", "green", "yellow"]
}

# 类别到类ID的映射
category_mapping = {
    "Arrow_signal_light": 0,
    "Non_motorized_vehiclesignal_light": 1,
    "Circle_signal_light": 2,
    "Pedestrian_signal_lights": 3,
    "Other_signal_light": 4,
    "right_sign_red": 5,
    "right_sign_green": 6,
    "right_sign_yellow": 7,
    "left_sign_red": 8,
    "left_sign_green": 9,
    "left_sign_yellow": 10,
    "circle_sign_red": 11,
    "circle_sign_green": 12,
    "circle_sign_yellow": 13,
    "bicycle_sign_red": 14,
    "bicycle_sign_green": 15,
    "pedestrian_sign_red": 16,
    "pedestrian_sign_green": 17,
    "straight_sign_red": 18,
    "straight_sign_green": 19,
    "straight_sign_yellow": 20,
    "uturn_sign_red": 21,
    "uturn_sign_green": 22,
    "uturn_sign_yellow": 23,
    "straight_left_arrow_red": 24,
    "straight_left_arrow_green": 25,
    "straight_left_arrow_yellow": 26
}


def convert_to_yolo_format(json_file, image_path, category_mapping):
    """
    将 JSON 格式的标签转换为 YOLO 格式。

    json_file: 要转换的 JSON 文件路径
    image_path: 对应的图像路径，用于获取图片的宽度和高度
    category_mapping: 类别到类ID的映射字典
    """
    # 读取图像并获取其宽高
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    yolo_labels = []

    for annotation in data:
        category = annotation.get("category")

        # 类别不需要，跳过
        if category not in required_categories:
            continue

        # 提取颜色属性（如果存在）
        colour = annotation.get("colour", "").lower() if "colour" in annotation else None

        # 颜色属性不存在或不符合限制，跳过
        if category in color_limits:
            if colour is None or colour not in color_limits[category]:
                continue

        # 类别有颜色属性，更新类别名称
        if colour:
            category = f"{category}_{colour}"

        class_id = category_mapping.get(category)
        if class_id is None:
            continue  # 如果类别未定义，则跳过

        # 计算框的左上角和右下角
        points = annotation.get("points")
        x1, y1 = float(points[0]['x']), float(points[0]['y'])
        x2, y2 = float(points[1]['x']), float(points[1]['y'])

        # 计算中心点坐标和宽高
        x_center = (x1 + x2) / 2.0 / image_width
        y_center = (y1 + y2) / 2.0 / image_height
        width = abs(x2 - x1) / image_width
        height = abs(y2 - y1) / image_height

        # 创建YOLO格式的标签
        yolo_label = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_labels.append(yolo_label)

    return yolo_labels


def save_yolo_labels(output_dir, image_filename, yolo_labels):
    """
    将 YOLO 格式的标签保存为 .txt 文件。

    output_dir: 保存输出标签文件的目录
    image_filename: 对应的图像文件名
    yolo_labels: 要保存的 YOLO 格式标签
    """
    txt_filename = os.path.join(output_dir, os.path.splitext(image_filename)[0] + ".txt")
    with open(txt_filename, "w", encoding="utf-8") as f:
        for label in yolo_labels:
            f.write(label + "\n")


def copy_image(image_path, output_image_dir, image_filename):
    """
    将对应的图片复制到指定文件夹。

    image_path: 原图片路径
    output_image_dir: 输出图片文件夹
    image_filename: 图片文件名
    """
    output_image_path = os.path.join(output_image_dir, image_filename)
    shutil.copy(image_path, output_image_path)


def process_labels(root_dir, image_dir, output_dir, output_image_dir, category_mapping):
    """
    遍历所有标签文件，将其转换为 YOLO 格式并保存为 .txt 文件，同时复制对应的图片。

    root_dir: 包含 JSON 标签文件的根目录
    image_dir: 存放图像文件的目录
    output_dir: 输出 YOLO 格式标签文件的目录
    output_image_dir: 输出图片文件夹
    category_mapping: 类别到类ID的映射字典
    """
    for json_file in os.listdir(root_dir):
        if json_file.endswith(".json"):
            json_path = os.path.join(root_dir, json_file)
            image_filename = f"{os.path.splitext(json_file)[0]}.jpg"  # 假设图片命名规则是 数字.jpg
            image_path = os.path.join(image_dir, image_filename)

            if os.path.exists(image_path):
                yolo_labels = convert_to_yolo_format(json_path, image_path, category_mapping)
                if yolo_labels:  # 如果有有效的标签
                    save_yolo_labels(output_dir, image_filename, yolo_labels)
                    copy_image(image_path, output_image_dir, image_filename)

# 设置路径
root_dir = "./training"  # 标签文件夹路径
image_dir = "./images"  # 图像文件夹路径
output_dir = "./yolo_labels"  # YOLO标签保存路径
output_image_dir = "./yolo_images"  # 输出图片文件夹
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_image_dir, exist_ok=True)

# 执行转换
process_labels(root_dir, image_dir, output_dir, output_image_dir, category_mapping)