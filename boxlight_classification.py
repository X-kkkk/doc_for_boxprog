import os
import json
import shutil
import cv2

# 需要的类别
required_categories_for_classification = {
    "pedestrian_sign",
    "bicycle_sign",
    "right_sign",
    "left_sign",
    "circle_sign",
    "straight_sign",
    "uturn_sign",
    "straight_left_arrow"
}

# 颜色限制
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

def classify_and_crop_images(root_dir, image_dir, output_dir):
    """
    根据颜色信息对图片进行分类，并裁剪出对应的区域。
    """
    os.makedirs(output_dir, exist_ok=True)
    # 创建颜色文件夹
    color_dirs = {}
    for color in ["red", "green", "yellow"]:
        color_path = os.path.join(output_dir, color)
        os.makedirs(color_path, exist_ok=True)
        color_dirs[color] = color_path

    # 用于记录每个颜色类别的图片数量
    color_count = {"red": 0, "green": 0, "yellow": 0}

    for json_file in os.listdir(root_dir):
        if json_file.endswith(".json"):
            json_path = os.path.join(root_dir, json_file)
            image_filename = f"{os.path.splitext(json_file)[0]}.jpg"
            image_path = os.path.join(image_dir, image_filename)

            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is None:
                    print(f"无法读取图像: {image_path}")
                    continue

                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for annotation in data:
                    category = annotation.get("category")
                    if category not in required_categories_for_classification:
                        continue

                    colour = annotation.get("colour", "").lower()
                    if colour not in color_limits[category]:
                        continue

                    points = annotation.get("points")
                    x1, y1 = int(float(points[0]['x'])), int(float(points[0]['y']))
                    x2, y2 = int(float(points[1]['x'])), int(float(points[1]['y']))

                    # 裁剪区域
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    if x_min < 0:
                        x_min = 0
                    if y_min < 0:
                        y_min = 0
                    if x_max > image.shape[1]:
                        x_max = image.shape[1]
                    if y_max > image.shape[0]:
                        y_max = image.shape[0]
                    cropped_image = image[y_min:y_max, x_min:x_max]
                    if cropped_image.size == 0:
                        continue

                    # 保存裁剪后的图片
                    color_count[colour] += 1
                    output_filename = f"{colour}_{color_count[colour]}.jpg"
                    output_path = os.path.join(color_dirs[colour], output_filename)
                    cv2.imwrite(output_path, cropped_image)

# 设置路径
root_dir = "./training"  # 标签文件夹路径
image_dir = "./images"  # 图像文件夹路径
output_dir = "./classification_images"  # 分类图片保存路径

# 执行分类和裁剪
classify_and_crop_images(root_dir, image_dir, output_dir)