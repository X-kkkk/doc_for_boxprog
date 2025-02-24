import os
import random
import shutil

def split_dataset(data_dir, train_ratio=0.95):
    """
    划分数据集为训练集和验证集
    :param data_dir: 数据集所在的目录
    :param train_ratio: 训练集占总数据的比例，默认为0.95
    """
    # 定义图片和标注文件的路径
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    # 创建训练集和验证集的目录
    train_images_dir = os.path.join(images_dir, "train")
    val_images_dir = os.path.join(images_dir, "val")
    train_labels_dir = os.path.join(labels_dir, "train")
    val_labels_dir = os.path.join(labels_dir, "val")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # 获取所有图片文件（假设只有.jpg格式）
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    random.shuffle(image_files)  # 打乱顺序

    # 计算训练集和验证集的数量
    total_images = len(image_files)
    train_count = int(total_images * train_ratio)
    val_count = total_images - train_count

    print(f"Total images: {total_images}")
    print(f"Train images: {train_count}")
    print(f"Validation images: {val_count}")

    # 划分训练集和验证集
    train_images = image_files[:train_count]
    val_images = image_files[train_count:]

    # 复制图片和对应的标注文件到训练集和验证集目录
    for img in train_images:
        # 复制图片
        shutil.copy(os.path.join(images_dir, img), os.path.join(train_images_dir, img))
        # 复制标注文件（假设标注文件与图片同名，但扩展名为.txt）
        annotation_file = img.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(os.path.join(labels_dir, annotation_file)):
            shutil.copy(os.path.join(labels_dir, annotation_file), os.path.join(train_labels_dir, annotation_file))

    for img in val_images:
        # 复制图片
        shutil.copy(os.path.join(images_dir, img), os.path.join(val_images_dir, img))
        # 复制标注文件
        annotation_file = img.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(os.path.join(labels_dir, annotation_file)):
            shutil.copy(os.path.join(labels_dir, annotation_file), os.path.join(val_labels_dir, annotation_file))

    print("Dataset split completed!")

# 使用示例
data_directory = "path/to/your/dataset"  # 替换为你的数据集路径
split_dataset(data_directory)