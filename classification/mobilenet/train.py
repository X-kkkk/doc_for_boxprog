import os
import sys
import json
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model_v2 import MobileNetV2

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    batch_size = 16
    epochs = 5

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 指定数据集的绝对路径，需要你自己修改
    image_path = "/your/absolute/path/to/your_dataset"
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # 创建模型，需要根据你的分类数量修改
    net = MobileNetV2(num_classes=5)

    # 冻结特征层权重
    for param in net.features.parameters():
        param.requires_grad = False

    net.to(device)

    # 定义损失函数
    loss_function = nn.CrossEntropyLoss(reduction='sum')  # 验证时累加总损失

    # 构建优化器
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    best_acc = 0.0
    save_path = './MobileNetV2.pth'
    train_steps = len(train_loader)
    results = []  # 存储所有指标的列表
    columns = ['Epoch', 'Train Loss', 'Train Acc', 'Train Recall', 'Train F1',
               'Val Loss', 'Val Acc', 'Val Recall', 'Val F1']  # 表格列名

    for epoch in range(epochs):
        # 训练
        net.train()
        running_loss = 0.0
        train_labels = []  # 收集真实标签
        train_preds = []   # 收集预测标签
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # 打印统计信息
            running_loss += loss.item()
            train_labels.extend(labels.cpu().numpy())  # 保存真实标签
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())  # 保存预测标签

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # 计算训练指标
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_recall = recall_score(train_labels, train_preds, average='macro')  # 多分类用macro
        train_f1 = f1_score(train_labels, train_preds, average='macro')

        # 验证
        net.eval()
        val_total_loss = 0.0  # 总损失（用于计算平均）
        val_labels = []  # 真实标签
        val_preds = []  # 预测标签
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_images, val_labels_batch in val_bar:
                outputs = net(val_images.to(device))

                # 计算单batch损失（累加总损失）
                loss = loss_function(outputs, val_labels_batch.to(device))
                val_total_loss += loss.item()

                # 收集标签（真实+预测）
                val_labels.extend(val_labels_batch.cpu().numpy())  # 真实标签
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())  # 预测标签

                val_bar.desc = f"valid epoch[{epoch + 1}/{epochs}]"

        # 计算验证指标（基于所有batch的标签）
        val_loss = val_total_loss / len(validate_loader)  # 总损失 / 总批次数量
        val_acc = accuracy_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds, average='macro')
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        # 保存指标
        results.append([
            epoch + 1,          # 轮数
            train_loss,       # 训练损失
            train_acc,        # 训练准确率
            train_recall,     # 训练召回率
            train_f1,         # 训练F1
            val_loss,         # 验证损失
            val_acc,          # 验证准确率
            val_recall,       # 验证召回率
            val_f1            # 验证F1
        ])

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)

    # 保存为表格（自动生成results.csv）
    df = pd.DataFrame(results, columns=columns)
    df.to_csv('results.csv', index=False)
    print("指标已保存至 results.csv")
    print('Finished Training')

if __name__ == '__main__':
    main()