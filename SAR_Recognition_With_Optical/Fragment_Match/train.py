# -*- coding: utf-8 -*-
# @Time    : 2024/3/6 0006 20:50
# @Author  : Ronin
# @File    : train.py
# @Software: PyCharm
import os.path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Fragment_Match.models.model import SRWO
from Fragment_Match.utils.dataloders import LoadImagesAndLabels

from Fragment_Match.utils.loss import ContrastiveLoss
from val import val


def train(
        # 参数设置
        weight,
        val_path='data/val',
        train_path='data/train',
        save_path='checkpoints',
        epoch=100,
        batch_size=8,
        learning_rate=0.01,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    print("使用{}进行训练".format(device))

    # 创建保存文件夹
    for i in range(100):
        result_path = os.path.join(save_path, 'result_' + str(i))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            save_path = result_path
            break

    # 加载数据
    val_data = LoadImagesAndLabels(val_path)
    train_data = LoadImagesAndLabels(train_path)
    train_data_size = len(train_data)
    val_data_size = len(val_data)

    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # 网络
    model = SRWO()
    if weight is not None:
        model.load_state_dict(torch.load(weight))
        print("使用权重文件:{}".format(weight))
    model = model.to(device)

    # 损失函数
    loss_fn = ContrastiveLoss()
    loss_fn = loss_fn.to(device)

    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Tensorboard
    writer = SummaryWriter(log_dir='logs')

    # 开始训练
    total_train_step = 0
    total_test_step = 0
    min_loss = float('inf')

    for i in range(epoch):
        model.train()
        pbar1 = tqdm(train_loader, desc="第{}轮训练".format(i + 1))
        for O_IMG, S_IMG, label, fit in pbar1:
            O_IMG = O_IMG.to(device)
            S_IMG = S_IMG.to(device)
            label = label.to(device)
            fit = fit.to(device)

            outputs = model(O_IMG, S_IMG)
            loss = loss_fn(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 100 == 0:
                writer.add_scalar('train_loss', loss.item(), total_train_step)

        # total_train_loss, total_train_accuracy, [TP_t, TN_t, FP_t, FN_t] = val(model, train_loader, loss_fn,
        #                                                                        train_data_size,
        #                                                                        device)
        total_val_loss, total_val_accuracy, [TP_v, TN_v, FP_v, FN_v] = val(model, val_loader, loss_fn, val_data_size,
                                                                           device)

        # print("整体训练集上的Loss：{}".format(total_train_loss))
        # print("整体训练集上的正确率：{}".format(total_train_accuracy))
        # print("总样本数：{}".format(train_data_size))
        # print("TP:{}, TN:{}, FP:{}, FN:{}".format(TP_t, TN_t, FP_t, FN_t))
        writer.add_scalar('val_loss', total_val_loss, i)
        writer.add_scalar('val_accuracy', total_val_accuracy, i)
        print("整体验证集上的Loss：{}".format(total_val_loss))
        print("整体验证集上的正确率：{}".format(total_val_accuracy))
        print("总样本数：{}".format(val_data_size))
        print("TP:{}, TN:{}, FP:{}, FN:{}".format(TP_v, TN_v, FP_v, FN_v))

        if total_val_loss < min_loss:
            min_loss = total_val_loss
            torch.save(model.state_dict(), os.path.join(save_path, "best.pt"))
            with open(os.path.join(save_path, "info.txt"), "w") as f:
                f.write("数据来源： {}".format(val_path) + "\n")
                f.write("预训练权重：{}".format(weight) + "\n")
                f.write("最佳权重来自第{}轮训练".format(i) + "\n")
                f.write("整体验证集上的Loss：{}".format(total_val_loss) + "\n")
                f.write("整体验证集上的正确率：{}".format(total_val_accuracy) + "\n")
                f.write("总样本数：{}".format(val_data_size) + "\n")
                f.write("TP:{}, TN:{}, FP:{}, FN:{}".format(TP_v, TN_v, FP_v, FN_v) + "\n")

    torch.save(model.state_dict(), os.path.join(save_path, "last.pt"))
    print("模型保存位置：", save_path)
    writer.close()


if __name__ == "__main__":
    train(
        weight=r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Fragment_Match\checkpoints\result_10\last.pt",
        val_path='data/PARTS7(721_redivided)/TRAIN/val',
        train_path='data/PARTS7(721_redivided)/TRAIN/train',
        save_path='checkpoints',
        epoch=100,
        batch_size=4,
        learning_rate=0.01,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
