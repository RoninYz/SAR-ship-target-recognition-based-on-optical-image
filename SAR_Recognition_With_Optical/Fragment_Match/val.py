# -*- coding: utf-8 -*-
# @Time    : 2024/3/6 0006 20:49
# @Author  : Ronin
# @File    : val.py
# @Software: PyCharm
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Fragment_Match.models.model import SRWO
from Fragment_Match.utils.dataloders import LoadImagesAndLabels
from Fragment_Match.utils.loss import ContrastiveLoss


def load_net(weight):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SRWO()
    model = model.to(device)
    loss_fn = ContrastiveLoss()
    loss_fn = loss_fn.to(device)
    if weight is not None:
        model.load_state_dict(torch.load(weight))
    return model, loss_fn, device


def val(model, val_loader, loss_fn, val_data_size, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    TP = 0  # 真正例， 真的预测成真的
    TN = 0  # 真反例， 假的预测成假的
    FP = 0  # 假正例， 假的预测成假真
    FN = 0  # 假反例， 真的预测成假的
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="测试数据集")
        for O_IMG, S_IMG, label, fit in pbar:
            O_IMG = O_IMG.to(device)
            S_IMG = S_IMG.to(device)
            fit = fit.to(device)
            label = label.to(device)

            outputs = model(O_IMG, S_IMG)
            loss = loss_fn(outputs, label)
            total_loss += loss.item()
            accuracy = (outputs.argmax(dim=1) == fit).sum()
            total_accuracy = total_accuracy + accuracy
            for x, y in zip(outputs.argmax(dim=1), fit):
                if x.item() == y.item():
                    if x.item() == 1:
                        TP += 1
                    else:
                        TN += 1
                else:
                    if x.item() == 1:
                        FP += 1
                    else:
                        FN += 1
    return total_loss, total_accuracy / val_data_size, [TP, TN, FP, FN]


class AnalyseDataset:
    def __init__(self, weight):
        self.model, self.loss_fn, self.device = load_net(weight)

    def load_data(self, path):
        data = LoadImagesAndLabels(path)
        data_size = len(data)
        loader = DataLoader(data, batch_size=1, shuffle=True)
        return loader, data_size

    def analyse_pos_neg(self, path, name):
        loader, data_size = self.load_data(path)
        total_test_loss, total_test_accuracy, [TP, TN, FP, FN] = val(self.model, loader, self.loss_fn, data_size,
                                                                     self.device)
        logs = "{}: 总样本数{}, 正确率{}".format(name, data_size, total_test_accuracy)
        return logs

    def analyse_test(self, path, name):
        loader, data_size = self.load_data(path)
        total_test_loss, total_test_accuracy, [TP, TN, FP, FN] = val(self.model, loader, self.loss_fn, data_size,
                                                                     self.device)

        logs = ["------{}------".format(name)]
        logs.append("Loss：{}".format(total_test_loss))
        logs.append("正确率：{}".format(total_test_accuracy))
        logs.append("总样本数：{}".format(data_size))
        logs.append("TP:{}, TN:{}, FP:{}, FN:{}".format(TP, TN, FP, FN))
        return logs


def analyse_pos_neg(weight, dirs1):
    analyse_tool = AnalyseDataset(weight)
    logs = ["共{}类目标".format(len(dirs1))]
    for i, path in enumerate(dirs1):
        print("正在测试类别{}".format(i + 1))
        logs.append("类别{}".format(i))
        pos_path = os.path.join(path, "pos")
        logs.append(analyse_tool.analyse_pos_neg(pos_path, "pos"))
        neg_path = os.path.join(path, "neg")
        logs.append(analyse_tool.analyse_pos_neg(neg_path, "neg"))
    for log in logs:
        print(log)


def analyse_test(weight, dirs1):
    analyse_tool = AnalyseDataset(weight)
    logs = ["共{}类目标".format(len(dirs1))]
    for i, path in enumerate(dirs1):
        print("正在测试类别{}".format(i + 1))
        logs.append("类别{}".format(i))
        logs.append(analyse_tool.analyse_test(path, os.path.basename(path)))
    for log in logs:
        print(log)


if __name__ == '__main__':
    weight = r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Fragment_Match\checkpoints\result_5\best.pt"
    # dirs1 = [r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Fragment_Match\data\PARTS4\VAL\Part1",
    #          r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Fragment_Match\data\PARTS4\VAL\Part2",
    #          r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Fragment_Match\data\PARTS4\VAL\Part3",
    #          r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Fragment_Match\data\PARTS4\VAL\Part4",
    #          r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Fragment_Match\data\PARTS4\VAL\Part5"]
    # print("----{测试正负样例}----")
    # analyse_pos_neg(weight, dirs1)

    dirs2 = [r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Fragment_Match\data\PARTS4\TRAIN\train",
             r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Fragment_Match\data\PARTS4\TRAIN\val",
             r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Fragment_Match\data\PARTS4\TRAIN\test"]
    print("----{测试训练、验证、测试数据集}----")
    analyse_test(weight, dirs2)
