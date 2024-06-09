# -*- coding: utf-8 -*-
# @Time    : 2024/3/15 0015 10:31
# @Author  : Ronin
# @File    : split_dataset.py
# @Software: PyCharm
import os
import shutil
import random

import torch
from torch.utils.data import Dataset


class TrainDataset:
    def __init__(self, src_path, dst_path, rate):
        self.src_path = src_path
        self.dst_path = dst_path
        self.rate = rate

    @staticmethod
    def create_dirs(path):
        if not os.path.exists(path):
            os.makedirs(path)
        dirs1 = ["train", "val", "test"]
        dirs2 = ["Label", "Optical", "SAR"]
        for dir1 in dirs1:
            for dir2 in dirs2:
                dir_path = os.path.join(path, dir1, dir2)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

    @staticmethod
    def split_list(lis, rate):
        lis_length = len(lis)
        length = [int(lis_length * x) for x in rate]
        total_length = 0
        split_lis = []
        for l in length:
            split_lis.append(lis[total_length:total_length + l - 1])
            total_length += l
        return split_lis

    def sta_data(self, data_path, rate):
        Label_path = os.path.join(data_path, "Label")
        Label_files = os.listdir(Label_path)
        data_dic = {"pos": [], "neg": []}
        for label in Label_files:
            path = os.path.join(Label_path, label)
            with open(path, "r") as f:
                fit = int(f.readline())
            if fit == 1:
                data_dic["pos"].append(label.strip(".txt"))
            else:
                data_dic["neg"].append(label.strip(".txt"))
        random.shuffle(data_dic["pos"])
        random.shuffle(data_dic["neg"])
        pos_num = len(data_dic["pos"])
        neg_num = len(data_dic["neg"])
        pos_list = self.split_list(data_dic["pos"], rate)
        neg_list = self.split_list(data_dic["neg"], rate)
        data_dic["total"] = data_dic["pos"] + data_dic["neg"]

        # 训练集和验证集要求正负样本1:1
        # length = min(len(pos_list[0]), len(neg_list[0]))
        data_dic["train"] = pos_list[0] + neg_list[0]
        random.shuffle(data_dic["train"])

        # length = min(len(pos_list[1]), len(neg_list[1]))
        data_dic["val"] = pos_list[1] + neg_list[1]
        random.shuffle(data_dic["val"])

        data_dic["test"] = pos_list[2] + neg_list[2]
        random.shuffle(data_dic["test"])

        return data_dic

    def split_dataset_to_train(self, src_path, dst_path, rate, logs=[]):
        self.create_dirs(dst_path)
        cls = os.listdir(src_path)
        cls_paths = []
        for cl1 in cls:
            cl1_dir = os.path.join(src_path, cl1)
            for cl2 in os.listdir(cl1_dir):
                cl2_dir = os.path.join(cl1_dir, cl2)
                cl3 = os.listdir(cl2_dir)
                for x in cl3:
                    cls_paths.append(os.path.join(cl2_dir, x))

        logs.append("数据集包括{}类目标,分别是:".format(len(cls)))
        logs.append(",".join(cls))
        print("数据集包括{}类目标".format(len(cls)))
        train_num = 0
        val_num = 0
        test_num = 0
        for i, cls_path in enumerate(cls_paths):
            print("正在处理类别{}".format(i + 1))
            dataset = self.sta_data(cls_path, rate)
            logs.append(
                "类别{}的总样本数为{},其中正样本{},负样本{}".format(i, len(dataset["total"]), len(dataset["pos"]),
                                                                    len(dataset["neg"])))

            for x in dataset["train"]:
                train_num += 1
                goal_path = os.path.join(dst_path, "train")
                path = os.path.join(cls_path, os.path.join("Label", x + ".txt"))
                shutil.copy(path, os.path.join(goal_path, os.path.join("Label", "P_{}.txt".format(train_num))))

                path = os.path.join(cls_path, os.path.join("Optical", x + ".jpg"))
                shutil.copy(path, os.path.join(goal_path, os.path.join("Optical", "P_{}.jpg".format(train_num))))

                path = os.path.join(cls_path, os.path.join("SAR", x + ".jpg"))
                shutil.copy(path, os.path.join(goal_path, os.path.join("SAR", "P_{}.jpg".format(train_num))))

            for x in dataset["val"]:
                val_num += 1
                goal_path = os.path.join(dst_path, "val")
                path = os.path.join(cls_path, os.path.join("Label", x + ".txt"))
                shutil.copy(path, os.path.join(goal_path, os.path.join("Label", "P_{}.txt".format(val_num))))

                path = os.path.join(cls_path, os.path.join("Optical", x + ".jpg"))
                shutil.copy(path, os.path.join(goal_path, os.path.join("Optical", "P_{}.jpg".format(val_num))))

                path = os.path.join(cls_path, os.path.join("SAR", x + ".jpg"))
                shutil.copy(path, os.path.join(goal_path, os.path.join("SAR", "P_{}.jpg".format(val_num))))

            for x in dataset["test"]:
                test_num += 1
                goal_path = os.path.join(dst_path, "test")
                path = os.path.join(cls_path, os.path.join("Label", x + ".txt"))
                shutil.copy(path, os.path.join(goal_path, os.path.join("Label", "P_{}.txt".format(test_num))))

                path = os.path.join(cls_path, os.path.join("Optical", x + ".jpg"))
                shutil.copy(path, os.path.join(goal_path, os.path.join("Optical", "P_{}.jpg".format(test_num))))

                path = os.path.join(cls_path, os.path.join("SAR", x + ".jpg"))
                shutil.copy(path, os.path.join(goal_path, os.path.join("SAR", "P_{}.jpg".format(test_num))))
        return logs

    @staticmethod
    def analyse_train_dataset(src_path, logs=[]):
        train_label_path = os.path.join(src_path, os.path.join("train", "Label"))
        val_label_path = os.path.join(src_path, os.path.join("val", "Label"))
        test_label_path = os.path.join(src_path, os.path.join("test", "Label"))

        paths = [train_label_path, val_label_path, test_label_path]
        pos_num = []
        neg_num = []
        num = []
        for path in paths:
            pos = 0
            for x in os.listdir(path):
                x_path = os.path.join(path, x)
                with open(x_path, "r") as f:
                    fit = int(f.readline())
                if fit == 1:
                    pos += 1
            num.append(len(os.listdir(path)))
            pos_num.append(pos)
            neg_num.append(len(os.listdir(path)) - pos)
        logs.append("训练集共{},其中正样本{}，负样本{}".format(num[0], pos_num[0], neg_num[0]))
        logs.append("验证集共{},其中正样本{}，负样本{}".format(num[1], pos_num[1], neg_num[1]))
        logs.append("测试集共{},其中正样本{}，负样本{}".format(num[2], pos_num[2], neg_num[2]))
        return logs

    def __call__(self):
        logs = self.split_dataset_to_train(self.src_path, self.dst_path, self.rate)
        logs = self.analyse_train_dataset(self.dst_path, logs)

        return logs


class ValDataset:
    def __init__(self, src_path, dst_path):
        self.src_path = src_path
        self.dst_path = dst_path

    @staticmethod
    def create_dirs(path):
        if not os.path.exists(path):
            os.makedirs(path)
        pos_path = path + '/pos/'
        neg_path = path + '/neg/'
        dirs1 = [pos_path, neg_path]
        dirs2 = ["Label", "Optical", "SAR"]
        for dir1 in dirs1:
            for dir2 in dirs2:
                dir_path = os.path.join(dir1, dir2)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
        return pos_path, neg_path

    def split_dataset_to_val(self):
        cls = os.listdir(self.src_path)
        logs = []
        logs.append("共{}个类别".format(len(cls)))
        print("共{}个类别".format(len(cls)))
        for i, cl in enumerate(cls):
            print("正在处理类别{}:{}".format(i, cl))
            label_path = os.path.join(self.src_path, os.path.join(cl, "Label"))
            Optical_path = os.path.join(self.src_path, os.path.join(cl, "Optical"))
            SAR_path = os.path.join(self.src_path, os.path.join(cl, "SAR"))
            labels = os.listdir(label_path)
            pos_path, neg_path = self.create_dirs(os.path.join(self.dst_path, cl))
            pos_num = 0
            neg_num = 0
            for label in labels:
                with open(os.path.join(label_path, label), "r") as f:
                    x = int(f.readline())
                    goal_path = pos_path if x == 1 else neg_path
                    pos_num += int(x == 1)
                    neg_num += int(x != 1)

                    path = os.path.join(label_path, label)
                    shutil.copy(path, os.path.join(goal_path, os.path.join("Label", "P_{}.txt".format(
                        pos_num if x == 1 else neg_num))))

                    path = os.path.join(Optical_path, os.path.splitext(label)[0] + ".jpg")
                    shutil.copy(path, os.path.join(goal_path, os.path.join("Optical", "P_{}.jpg".format(
                        pos_num if x == 1 else neg_num))))

                    path = os.path.join(SAR_path, os.path.splitext(label)[0] + ".jpg")
                    shutil.copy(path, os.path.join(goal_path, os.path.join("SAR", "P_{}.jpg".format(
                        pos_num if x == 1 else neg_num))))
            logs.append("共{}个正样例,{}个负样例".format(pos_num, neg_num))
        return logs

    def __call__(self):
        logs = self.split_dataset_to_val()
        return logs


def divide_data_to_train(src_path, dst_path, rate):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    div_train_data = TrainDataset(src_path, dst_path, rate)
    logs = div_train_data()
    for log in logs:
        print(log)
    with open(os.path.join(dst_path, "logs.txt"), "w") as f:
        for log in logs:
            f.write(log + "\n")
        f.write("训练集：验证集：测试集={}".format(rate))


def divide_data_to_val(src_path, dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    div_val_data = ValDataset(src_path, dst_path)
    logs = div_val_data()
    for log in logs:
        print(log)
    with open(os.path.join(dst_path, "logs.txt"), "w") as f:
        for log in logs:
            f.write(log + "\n")


if __name__ == '__main__':
    path = r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Fragment_Match\Original_Data\PARTS7"
    name = os.path.basename(path) + "(721_redivided)"
    print("--- {按比例分割训练、验证、测试数据集} ---")
    divide_data_to_train(src_path=path,
                         dst_path=os.path.join("data", name, "TRAIN"),
                         rate=[0.7, 0.2, 0.1])
    # print("--- {按样本正负分割数据集} ---")
    # divide_data_to_val(src_path=path,
    #                    dst_path=os.path.join("data", name, "VAL"), )
