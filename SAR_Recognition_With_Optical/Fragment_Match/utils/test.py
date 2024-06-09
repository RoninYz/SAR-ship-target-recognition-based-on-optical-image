# -*- coding: utf-8 -*-
# @Time    : 2024/3/11 0011 21:55
# @Author  : Ronin
# @File    : split_dataset.py
# @Software: PyCharm
import os
import random
import shutil


def create_dirs(path):
    dirs1 = ["train", "val", "test"]
    dirs2 = ["Label", "Optical_crop(20)", "SAR"]
    for dir1 in dirs1:
        for dir2 in dirs2:
            dir_path = os.path.join(path, dir1, dir2)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)


# op_names = [os.path.splitext(x)(0) for x in op_names]
def split_dataset(src_path, goal_path, train_rate):
    op_path = os.path.join(src_path, "output_op")
    sar_path = os.path.join(src_path, "output_sar")
    tag_path = os.path.join(src_path, "output_tags")
    names = [os.path.splitext(x)[0] for x in os.listdir(op_path)]
    pos_name = []
    neg_name = []
    for name in names:
        txt_path = os.path.join(tag_path, name + ".txt")
        with open(txt_path, "r") as f:
            tag = int(f.readline().strip())
            if tag == 1:
                pos_name.append(name)
            else:
                neg_name.append(name)
    random.shuffle(pos_name)
    random.shuffle(neg_name)
    neg_name = neg_name[0:len(pos_name)]  # 保证正样本和负样本1:1

    img_format = ".jpg"
    create_dirs(goal_path)
    train_dir = os.path.join(goal_path, "train")
    val_dir = os.path.join(goal_path, "val")
    test_dir = os.path.join(goal_path, "test")

    for name in names:
        shutil.copy(os.path.join(op_path, name + img_format), os.path.join(test_dir, "Optical_crop(20)"))
        shutil.copy(os.path.join(sar_path, name + img_format), os.path.join(test_dir, "SAR"))
        shutil.copy(os.path.join(tag_path, name + ".txt"), os.path.join(test_dir, "Label"))

    for i, name in enumerate(pos_name):
        goal_dir = train_dir if i < len(pos_name) * train_rate else val_dir
        shutil.copy(os.path.join(op_path, name + img_format), os.path.join(goal_dir, "Optical_crop(20)"))
        shutil.copy(os.path.join(sar_path, name + img_format), os.path.join(goal_dir, "SAR"))
        shutil.copy(os.path.join(tag_path, name + ".txt"), os.path.join(goal_dir, "Label"))

    for i, name in enumerate(neg_name):
        goal_dir = train_dir if i < len(neg_name) * train_rate else val_dir
        shutil.copy(os.path.join(op_path, name + img_format), os.path.join(goal_dir, "Optical_crop(20)"))
        shutil.copy(os.path.join(sar_path, name + img_format), os.path.join(goal_dir, "SAR"))
        shutil.copy(os.path.join(tag_path, name + ".txt"), os.path.join(goal_dir, "Label"))

    train_label_dir = os.path.join(train_dir, "Label")
    val_label_dir = os.path.join(val_dir, "Label")
    test_label_dir = os.path.join(test_dir, "Label")

    train_labels = os.listdir(train_label_dir)
    val_labels = os.listdir(val_label_dir)
    test_labels = os.listdir(test_label_dir)

    train_pos_num = 0
    for label in train_labels:
        with open(os.path.join(train_label_dir, label), "r") as f:
            train_pos_num += int(f.readline().strip())
    train_neg_num = len(train_labels) - train_pos_num

    val_pos_num = 0
    for label in val_labels:
        with open(os.path.join(val_label_dir, label), "r") as f:
            val_pos_num += int(f.readline().strip())
    val_neg_num = len(val_labels) - val_pos_num

    test_pos_num = 0
    for label in test_labels:
        with open(os.path.join(test_label_dir, label), "r") as f:
            test_pos_num += int(f.readline().strip())
    test_neg_num = len(test_labels) - test_pos_num

    with open(os.path.join(goal_path, "data_info.txt"), "w") as f:
        f.write("训练集共{}, 其中正样本{}, 负样本{}".format(len(train_labels), train_pos_num, train_neg_num) + "\n")
        f.write("验证集共{}, 其中正样本{}, 负样本{}".format(len(val_labels), val_pos_num, val_neg_num) + "\n")
        f.write("测试集共{}, 其中正样本{}, 负样本{}".format(len(test_labels), test_pos_num, test_neg_num) + "\n")
    print("训练集共{}, 其中正样本{}, 负样本{}".format(len(train_labels), train_pos_num, train_neg_num))
    print("验证集共{}, 其中正样本{}, 负样本{}".format(len(val_labels), val_pos_num, val_neg_num))
    print("测试集共{}, 其中正样本{}, 负样本{}".format(len(test_labels), test_pos_num, test_neg_num))


if __name__ == "__main__":
    src_dir = r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Fragment_Match\pairs"
    goal_dir = r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Fragment_Match\data"
    split_dataset(src_dir, goal_dir, 0.7)
# -*- coding: utf-8 -*-
# @Time    : 2024/3/15 0015 10:30
# @Author  : Ronin
# @File    : test.py
# @Software: PyCharm
