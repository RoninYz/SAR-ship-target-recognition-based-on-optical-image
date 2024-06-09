# -*- coding: utf-8 -*-
# @Time    : 2024/3/12 0012 22:52
# @Author  : Ronin
# @File    : compare.py
# @Software: PyCharm
import os

import numpy as np
import torch
from tqdm import tqdm

from Fragment_Match.predict import Predict
from utils.dataset import split_img, load_img


class Compare:  # 单个图片比较器
    def __init__(self, pred, row, column):
        self.pred = pred
        self.row = row
        self.column = column

    def __call__(self, o_img, s_img):
        o_img_frag = split_img(o_img, self.row, self.column)
        s_img_frag = split_img(s_img, self.row, self.column)
        outputs = []
        for i in range(self.row):
            for j in range(self.column):
                output = self.pred(o_img_frag[i][j], s_img_frag[i][j])
                outputs.append(output)

        # # 计算余弦相似度
        # vec1 = np.array(outputs)
        # vec2 = np.ones(len(outputs))
        #
        # cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        # return sum(outputs) / len(outputs)  # 平均相似度
        # return np.zeros(len(outputs)) if min(outputs) < 0.1 else np.array(outputs)
        return np.array(outputs)


class DeterMineCls:
    def __init__(self, cmp, o_img_dir):
        self.cmp = cmp
        self.cls = os.listdir(o_img_dir)
        self.o_img_paths = {}
        self.size = cmp.row * cmp.column
        for cl in self.cls:
            cl_path = os.path.join(o_img_dir, cl)
            img_names = os.listdir(cl_path)
            self.o_img_paths[cl] = [os.path.join(cl_path, img) for img in img_names]

    # def __call__(self, s_img):
    #     sims = []
    #     for cl in self.cls:
    #         sim = 0
    #         for o_img_path in self.o_img_paths[cl]:
    #             o_img = load_img(o_img_path)
    #             x = self.cmp(o_img, s_img)
    #             sim += (sum(x) - max(x) - min(x)) / (self.size - 2)
    #         sims.append(sim/len(self.o_img_paths[cl]))
    #     return self.cls[np.argmax(sims)], max(sims), sims

    def __call__(self, s_img):
        sims = []
        for cl in self.cls:
            sim = np.zeros(self.cmp.row * self.cmp.column)
            for o_img_path in self.o_img_paths[cl]:
                o_img = load_img(o_img_path)
                sim += self.cmp(o_img, s_img)
            sims.append((sim / len(self.o_img_paths[cl])).mean())
        return self.cls[np.argmax(sims)], max(sims), sims


if __name__ == '__main__':
    w = r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Fragment_Match\checkpoints\result_11\best.pt"
    optical_img_path = r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Slice_Recognition\data\Optical_crop(20)"
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred = Predict(w, dev)
    comp = Compare(pred, 5, 1)
    det_cls = DeterMineCls(comp, optical_img_path)

    sar_img_dirs = r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Slice_Recognition\data\test"
    sar_cls = os.listdir(sar_img_dirs)
    logs = []
    log = []
    log.append("权重文件:{}".format(w))
    log.append("光学图片来源: {}".format(optical_img_path))
    log.append("SAR图片来源:{}".format(sar_img_dirs))
    logs.append(log)
    for cl in sar_cls:
        log = []

        print("图片类别{}".format(cl))
        log.append("图片类别{}".format(cl))

        sar_img_dir = os.path.join(sar_img_dirs, cl)
        sar_img_names = os.listdir(sar_img_dir)
        pos_num = 0
        for sar_img_name in tqdm(sar_img_names):
            sar_img_path = os.path.join(sar_img_dir, sar_img_name)
            s_img = load_img(sar_img_path)
            cls, sim, sims = det_cls(s_img)
            log.append("{} {} {}".format(cls, sim, sims))
            if cls == cl:
                pos_num += 1
        print("准确率为:{}".format(pos_num / len(sar_img_names)))
        log.append("准确率为:{}".format(pos_num / len(sar_img_names)))
        logs.append(log)

    # 创建保存文件夹
    save_path = r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Slice_Recognition\result"
    for i in range(100):
        result_path = os.path.join(save_path, 'result_' + str(i))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            save_path = result_path
            break
    with open(os.path.join(save_path, 'logs.txt'), 'w') as f:
        for log in logs:
            for lin in log:
                f.write(lin + "\n")
        for log in logs:
            f.write(log[0] + " ")
            f.write(log[-1] + "\n")
