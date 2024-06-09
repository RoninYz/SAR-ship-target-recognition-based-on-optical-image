# -*- coding: utf-8 -*-
# @Time    : 2024/3/6 0006 20:51
# @Author  : Ronin
# @File    : predict.py
# @Software: PyCharm
import os

import torch
from PIL import Image
from torchvision import transforms

from Fragment_Match.models.model import SRWO


class Predict(object):
    def __init__(self, wight, device):
        print("加载网络模型")
        self.model = SRWO()
        self.device = device
        self.model.to(self.device)

        self.model.load_state_dict(torch.load(wight))
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((100, 100))
                                             ])
        print("加载网络模型成功")

    def __call__(self, O_IMG, S_IMG):
        O_IMG = self.transform(O_IMG).unsqueeze(0)
        S_IMG = self.transform(S_IMG).unsqueeze(0)
        O_IMG = O_IMG.to(self.device)
        S_IMG = S_IMG.to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(O_IMG, S_IMG)
            return output.detach().cpu().numpy()[0][1]


def load_img(img_path):
    img = Image.open(img_path).convert('L')
    return img


if __name__ == "__main__":
    w = r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Fragment_Match\checkpoints\result_18\best.pt"
    O_IMG_path = r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Fragment_Match\PARTS3\Part1\Optical\Air_sub1_1&Air_sub1_1.jpg"
    S_IMG_path = r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Fragment_Match\PARTS3\Part1\Optical\Air_sub1_1&Air_sub1_1.jpg"
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred = Predict(w, dev)

    O_IMG = load_img(O_IMG_path)
    S_IMG = load_img(S_IMG_path)
    output = pred(O_IMG, S_IMG)
    print(output)
