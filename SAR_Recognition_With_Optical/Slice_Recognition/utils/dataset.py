# -*- coding: utf-8 -*-
# @Time    : 2024/3/12 0012 22:22
# @Author  : Ronin
# @File    : split_dataset.py
# @Software: PyCharm

from PIL import Image


def load_img(img_path):
    img = Image.open(img_path).convert('L')
    return img


def split_img(img, row, column):
    w, h = img.size
    w_len = w // column
    h_len = h // row
    im_list = []
    for i in range(row):
        lis = []
        for j in range(column):
            im = img.crop((w_len * j,
                           h_len * i,
                           min(w, w_len * (j + 1)),
                           min(h, h_len * (i + 1))
                           )
                          )
            # im.save("{}_{}.jpg".format(i + 1, j + 1))
            lis.append(im)
        im_list.append(lis)
    return im_list


if __name__ == "__main__":
    path = r"H:\Pycharm_Project\SAR_Recognition_With_Optical\Slice_Recognition\data\Optical\Air\Air_1.jpg"
    img = Image.open(path)
    split_img(img, 4, 1)
