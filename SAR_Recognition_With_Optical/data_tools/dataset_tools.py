import cv2
import os
import numpy as np
from tqdm import tqdm


# 获取目标文件夹下的所有图片名称
def read_image_name(dir_name):
    dirs = os.listdir(dir_name)
    image_name = []
    image_format = []
    for i in dirs:  # 循环读取路径下的文件并筛选输出
        if os.path.splitext(i)[1] in (".jpg", "png"):  # 筛选指定格式的图片文件
            image_name.append(os.path.splitext(i)[0])
            image_format.append(os.path.splitext(i)[1])
    return image_name, image_format


# 读一张图片的DOTA格式标签信息，返回标签内记录的目标坐标以及类别
def read_label(file_name):
    with open(file_name, "r") as file:
        lines = file.readlines()
        label = []
        for lin in lines:
            if not lin[0].isdigit():  # 如果不是数字就直接跳过
                continue
            lin = lin.split()  # 数据格式为  x1 y1 x2 y2 x3 y3 x4 y4 cls difficult
            label.append(lin[0:9])
    return label


# 得到有标注框的图片
def get_anchor(image, labels):
    for lin in labels:
        cls = lin[-1]
        x1, y1, x2, y2, x3, y3, x4, y4 = [int(x) for x in lin[0:-1]]
        cv2.line(image, [x1, y1], [x2,y2], [0, 255, 255], 2)
        cv2.line(image, [x2, y2], [x3, y3], [0, 255, 255], 2)
        cv2.line(image, [x3, y3], [x4, y4], [0, 255, 255], 2)
        cv2.line(image, [x4, y4], [x1, y1], [0, 255, 255], 2)
        #  照片   /添加的文字    /左下角坐标  /字体                            /字体大小 /颜色            /字体粗细
        cv2.putText(image, cls, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return image


def get_cropped_image(image, label, addition):
    # additon[0]为边框缩放比例, addition为最大增量,即 一个是 x*addition[0] 一个是 x+addition[1]
    cls = label[-1]
    coordinate = np.array([[int(label[i]), int(label[i+1])] for i in range(0, len(label)-1, 2)])
    # 获取旋转矩形的中心点、宽度、高度和旋转角度信息
    rect = cv2.minAreaRect(coordinate)
    center, size, angle = rect
    # 获取旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # 根据旋转矩阵进行旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    # 将旋转矩形区域裁剪出来
    size = [min(x*addition[0], x+addition[1]) for x in size]  # 矩形框略微放大，避免损失过多信息
    cropped_image = cv2.getRectSubPix(rotated_image, tuple(map(lambda x: int(x), size)),
                                      tuple(map(lambda x: int(x), center)))
    return cls, cropped_image


# 得到裁剪后的图片
def get_picture_cropped(image_path, labels_path, save_path, cls, addition):
    image_name, image_format = read_image_name(image_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    process = tqdm(range(0, len(image_name)), desc="Crop picture")
    for i in process:  # 一张图片一张图片的处理
        image = cv2.imread(os.path.join(image_path, image_name[i] + image_format[i]))
        labels = read_label(os.path.join(labels_path, image_name[i] + ".txt"))

        save_single_path = os.path.join(save_path, image_name[i])
        if not os.path.exists(save_single_path):  # 给每张图片创建一个文件夹
            os.makedirs(save_single_path)
        save_id = np.zeros(len(cls))
        for lin in labels:
            name, cropped_image = get_cropped_image(image, lin, addition)
            save_id[cls.index(name)] += 1
            cv2.imwrite(os.path.join(save_single_path, name + "_" + str(save_id[cls.index(name)]) + ".jpg"), cropped_image)


# 根据原图和标签信息画坐标框并保存
def get_picture_with_anchor(image_path, labels_path, save_path):
    image_name, image_format = read_image_name(image_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    process = tqdm(range(0, len(image_name)), desc='Get picture with anchor')
    for i in process:
        image = cv2.imread(os.path.join(image_path, image_name[i] + image_format[i]))
        label = read_label(os.path.join(labels_path, image_name[i] + ".txt"))
        img = get_anchor(image, label)
        cv2.imwrite(os.path.join(save_path, image_name[i] + '.jpg'), img)  # 统一保存为jpg格式


# 进行不同数据集之间的格式转换
def modify_labels_dota(labels_path, save_path, data_format, cls):
    dirs = os.listdir(labels_path)
    labels_name = []  # 保存所有标签文件路径
    for i in dirs:  # 循环读取路径下的文件并筛选输出
        if os.path.splitext(i)[1] == ".txt":  # 筛选指定格式的图片文件
            labels_name.append(i)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    process = tqdm(labels_name, desc='Modify Label to Dota format')

    if data_format == 'DOTA_Label':
        for file_name in process:  # 处理每个标签文件
            with open(os.path.join(labels_path, file_name), "r") as file:
                lines = file.readlines()
            with open(os.path.join(save_path, file_name), "w") as file:
                for lin in lines:
                    file.write(lin)

    elif data_format == 'summer_competition':  # 数据格式为：x1 y1 x2 y2 x3 y3 x4 y4 cls
        for file_name in process:  # 处理每个标签文件
            with open(os.path.join(labels_path, file_name), "r") as file:
                lines = file.readlines()
                label = []
                for lin in lines:
                    if not lin[0].isdigit():  # 如果不是数字就直接跳过
                        continue
                    label.append(lin.split())
            label = [lin + ["0"] for lin in label]  # 格式转换
            with open(os.path.join(save_path, file_name), "w") as file:
                for lin in label:
                    file.write(" ".join(lin) + "\n")


if __name__ == '__main__':
    image_path = r"H:\YOLO-Project\data_tools\train_data\Original\Image"
    original_label_path = r"H:\YOLO-Project\data_tools\train_data\Original\Label"
    cls = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]

    modify_label_save_path = r"H:\YOLO-Project\data_tools\train_data\DOTA_Label"
    image_with_anchor_save_path = r"H:\YOLO-Project\data_tools\train_data\Image_with_anchor"
    cropped_image_save_path = r"H:\YOLO-Project\data_tools\train_data\Cropped_Image"

    # modify_labels_dota(original_label_path, modify_label_save_path, 'summer_competition', cls)
    # get_picture_with_anchor(image_path, modify_label_save_path, image_with_anchor_save_path)
    get_picture_cropped(image_path, modify_label_save_path, cropped_image_save_path, cls, [1.2, 40])




