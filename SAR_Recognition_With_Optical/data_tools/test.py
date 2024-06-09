import cv2
import os


def del_labels_lin1(file_name):
    labels = []
    with open(file_name, "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines)):
            lin = lines[i].split()
            lin.append(str(0))
            labels.append(" ".join(lin))
    with open(file_name, "w") as file:
        for label in labels:
            file.write(label+'\n')


if __name__ == '__main__':
    labels_path = r"H:\YOLO-Project\data_tools\DataTxt"
    dirs = os.listdir(labels_path)
    for n in dirs:
        del_labels_lin1(labels_path+'/'+n)


