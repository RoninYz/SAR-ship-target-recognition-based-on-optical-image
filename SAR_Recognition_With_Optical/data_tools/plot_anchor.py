import cv2
import os


def read_labels(file_name):
    with open(file_name, "r") as file:
        lines = file.readlines()
        labels = []
        for lin in lines:
            lin = lin.split()
            label = []
            for i in range(0, 9):
                label.append(int(lin[i]))
            labels.append(label)
    return labels


def show_image(image, labels, image_name):
    image_name = "tools/Image/"+image_name+'.jpg'
    cv2.imwrite(image_name, image)


def draw_horizontal_anchor(image, labels):
    for lin in labels:
        lin = [int(x) for x in lin]
        cls, x1, y1, x2, y2, x3, y3, x4, y4 = lin
        cv2.line(image, [x1, y1], [x2,y2], [0, 255, 255], 1)
        cv2.line(image, [x2, y2], [x3, y3], [0, 255, 255], 1)
        cv2.line(image, [x3, y3], [x4, y4], [0, 255, 255], 1)
        cv2.line(image, [x4, y4], [x1, y1], [0, 255, 255], 1)
    return image


if __name__ == '__main__':
    image_path = r"H:\YOLO-Project\data_tools\Image\original"
    labels_path = r"H:\YOLO-Project\data_tools\DataTxt"
    save_path = r"H:\YOLO-Project\data_tools\Image\image_with_anchor"
    dirs = os.listdir(image_path)
    for i in range(0, len(dirs)):
        n = dirs[i]
        image = cv2.imread(image_path+'/'+n)
        labels = read_labels(labels_path+'/'+n[:-4]+'.txt')
        img = draw_horizontal_anchor(image, labels)
        cv2.imwrite(save_path+'/'+str(i)+'.jpg', image)


