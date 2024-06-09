import cv2
import numpy as np


def read_img(filename):
    img = cv2.imread(filename, 1)
    return img


def gaussian_blur(img):
    gaussian_img = cv2.GaussianBlur(img, (5, 5), 10)
    return gaussian_img


def gray_procession(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


def threshold_procession(img, threshold):
    _, threshold_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return threshold_img


def draw_shape(img1, img2):
    contours, hierarchy = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img2, contours, -1, (0, 0, 255), 1)
    # cv2.namedWindow('final_img', 1)
    # cv2.imshow('final_img', img2)
    # cv2.waitKey()
    cv2.imwrite("2.jpg", img2)


def main():
    filename = r'H:\YOLO-Project\data_tools\train_data\Cropped_Image\0003\S3_2.0.jpg'
    img = read_img(filename)
    gaussian_img = gaussian_blur(img)
    gray_img = gray_procession(gaussian_img)
    threshold_img = threshold_procession(gray_img, 50)
    draw_shape(threshold_img, img)


if __name__ == '__main__':
    main()
