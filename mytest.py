# encoding=utf-8
import os
import cv2
import numpy as np
blue = [255, 144, 30, 255]
red = [0, 0, 255, 255]
white = [255, 255, 255, 255]


def changebgcolor(img_path, alpha_path, color, save_path):

    img1 = cv2.imread(img_path)
    img2 = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
    height, width, _ = img1.shape
    img3 = np.zeros((height, width, 4))
    img3[:, :, 0:3] = img1
    img3[:, :, 3] = img2
    for h in range(height):
        for w in range(width):
            c = img3[h, w]
            if c[3] < 50:
                img3[h, w] = color
    cv2.imwrite(save_path, img3)


if __name__ == '__main__':
    img_path = './test_data/test_images/mytest.jpg'
    alpha_path = './test_data/u2net_results/mytest.png'
    save_path = './test_data/u2net_results/0mytest_result.png'
    changebgcolor(img_path, alpha_path, blue, save_path)
