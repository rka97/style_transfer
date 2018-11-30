import cv2
from color_transfer.commonfunctions import *
from color_transfer.color_transfer import *
LMAX = 3
SIZEX = 400
SIZEY = 400


def build_gaussian(img, L):
    img_arr = []
    img_arr.append(img)
    for i in range(L - 1):
        img_arr.append(cv2.pyrDown(img_arr[-1]))
    return img_arr


def main():
    content = cv2.resize(io.imread('ocean_day.jpg'), (SIZEX, SIZEY)) / 255.0
    style = cv2.resize(io.imread('autumn.jpg'), (SIZEX, SIZEY)) / 255.0
    show_images([content, style, color_transfer_lab(content, style), color_transfer_hm(content, style)])
    content = color_transfer_lab(content, style)
    segmentation_mask = np.ones((SIZEX, SIZEY))
    content_arr = build_gaussian(content, LMAX)
    style_arr = build_gaussian(style, LMAX)
    # prepare NN
    show_images(content_arr)
    show_images(style_arr)

main()
