import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian


def face_segmentation(
    img, scale_factor=1.3, min_neighbours=5, fs_gaussian_sigma=5, fs_dialtion_sigma=2, grab_cut_num_iter=10, model_size=65, grab_cut_mode=cv2.GC_INIT_WITH_MASK,
    canny_sigma=2, mcv_gaussian_sigma=2, canny_low_threshold=0.1, canny_high_threshold=0.2, num_dialation=1, fs_mcv_c1=1.0, fs_mcv_c2=1.0, fs_mcv_init_level="edges", fs_mcv_num_iter=35, fs_mcv_smoothing=1, fs_mcv_threshold=0
):
    face_cascade = cv2.CascadeClassifier('face_segmentation/haarcascade_frontalface_default.xml')
    gray_img = (rgb2gray(img) * 255).astype(np.uint8)
    faces = face_cascade.detectMultiScale(gray_img, scale_factor, min_neighbours)
    if len(faces) == 0:
        print("Found no faces. Will Hallucinate.")
        return np.zeros_like(gray_img)
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        xs = x  # max([x - int(w / 2), 0])
        ys = y  # max([y - int(h / 2), 0])
        xe = x + w  # int(3 * w / 2)
        ye = y + h  # int(3 * h / 2)
        mask = np.zeros_like(gray_img)
        mask[xs:xe, ys:ye] = 3
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut((img * 255).astype(np.uint8), mask=mask, rect=(x, y, w, h), bgdModel=bgdModel, fgdModel=fgdModel, iterCount=grab_cut_num_iter, mode=grab_cut_mode)
        return gaussian(mask, fs_gaussian_sigma)
