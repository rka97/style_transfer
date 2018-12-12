import skimage.io as io
import numpy as np
from .commonfunctions import *
import cv2


# img: RGB image to apply the filter t.
# abs_der: | d/dx (img(x)) |
# J[n] = I[n] + a^d (J[n-1] - I[n])
def recursive_filter(img, abs_der, h):
    im = np.copy(img)
    a = np.exp(-1 * np.sqrt(2) / h)
    var = np.power(a, abs_der)
    l, m, n = img.shape
    for i in range(1, m):
        for j in range(n):
            im[:, i, j] = img[:, i, j] + np.multiply(var[:, i], (im[:, i - 1, j] - img[:, i, j]))
    for i in range(m - 2, 0, -1):
        for j in range(n):
            im[:, i, j] = img[:, i, j] + np.multiply(var[:, i + 1], (im[:, i + 1, j] - img[:, i, j]))
    return im


# img: the image to denoise.
# sigma_r: controls variance over the signal's range.
# sigma_s: controls variance over the signal's spatial domain.
# denoises img through applying the domain transform then a RecursiveFilter. Returns denoised image.
def denoise(img, sigma_r=0.77, sigma_s=40):
    l, m, n = img.shape
    # using finite diffrence to get partial derivative
    dIdx = np.abs(np.diff(img, 1, 1))  # diff(A,1,1) works on successive elements in the columns of A and returns a p-by-(m-1) difference matrix.
    dIdy = np.abs(np.diff(img, 1, 0))  # diff(A,1,2) works on successive elements in the rows of A and returns a (p-1)-by-m difference matrix.
    derx = np.zeros((l, m))
    dery = np.zeros((l, m))
    derx[:, 1:m] = np.sum(dIdx, axis=2)
    dery[1:l, :] = np.sum(dIdy, axis=2)
    # horizontal and vertical derivatives
    dhdx = (1 + sigma_s / sigma_r * derx)
    dvdy = np.transpose((1 + sigma_s / sigma_r * dery))
    # to get ct, we integrate, not needed in case of using the recursive filter
    # cth = np.cumsum(dhdx,2)
    # ctv = np.cumsum(dvdy,1)
    const = sigma_s * np.sqrt(3) / np.sqrt(4**3 - 1)
    t_img = np.copy(img)
    for i in range(3):  # 3 is the no of iterations usually used, we could change it
        sigma_h = const * 2**(3 - i - 1)
        t_img = recursive_filter(t_img, dhdx, sigma_h)
        t_img = np.transpose(t_img, axes=(1, 0, 2))  # (m,l,n)
        t_img = recursive_filter(t_img, dvdy, sigma_h)
        t_img = np.transpose(t_img, axes=(1, 0, 2))  # (l, m, n)
    return t_img


# def main():
#     img = io.imread('../images/cow.jpg') / 255.0
#     original = np.copy(img)
#     # the opencv function
#     # dst = cv2.edgePreservingFilter(src, flags=1, sigma_s=60, sigma_r=0.4)
#     # typ =np.info(img.dtype).max  #in case of a binary image
#     img = img.astype(np.float32)
#     img_bilateral = cv2.bilateralFilter(img, 5, 50, 50)
#     denoised_img = denoise(img)
#     show_images([img, img_bilateral, denoised_img])

# main()
