from numpy.linalg import eig
from .commonfunctions import *
from sklearn.feature_extraction.image import extract_patches
from numpy.linalg import multi_dot
import cv2
from numpy import pi, exp, sqrt


def gaussian_kernel(n):
    # n must be odd number
    k = round((n-1)/2)  # n=2k+1 => k=(n-1)/2
    # generate a n*n gaussian kernel with mean=0 and sigma = s
    s = 1
    # create one vector of gaussian distribution
    probs = [exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k, k+1)]
    kernel = np.outer(probs, probs)  # construct 2d-gaussian kernel
    kernel = kernel.reshape(-1)  # vectorize it
    kernel = np.diag(kernel)  # make it diagonal matrix
    return kernel


def get_gradient(patch):
    gx = cv2.Sobel(np.float32(patch), cv2.CV_32F, 1, 0).reshape(-1)
    gy = cv2.Sobel(np.float32(patch), cv2.CV_32F, 0, 1).reshape(-1)
    G = np.zeros((gy.size, 2))
    G[:, 0] = gx
    G[:, 1] = gy
    # G = np.concatenate((gx,gy), axis=1)
    return G


def edge_detection(content, n):
    content = rgb2gray(content)
    patches = extract_patches(content, patch_shape=(n, n), extraction_step=1)
    patches = patches.reshape((-1, n, n))
    W = gaussian_kernel(n)
    strength_threshold = 0.04
    coherent_threshold = 0.5
    img = np.zeros(patches.shape[0])
    for k in range(0, patches.shape[0]):  # patches.shape[0]=>patches count
        # for each patch -corresponding to a pixel- calc gradient for all pixels
        Gk = get_gradient(patches[k])
        GWG = multi_dot([Gk.T, W, Gk])
        e_val, e_vect = eig(GWG)
        if e_val[0] > e_val[1]:
            largest = 0  # indx, = np.where(e_val == largest_lambda)
        else:
            largest = 1
        x, y = e_vect[largest, 0], e_vect[largest, 1]  # e_vect corresponding to largest e_val
        if x != 0:
            angle = math.degrees(math.atan(y/x))
        else:
            angle = 90
        strength = math.sqrt(e_val[largest])

        dominator = float(sqrt(e_val[largest]) + sqrt(e_val[1-largest]))
        if dominator != 0:
            coherent = (sqrt(e_val[largest])-sqrt(e_val[1-largest]))/dominator
        if strength >= strength_threshold and coherent >= coherent_threshold:
            img[k] = strength
    img = img.reshape(int(sqrt(patches.shape[0])), int(sqrt(patches.shape[0])))
    return img


# def main():
#     IM_SIZE = 400
#     content = io.imread('../paper_images/Spadena_Witch_House.jpg') / 255.0
#     content = (cv2.resize(content, (IM_SIZE, IM_SIZE))).astype(np.float32)
#     root_n = 5
#     edge = edge_detection(content, root_n)  # root_n should be odd number
#     show_images([edge])


# main()
