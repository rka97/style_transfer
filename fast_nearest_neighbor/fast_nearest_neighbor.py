import cv2
import collections
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches
from sklearn.neighbors import NearestNeighbors

def meansMat(mat):
    means = np.sum(mat, axis=0) / mat.shape[0]
    means = np.matmul(np.ones((mat.shape[0],1)), np.transpose(means.reshape(mat.shape[1],1)))
    return means

def coVarianceMat(mat):
#   removing mean to make the data with zero means so it varies around the origion
    means = meansMat(mat)
    mat = mat - means
    cov_mat = np.cov(mat.T)
    
    return cov_mat

def projectionMat(mat):
    new_mat = mat.copy()
    
    # creating covarianve matrix to perform get its eigen values and eigen vectors
    covariance_matrix = coVarianceMat(new_mat)
    
    # eigenvectors and eigenvalues of the covariance matrix
    eig_val_cov, eig_vec_cov = np.linalg.eig(covariance_matrix)

    # creating a dectionary of eigen values and egen vectors to order them respeactivly
    eig_val_vec = {}
    for i in range(eig_val_cov.shape[0]):
        eig_val_vec[eig_val_cov[i]] = eig_vec_cov[:,i]

    # ordering eigen values in decending order
    eig_val_vec = dict(collections.OrderedDict(sorted(eig_val_vec.items(), reverse=True)))

    i = 0
    for key in eig_val_vec:
        eig_val_cov[i] = key
        eig_vec_cov[:,i] = eig_val_vec[key]
        i = i+1

    # normlizing eigen values
    eig_val_cov = eig_val_cov / np.sum(eig_val_cov)
    
    k = 0
    summation = 0
    for i in range(0, eig_val_cov.shape[0]):
        if summation >= 0.95:
            break
        
        k = k + 1
        summation = summation + eig_val_cov[i]

    ep = np.zeros((k, new_mat.shape[1]))
    for i in range(k):
        ep[i] = eig_vec_cov[:,i]
    
    return ep

def pca(mat):
    new_mat = mat.copy()
    ep = projectionMat(new_mat)
    new_mat = new_mat - meansMat(new_mat)
    output = (np.matmul(ep, new_mat.T)).T
    return (output, ep)

def project(mat, ep):
    new_mat = mat.copy()
    new_mat = new_mat - meansMat(new_mat)
    return (np.matmul(ep, new_mat.T)).T

def test1(x, p):
    new_x = x.copy()
    new_p = p.copy()

    ep = projectionMat(new_p)

    new_x = new_x - meansMat(new_x)
    new_p = new_p - meansMat(new_p)

    x_prime = np.matmul(ep, new_x)
    p_prime = np.matmul(ep, new_p)

    print(x_prime.shape)
    print(p_prime.shape)

# PATCH_SIZES = np.array([33, 21, 13, 9])
# SAMPLING_GAPS = np.array([28, 18, 8, 5])
# def test2():
#     LMAX = 3
#     IM_SIZE = 400
#     content = cv2.resize(io.imread('../images/ocean_day.jpg'), (IM_SIZE, IM_SIZE)) / 255.0
#     content = content.astype(np.float32)
#     style = cv2.resize(io.imread('../images/autumn.jpg'), (IM_SIZE, IM_SIZE)) / 255.0
#     style = style.astype(np.float32)

#     style_arr = build_gaussian_pyramid(style, LMAX)
#     content_arr = build_gaussian_pyramid(content, LMAX)

#     current_style = style_arr[0]
#     style_patches = extract_patches(current_style, patch_shape=(13, 13, 3), extraction_step=8)
#     npatchx, npatchy, _, _, _, _ = style_patches.shape
#     npatches = npatchx * npatchy
#     style_patches = style_patches.reshape(-1, 13 * 13 * 3)

#     reduced_mat, ep = pca(style_patches)
#     print(style_patches.shape)
#     print(reduced_mat.shape)
#     print(ep.shape)

# test2()