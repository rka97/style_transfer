import skimage.io as io
from skimage.color import rgb2gray
import numpy as np
from sklearn.mixture import GaussianMixture
import cv2
import maxflow


def init_GMM(img, trimap):
    l, m, n = img.shape
    img_bk = np.zeros((np.count_nonzero(trimap == 0), n))
    # print(img_bk.shape)
    img_fr = np.zeros((np.count_nonzero(trimap != 0), n))
    img2d = img.reshape(l * m, n)
    # print(img_fr.shape)
    # print(img_bk.shape)
    # print(img2d.shape)
    # how to linearize this np.where(??)
    # this causes a problem as it creates one extra component (the 0 values) that affects the fitting =>done
    # another problem is allocating too many arrays, so that i could retrieve the indexes and create the k vector
    j = 0
    k = 0
    bk_indexes = np.zeros((img_bk.shape[0]))
    fr_indexes = np.zeros((img_fr.shape[0]))
    for i in range(l * m):
        if (trimap[i] == 0):
            img_bk[j, :] = img2d[i, :]
            bk_indexes[j] = i
            j += 1
        else:
            img_fr[k, :] = img2d[i, :]
            fr_indexes[k] = i
            k += 1
    return img_bk, img_fr, img2d, bk_indexes, fr_indexes


def GMM(img, trimap):
    img_bk, img_fr, img2d, bk_indexes, fr_indexes = init_GMM(img, trimap)
    l, m, n = img.shape
    # define 2 GMMs one for background, one for foreground each with k = 5
    # don't forget to create the k vector containing k values of all pixels, return covariances and mean =>done
    # should the convariance be regularized?
    # weights are set to default (kmeans)
    # creating and fitting 2 models for the background and forground
    gmm_fr = GaussianMixture(n_components=5, covariance_type="full")
    gmm_bk = GaussianMixture(n_components=5, covariance_type="full")
    gmm_fr = gmm_fr.fit(X=img_fr)
    gmm_bk = gmm_bk.fit(X=img_bk)
    labels_fr = gmm_fr.predict(X=img_fr)
    labels_bk = gmm_bk.predict(X=img_bk)
    probabilities_fr = gmm_fr.predict_proba(img2d)
    probabilities_bk = gmm_bk.predict_proba(img2d)
    K = np.zeros((l * m))
    # update k
    for i in range(img_fr.shape[0]):
        K[int(fr_indexes[i])] = labels_fr[i]
    # print (np.unique(K))
    for i in range(img_bk.shape[0]):
        K[int(bk_indexes[i])] = labels_bk[i]
    # print (np.unique(K))
    # for testing purpose
    # for i in range(l * m):
    #     if (K[i] == 1):
    #         img2d[i] = [1, 1, 1]
    #     elif (K[i] == 0):
    #         img2d[i] = [1, 0, 0]
    #     elif (K[i] == 2):
    #         img2d[i] = [0, 1, 0]
    #     elif (K[i] == 3):
    #         img2d[i] = [0, 0, 1]
    #     else:
    #         img2d[i] = [1, 1, 0]
    # img2d = img2d * 255.0  # need to handle cases where the image is binary...etc
    # img2d.astype(np.uint8)
    # im = img2d.reshape(l, m, n)
    # # # print(im)
    # io.imshow(im)
    # io.show()
    # weights, means, covariances
    weights_per_alpha = np.array([gmm_bk.weights_, gmm_fr.weights_])
    means_per_alpha = np.array([gmm_bk.means_, gmm_fr.means_])
    covariances_per_alpha = np.array([gmm_bk.covariances_, gmm_fr.covariances_])
    precisions_per_alpha = np.array([gmm_bk.precisions_, gmm_fr.precisions_])
    probabilities_per_alpha = np.array([probabilities_bk, probabilities_fr])
    return K, weights_per_alpha, means_per_alpha, covariances_per_alpha, precisions_per_alpha, probabilities_per_alpha


def build_energy_function(img, trimap):
    trimap = trimap.reshape(-1)
    K, pi, mu, sigma, inv_sigmas, Pr = GMM(img, trimap)
    mu = mu
    sigma = sigma
    inv_sigmas = inv_sigmas
    l, m, n = img.shape
    Pr = np.reshape(Pr, (2, l, m, 5))
    trimap = trimap.reshape((l, m))
    D = np.zeros((2, l, m))
    det_sigmas = np.linalg.det(sigma)
    D_bias = -1.0 * np.log(pi) + 0.5 * np.log(det_sigmas)
    for i in range(l):
        for j in range(m):
            k = int(K[i * m + j])
            if trimap[i][j] == 0:
                alpha = 0
            else:
                alpha = 1
            diff = img[i, j, :] - mu[alpha][k]
            # D[alpha, i, j] = D_bias[alpha, k] + 0.5 * np.dot(np.dot(np.transpose(diff), inv_sigmas[alpha, k]), diff)
            D[alpha, i, j] = - 1.0 * (np.log(pi[alpha, k]) + -1 * np.log(Pr[alpha, i, j, k]))
            # D[alpha, i, j] = Pr[alpha, i, j, k]
    return D


def GrabCut(img, trimap):
    D = build_energy_function(img, trimap)
    flattened_img = img.reshape((-1, 3))
    l, m, _ = img.shape
    D = np.reshape(D, (2, l, m))
    Beta = 0.0
    for i in range(l * m):
        for j in range(l * m):
            if (i == j):
                continue
            diff = flattened_img[i] - flattened_img[j]
            Beta += np.sum(diff * diff)
    Beta = Beta / (l * m)
    Beta = 1 / (2 * Beta)
    pixel_indices = np.reshape(np.arange(0, l * m, 1), (l, m))
    gr = maxflow.GraphFloat()
    nodes = gr.add_nodes(l * m)
    K = 0
    for i in range(1, l, 2):
        for j in range(1, m, 2):
            current_index = i * m + j
            neighbors = pixel_indices[max(0, i - 1):i + 2, max(0, j - 1):j + 2]
            curr_K = 0
            for neighbor_index in np.nditer(neighbors):
                if (neighbor_index == current_index):
                    continue
                diff = flattened_img[current_index] - flattened_img[neighbor_index]
                dist = np.linalg.norm(diff)
                capacity = 50 * np.exp(-1 * Beta * np.sum(diff))
                gr.add_edge(nodes[current_index], nodes[neighbor_index], capacity, capacity)
                curr_K += capacity
            K = max(K, curr_K)
    K = K + 1
    for i in range(l):
        for j in range(m):
            current_index = i * m + j
            if (trimap[i][j] == 0):  # (i, j) is BG.
                cap_src = 0
                cap_dst = K
            elif (trimap[i][j] == 1):  # (i, j) is FG.
                cap_src = K
                cap_dst = 0
            else:
                cap_src = D[0, i, j]
                cap_dst = D[1, i, j]
            gr.add_tedge(current_index, cap_src, cap_dst)
    gr.maxflow()
    sgm = gr.get_grid_segments(nodes)
    return sgm * 1.0


def test_GrabCut():
    img = cv2.resize(io.imread('../images/emilia2.jpg'), (400, 400))
    l, m, n = img.shape
    trimap = np.zeros((l, m))
    # start with trimap where everything out of the bounding box is 0(background), everything inside is -1(unknown),
    # and no 1s (forground)
    # assuming that foreground is everything in the boundingbox till we start iterating, then we are back to our default def
    # arbitry values for testing purpose
    x = 77
    y = 41
    w = 168
    h = 168
    trimap[x:x + w, y:y + h] = 1
    img_r = cv2.resize(img, (25, 25)) / 255.0
    trimap_r = cv2.resize(trimap, (25, 25))
    alphas = GrabCut(img_r, trimap_r) * 1.0
    # print(alphas)
    io.imshow(np.reshape(cv2.resize(alphas, (l, m)), (l, m, 1)) * cv2.resize(img, (l, m)) / 255.0)
    io.show()
    # bgdModel = np.zeros((1, 65), np.float64)
    # fgdModel = np.zeros((1, 65), np.float64)
    # mask = cv2.grabCut(img_r.astype(np.uint8), trimap_r.astype(np.uint8), (0, 0, 0, 0), bgdModel, fgdModel, 1, mode=cv2.GC_INIT_WITH_MASK)
    # print(mask)
    # alphas = cv2.resize(alphas, (l, m))
    # show_images([alphas, cv2.resize(alphas, (l, m))])

test_GrabCut()
