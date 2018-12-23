import skimage.io as io
from skimage.color import rgb2gray
import numpy as np
from sklearn.mixture import GaussianMixture
import graph_tool.all as gt
import cv2


def init_GMM(img, trimap):
    l, m, n = img.shape
    img_bk = np.zeros((np.count_nonzero(trimap == 0), n))
    # print(img_bk.shape)
    img_fr = np.zeros((np.count_nonzero(trimap == 1), n))
    img2d = img.reshape(l * m, n)
    # print(img_fr.shape)
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
    K = np.zeros((l * m))
    # update k
    for i in range(img_fr.shape[0]):
        K[int(fr_indexes[i])] = labels_fr[i]
    # print (np.unique(K))
    for i in range(img_bk.shape[0]):
        K[int(bk_indexes[i])] = labels_bk[i]
    # print (np.unique(K))
    # for testing purpose
    for i in range(l * m):
        if (K[i] == 1):
            img2d[i] = [1, 1, 1]
        elif (K[i] == 0):
            img2d[i] = [1, 0, 0]
        elif (K[i] == 2):
            img2d[i] = [0, 1, 0]
        elif (K[i] == 3):
            img2d[i] = [0, 0, 1]
        else:
            img2d[i] = [1, 1, 0]
    # img2d = img2d*255.0  # need to handle cases where the image is binary...etc
    # img2d.astype(np.uint8)
    im = img2d.reshape(l, m, n)
    # # print(im)
    # io.imshow(im)
    # io.show()
    # weights, means, covariances
    weights_per_alpha = np.array([gmm_bk.weights_, gmm_fr.weights_])
    means_per_alpha = np.array([gmm_bk.means_, gmm_fr.means_])
    covariances_per_alpha = np.array([gmm_bk.covariances_, gmm_fr.covariances_])
    return K, weights_per_alpha, means_per_alpha, covariances_per_alpha


def build_energy_function(img, trimap):
    trimap = trimap.reshape(-1)
    K, pi, mu, sigma = GMM(img, trimap)
    l, m, n = img.shape
    trimap = trimap.reshape((l, m))
    D = np.zeros((2, l, m))
    inv_sigmas = np.zeros_like(sigma)
    det_sigmas = np.zeros_like(pi)
    for i in range(inv_sigmas.shape[0]):
        for j in range(inv_sigmas.shape[1]):
            inv_sigmas[i, j] = np.linalg.inv(sigma[i, j])
            det_sigmas[i, j] = np.linalg.det(sigma[i, j])
    D_bias = - np.log(pi) + 0.5 * np.log(det_sigmas)
    for i in range(l):
        for j in range(m):
            k = int(K[i * m + j])
            if trimap[i][j] == 0:
                alpha = 0
            else:
                alpha = 1
            diff = img[i, j, :] - mu[alpha][k]
            # print(D_bias[alpha, k], diff.shape, (inv_sigmas[alpha, k]).shape)
            D[alpha, i, j] = - D_bias[alpha, k] + 0.5 * np.dot(np.dot(np.transpose(diff), inv_sigmas[alpha, k]), diff)
    return D



def GrabCut(img, trimap):
    D = build_energy_function(img, trimap)
    gray_img = rgb2gray(img)
    flattened_img = gray_img.reshape(-1).astype(np.float32)
    l, m = gray_img.shape
    Beta = 0.0
    for i in range(l * m):
        for j in range(l * m):
            Beta += np.power((flattened_img[i] - flattened_img[j]), 2)
    Beta = l * m / (2 * Beta)
    pixel_indices = np.reshape(np.arange(0, l * m, 1), (l, m))
    # U_n = D_n + V
    # D_n = D_bias[alpha_n, k_n] + 0.5 * (z_n - mu(alpha_n, k_n)).T * (sigma(alpha_n, k_n))^-1 * (z_n - mu(alpha_n, k_n)
    G = gt.Graph(directed=True)
    G.add_vertex(l * m + 2)
    S = G.vertex(l * m)
    T = G.vertex(l * m + 1)
    cap = G.new_edge_property("double")
    K = 0
    for i in range(l):
        for j in range(m):
            current_index = i * m + j
            current_vertex = G.vertex(current_index)
            neighbors = pixel_indices[max(0, i - 1):i + 2, max(0, j - 1):j + 2]
            curr_K = 0
            for neighbor_index in np.nditer(neighbors):
                if (neighbor_index == current_index):
                    continue
                neighbor_vertex = G.vertex(neighbor_index)
                dist = max(np.power(flattened_img[current_index] - flattened_img[neighbor_index], 2), 0.0001)
                capacity = (50.0 / np.sqrt(dist)) * np.exp(-1 * Beta * dist)
                e = G.add_edge(current_vertex, neighbor_vertex)
                cap[e] = capacity
                curr_K += capacity
            K = max(K, curr_K)
    K = K + 1
    for i in range(l):
        for j in range(m):
            current_index = i * m + j
            current_vertex = G.vertex(current_index)
            src_edge = G.add_edge(S, current_vertex)
            dst_edge = G.add_edge(current_vertex, T)
            if (trimap[i][j] == 0):  # (i, j) is BG.
                cap[src_edge] = 0
                cap[dst_edge] = K
            elif (trimap[i][j] == 1):  # (i, j) is FG.
                cap[src_edge] = K
                cap[dst_edge] = 0
            else:
                cap[src_edge] = D[0, i, j]
                cap[dst_edge] = D[1, i, j]
    res = gt.boykov_kolmogorov_max_flow(G, S, T, cap)
    part = gt.min_st_cut(G, S, cap, res)
    alphas = np.zeros(l * m)
    # max_flow = sum(res[e] for e in tgt.in_edges())
    for v in G.vertices():
        if part[v] == 1:
            vertex_index = int(v)
            if (vertex_index < l * m):
                alphas[vertex_index] = 1
    return alphas


def test_GrabCut():
    img = io.imread('../images/emilia2.jpg')
    img = (img.astype(np.float))
    img = cv2.resize(img, (200, 200)) / 255.0
    l, m, n = img.shape
    # start with trimap where everything out of the bounding box is 0(background), everything inside is -1(unknown),
    # and no 1s (forground)
    # assuming that foreground is everything in the boundingbox till we start iterating, then we are back to our default def
    trimap = np.zeros((l, m))
    # arbitry values for testing purpose
    x = 0
    w = 50
    y = 0
    h = 50
    trimap[x:x + w, y:y + h] = 1
    alphas = GrabCut(rgb2gray(img_r), trimap_r)
    print(alphas)
    io.imshow(cv2.resize(alphas, (l, m)))
    io.show()
    # alphas = cv2.resize(alphas, (l, m))
    # show_images([alphas, cv2.resize(alphas, (l, m))])

test_GrabCut()