import skimage.io as io
import numpy as np
from sklearn.mixture import GaussianMixture
import graph_tool.all as gt


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
    # print(im)
    io.imshow(im)
    io.show()
    # weights, means, covariances
    weights_per_alpha = np.array([gmm_bk.weights_, gmm_fr.weights_])
    means_per_alpha = np.array([gmm_bk.means_, gmm_fr.means_])
    covariances_per_alpha = np.array([gmm_bk.covariances_, gmm_fr.covariances_])
    return weights_per_alpha, means_per_alpha, covariances_per_alpha


# bg is a boolean array with bg.shape[0] = img.shape[0] and bg.shape[1] = img.shape[1], bg=0 => pixel is bg, otherwise unknown.
def GrabCut(img, trimap):
    l, m, n = img.shape
    # trimap = np.zeros((l, m))
    # trimap[bg == 0] = 0  # Background => 0
    # trimap[bg == 1] = -1  # Everything other than the background is unknown
    # pi, mu, sigma = GMM(img, trimap.reshape(l * m))
    # D_bias = - np.log(pi) + 0.5 * np.log(np.linalg.det(sigma))
    # z = np.reshape(img, (l * m, n))
    pixel_indices = np.reshape(np.arange(0, l * m, 1), (l, m))
    # print(pixel_indices[0:2, 0:2])
    # U_n = D_n + V
    # D_n = D_bias[alpha_n, k_n] + 0.5 * (z_n - mu(alpha_n, k_n)).T * (sigma(alpha_n, k_n))^-1 * (z_n - mu(alpha_n, k_n)
    G = gt.Graph(directed=False)
    G.add_vertex(l * m + 2)
    S = G.vertex(l * m)
    T = G.vertex(l * m + 1)
    cap = G.new_edge_property("double")
    for i in range(l):
        for j in range(m):
            current_index = i * m + j
            current_vertex = G.vertex(current_index)
            neighbors = pixel_indices[i - 1:i + 2, j - 1:j + 2]
            if (i > 0) and (j > 0):
                neighbors = pixel_indices[i - 1:i + 2, j - 1:j + 2]
            elif (i == 0) and (j > 0):
                neighbors = pixel_indices[i: i + 2, j - 1: j + 2]
            elif (i == 0) and (j == 0):
                neighbors = pixel_indices[i:i + 2, j:j + 2]
            else:
                neighbors = pixel_indices[i - 1:i + 2, j - 1:j + 2]
            neighbors = np.reshape(neighbors, -1)
            for neighbor_index in neighbors:
                neighbor_vertex = G.vertex(neighbor_index)
                e = G.add_edge(current_vertex, neighbor_vertex)
                cap[e] = current_index * neighbor_index
    print(cap)
    return img


def test():
    img = io.imread('../images/gmm_test.png')
    img = (img.astype(np.float)) / 255.0
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
    # print(trimap)
    GrabCut(img, trimap)
