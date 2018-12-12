import skimage.io as io
import numpy as np
from sklearn.mixture import GaussianMixture


def init_GMM(img, trimap):
    l, m, n = img.shape
    img_bk = np.zeros((np.count_nonzero(trimap == 0), n))
    # print(img_bk.shape)
    img_fr = np.zeros((np.count_nonzero(trimap == 1), n))
    img2d = img.reshape(l*m, n)
    # print(img_fr.shape)
    # how to linearize this np.where(??)
    # this causes a problem as it creates one extra component (the 0 values) that affects the fitting =>done
    # another problem is allocating too many arrays, so that i could retrieve the indexes and create the k vector
    j = 0
    k = 0
    bk_indexes = np.zeros((img_bk.shape[0]))
    fr_indexes = np.zeros((img_fr.shape[0]))
    for i in range(l*m):
        if (trimap[i] == 0):
            img_bk[j, :] = img2d[i, :]
            bk_indexes[j] = i
            j += 1
        else:
            img_fr[k, :] = img2d[i, :]
            fr_indexes[k] = i
            k += 1
    return img_bk, img_fr, img2d, bk_indexes, fr_indexes


def GMM(img, trimap, K):
    img_bk, img_fr, img2d, bk_indexes, fr_indexes = init_GMM(img, trimap)
    l, m, n = img.shape
    # define 2 GMMs one for background, one for foreground each with k = 5
    # don't forget to create the k vector containing k values of all pixels, return covariances and mean =>done
    # should the convariance be regularized?
    # weights are set to default (kmeans)
    # creating and fitting 2 models for the background and forground
    gmm_fr = GaussianMixture(n_components=5, covariance_type="full", verbose=2)
    print('\n')
    gmm_bk = GaussianMixture(n_components=5, covariance_type="full", verbose=2)
    gmm_fr = gmm_fr.fit(X=img_fr)
    gmm_bk = gmm_bk.fit(X=img_bk)
    labels_fr = gmm_fr.predict(X=img_fr)
    labels_bk = gmm_bk.predict(X=img_bk)
    # update k
    for i in range(img_fr.shape[0]):
        K[int(fr_indexes[i])] = labels_fr[i]
    # print (np.unique(K))
    for i in range(img_bk.shape[0]):
        K[int(bk_indexes[i])] = labels_bk[i]
    # print (np.unique(K))
    # for testing purpose
    for i in range(l*m):
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
    # means_,covariances_
    return gmm_fr.means_, gmm_fr.covariances_, gmm_bk.means_, gmm_bk.covariances_


def main():
    img = io.imread('face_segmentation/a.png')
    img = (img.astype(np.float))/255.0
    l, m, n = img.shape
    K = np.zeros((l*m))
    # start with trimap where everything out of the bounding box is 0(background), everything inside is -1(unknown),
    # and no 1s (forground)
    # assuming that foreground is everything in the boundingbox till we start iterating, then we are back to our default def
    trimap = np.zeros((l, m))
    # arbitry values for testing purpose
    x = 0
    w = 50
    y = 0
    h = 50
    trimap[x:x+w, y:y+h] = 1
    trimap = trimap.reshape(l*m)
    # print(trimap)
    GMM(img, trimap, K)
main()
