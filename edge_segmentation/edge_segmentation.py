import cv2
from skimage import img_as_float
from numpy.linalg import eig
from .commonfunctions import *
from numpy import pi, exp, sqrt
from scipy import ndimage as ndi
from skimage.draw import polygon
from skimage.filters import gaussian
from skimage.filters import rank
from scipy.spatial import Delaunay
from numpy.linalg import multi_dot
from skimage.util import img_as_ubyte
from skimage.segmentation import chan_vese, morphological_chan_vese
from scipy.ndimage import binary_fill_holes
from skimage.morphology import watershed, disk, dilation
from sklearn.feature_extraction.image import extract_patches


def gaussian_kernel(n):
    # n must be odd number
    k = round((n - 1) / 2)  # n=2k+1 => k=(n-1)/2
    # generate a n*n gaussian kernel with mean=0 and sigma = s
    s = 1
    # create one vector of gaussian distribution
    probs = [exp(-z * z / (2 * s * s)) / sqrt(2 * pi * s * s) for z in range(-k, k + 1)]
    kernel = np.outer(probs, probs)  # construct 2d-gaussian kernel
    kernel = kernel.reshape(-1)  # vectorize it
    kernel = np.diag(kernel)  # make it diagonal matrix
    return kernel


def get_gradient(patch, n):
    gx = cv2.Sobel(np.float32(patch), cv2.CV_32F, 1, 0, ksize=n)
    gy = cv2.Sobel(np.float32(patch), cv2.CV_32F, 0, 1, ksize=n)
    G = np.stack((gx, gy), axis=-1)
    return G


def edge_detection(content, n=5, strength_threshold=0.04, coherence_threshold=0.5):
    content = rgb2gray(content)
    W = gaussian_kernel(n)

    G = get_gradient(content, n)
    patches = extract_patches(G, patch_shape=(n, n, 2), extraction_step=1)
    patches = patches.reshape((-1, n**2, 2))

    l = list(patches)
    img = np.zeros((len(l)))
    GWG = [multi_dot([Gk.T, W, Gk]) for Gk in l]

    eigen = [eig(GWGi) for GWGi in GWG]
    e_val, e_vect = zip(*eigen)
    e_val = np.asarray(e_val, dtype=np.float64)
    e_vect = np.asarray(e_vect, dtype=np.float64)
    for k in range(0, len(eigen)):
        if e_val[k][0] > e_val[k][1]:
            largest = 0  # indx, = np.where(e_val == largest_lambda)
        else:
            largest = 1
        x, y = e_vect[k][largest, 0], e_vect[k][largest, 1]  # e_vect corresponding to largest e_val
        if x != 0:
            angle = math.degrees(math.atan(y / x))
        else:
            angle = 90
        strength = math.sqrt(e_val[k][largest])

        dominator = float(sqrt(e_val[k][largest]) + sqrt(e_val[k][1 - largest]))
        if dominator != 0:
            coherent = (sqrt(e_val[k][largest]) - sqrt(e_val[k][1 - largest])) / dominator
        if strength >= strength_threshold and coherent >= coherence_threshold:
            img[k] = strength
    img = img.reshape(int(sqrt(patches.shape[0])), int(sqrt(patches.shape[0])))
    return img


def edge_segmentation(
    img, strength_threshold=8, coherence_threshold=0.5, mode=4,
    ch_ethreshold=0.8,
    ws_ethreshold=0.2, ws_mdisk_size=5, ws_mthreshold=20, ws_gdisk_size=2, ws_glevel_threshold=4,
    cv_ethreshold=0, cv_mu=0.1, cv_lamda_1=0.06, cv_lamda_2=1, cv_tol=1e-3, cv_max_iter=2000, cv_dt=0.52, cv_init_level_set="checkerboard",
    mcv_init_level_set="edges", mcv_c1=1.0, mcv_c2=1.0, mcv_max_iter=35, mcv_smoothing=1, mcv_sigma=5
):
    IM_SIZE = 400
    img = (cv2.resize(img, (IM_SIZE, IM_SIZE))).astype(np.float32)
    root_n = 5
    edges = edge_detection(img, root_n, strength_threshold=strength_threshold, coherence_threshold=coherence_threshold)  # root_n should be odd number #8-0.5
    final_image = np.zeros((img.shape[0], img.shape[1]))
    if mode == 0:
        # thresholding edges for convex hull with threshold to remove as much noise as possible
        edges[edges >= ch_ethreshold] = 1
        edges[edges < ch_ethreshold] = 0
        chull = convex_hull(edges)
        final_image[:chull.shape[0], :chull.shape[1]] = chull
        return final_image
    elif mode == 1:
        # thresholding edges for watershed on edges with low threshold to include as much edges as possible
        edges[edges >= ws_ethreshold] = 1
        edges[edges < ws_ethreshold] = 0
        watershed_edges_bin = watershed_edges(edges)
        final_image[:watershed_edges_bin.shape[0], :watershed_edges_bin.shape[1]] = watershed_edges_bin
        return final_image
    elif mode == 2:
        edge_chull = edges
        edge_watershed = edge_chull.copy()

        # thresholding edges for convex hull with threshold to remove as much noise as possible
        edge_chull[edge_chull >= ch_ethreshold] = 1
        edge_chull[edge_chull < ch_ethreshold] = 0
        # thresholding edges for watershed on edges with low threshold to include as much edges as possible
        edge_watershed[edge_watershed >= ws_ethreshold] = 1
        edge_watershed[edge_watershed < ws_ethreshold] = 0

        chull = convex_hull(edge_chull)
        watershed_edges_bin = watershed_edges(edge_watershed, ws_mdisk_size, ws_mthreshold, ws_gdisk_size, ws_glevel_threshold)
        watershed_cull = chull * watershed_edges_bin[:chull.shape[0], :chull.shape[1]]
        final_image[:watershed_cull.shape[0], :watershed_cull.shape[1]] = watershed_cull
        return final_image
    elif mode == 3:
        # Feel free to play around with the parameters to see how they impact the result
        edges[edges != cv_ethreshold] = 1

        cv_init_level_set = cv_init_level_set.split(',')
        cv_init_level = cv_init_level_set[0]
        if cv_init_level_set[0] == "edges":
            cv_init_level = edges
        elif cv_init_level_set[0] == "original gray":
            cv_init_level = rgb2gray(img)
            cv_init_level = cv_init_level[:edges.shape[0], :edges.shape[1]]
        elif cv_init_level_set[0] == "path":
            cv_init_level = io.imread(cv_init_level_set[1])
            cv_init_level = (cv2.resize(cv_init_level, (IM_SIZE, IM_SIZE))).astype(np.float32)
            cv_init_level = rgb2gray(cv_init_level)
            cv_init_level = cv_init_level[:edges.shape[0], :edges.shape[1]]

        cv = chan_vese(edges, mu=cv_mu, lambda1=cv_lamda_1, lambda2=cv_lamda_2, tol=cv_tol, max_iter=cv_max_iter,
                       dt=cv_dt, init_level_set=cv_init_level)

        mask = cv
        final_image[:mask.shape[0], :mask.shape[1]] = mask
        return final_image
    else:
        E = np.zeros((img.shape[0], img.shape[1]))
        E[:edges.shape[0], :edges.shape[1]] = edges

        mcv_init_level_set = mcv_init_level_set.split(',')
        mcv_init_level = mcv_init_level_set[0]
        if mcv_init_level_set[0] == "edges":
            mcv_init_level = E
        elif mcv_init_level_set[0] == "original gray":
            mcv_init_level = rgb2gray(img)
        elif mcv_init_level_set[0] == "path":
            mcv_init_level = io.imread(mcv_init_level_set[1])
            mcv_init_level = (cv2.resize(mcv_init_level, (IM_SIZE, IM_SIZE))).astype(np.float32)
            mcv_init_level = rgb2gray(mcv_init_level)

        mask = mcv_c1 * (gaussian(mcv_c2 * E + morphological_chan_vese(rgb2gray(img), iterations=mcv_max_iter, init_level_set=mcv_init_level, smoothing=mcv_smoothing), sigma=mcv_sigma))
        return mask


# -------------------------------------------------------------------------------------
# using graham scan algorithm to construct convex hull and fill it
def right_turn(p1, p2, p3):
    if (p3[1] - p1[1]) * (p2[0] - p1[0]) >= (p2[1] - p1[1]) * (p3[0] - p1[0]):
        return False
    return True


def graham_scan(p):
    p.sort()			# Sort the set of points
    l_upper = [p[0], p[1]]		# Initialize upper part
    # Compute the upper part of the hull
    for i in range(2, len(p)):
        l_upper.append(p[i])
        while len(l_upper) > 2 and not right_turn(l_upper[-1], l_upper[-2], l_upper[-3]):
            del l_upper[-2]
    l_lower = [p[-1], p[-2]]  # Initialize the lower part
    # Compute the lower part of the hull
    for i in range(len(p) - 3, -1, -1):
        l_lower.append(p[i])
        while len(l_lower) > 2 and not right_turn(l_lower[-1], l_lower[-2], l_lower[-3]):
            del l_lower[-2]
    del l_lower[0]
    del l_lower[-1]
    l = l_upper + l_lower		# Build the full hull
    return np.array(l)


def convex_hull(img):
    image = img.copy()
    points = []
    for j in range(image.shape[1]):
        for i in range(image.shape[0]):
            if image[i, j] == 1:
                points.append((j, i))

    l = graham_scan(points)
    r = l[:, 1]
    c = l[:, 0]
    rr, cc = polygon(r, c)
    image[rr, cc] = 1
    return image


# -------------------------------------------------------------------------------------
# using skimage watershed algorithm with histogram thresholding to remove background regions from image
def watershed_edges(img, ws_mdisk_size=5, ws_mthreshold=20, ws_gdisk_size=2, ws_glevel_threshold=4):
    image = img.copy()
    image = img_as_ubyte(image)
    markers = rank.gradient(image, disk(ws_mdisk_size)) < ws_mthreshold
    markers = ndi.label(markers)[0]
    gradient = rank.gradient(image, disk(ws_gdisk_size))
    labels = watershed(gradient, markers)
    # show_images([image, labels], ["edges", "edges' labels"])
    labels[labels <= ws_glevel_threshold] = 1
    labels[labels > ws_glevel_threshold] = 0
    labels = np.invert(labels)
    labels[labels == -1] = 1
    labels[labels == -2] = 0
    return labels


def test_ed():
    IM_SIZE = 400
    content = io.imread('../images/cow.jpg') / 255.0
    content = (cv2.resize(content, (IM_SIZE, IM_SIZE))).astype(np.float32)
    filled_edges = edge_segmentation(content)
    show_images([content, filled_edges])
