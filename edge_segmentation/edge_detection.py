import cv2
from skimage import img_as_float
from numpy.linalg import eig
from .commonfunctions import *
from numpy import pi, exp, sqrt
from scipy import ndimage as ndi
from skimage.draw import polygon
from skimage.filters import rank
from scipy.spatial import Delaunay
from numpy.linalg import multi_dot
from skimage.util import img_as_ubyte
from skimage.segmentation import chan_vese
from scipy.ndimage import binary_fill_holes
from skimage.morphology import watershed, disk, dilation
from sklearn.feature_extraction.image import extract_patches


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
            angle = math.degrees(math.atan(y/x))
        else:
            angle = 90
        strength = math.sqrt(e_val[k][largest])

        dominator = float(sqrt(e_val[k][largest]) + sqrt(e_val[k][1-largest]))
        if dominator != 0:
            coherent = (sqrt(e_val[k][largest])-sqrt(e_val[k][1-largest])) / dominator
        if strength >= strength_threshold and coherent >= coherence_threshold:
            img[k] = strength
    img = img.reshape(int(sqrt(patches.shape[0])), int(sqrt(patches.shape[0])))
    return img


def edge_segmentation(img, mode=4):
    IM_SIZE = 400
    img = (cv2.resize(img, (IM_SIZE, IM_SIZE))).astype(np.float32)
    root_n = 5
    edges = edge_detection(img, root_n, strength_threshold=8, coherence_threshold=0.5)  # root_n should be odd number #8-0.5
    final_image = np.zeros((img.shape[0], img.shape[1]))
    if mode == 0:
        # thresholding edges for convex hull with threshold to remove as much noise as possible
        edges[edges >= 0.8] = 1
        edges[edges < 0.8] = 0
        return convex_hull(edges)
        final_image[:chull.shape[0], :chull.shape[1]] = chull
        return final_image
    elif mode == 1:
        # thresholding edges for watershed on edges with low threshold to include as much edges as possible
        edges[edges >= 0.2] = 1
        edges[edges < 0.2] = 0
        watershed_edges_bin = watershed_edges(edges)
        final_image[:watershed_edges_bin.shape[0], :watershed_edges_bin.shape[1]] = watershed_edges_bin
        return final_image
    elif mode == 2:
        edge_chull = edges
        edge_watershed = edge_chull.copy()

        # thresholding edges for convex hull with threshold to remove as much noise as possible
        edge_chull[edge_chull >= 0.8] = 1
        edge_chull[edge_chull < 0.8] = 0
        # thresholding edges for watershed on edges with low threshold to include as much edges as possible
        edge_watershed[edge_watershed >= 0.2] = 1
        edge_watershed[edge_watershed < 0.2] = 0

        chull = convex_hull(edge_chull)
        watershed_edges_bin = watershed_edges(edge_watershed)
        watershed_cull = chull * watershed_edges_bin[:chull.shape[0], :chull.shape[1]]
        final_image[:watershed_cull.shape[0], :watershed_cull.shape[1]] = watershed_cull
        return final_image
    elif mode == 3:  # never use this one it just never ends
        # thresholding edges for convex hull with threshold to remove as much noise as possible
        edges[edges >= 0.8] = 1
        edges[edges < 0.8] = 0
        chull = concave_hull(edges)
        final_image[:chull.shape[0], :chull.shape[1]] = chull
        return final_image
    else:
        edges[edges != 0] = 1
        edges = dilation(edges)
        # Feel free to play around with the parameters to see how they impact the result
        cv = chan_vese(edges, mu=0.1, lambda1=0.06, lambda2=1, tol=1e-3, max_iter=2000,
                    dt=0.52, init_level_set="checkerboard", extended_output=True)

        image = dilation(cv[0])
        image = dilation(image)
        image = dilation(image)
        image = binary_fill_holes(image)
        final_image[:image.shape[0], :image.shape[1]] = image
        return final_image
# -------------------------------------------------------------------------------------
# concave hull using alpha shape algorithm
# alogrithm was taken from stackoverflow response to question: Calculate bounding polygon of alpha shape from the Delaunay triangulation
# link to answer stackoverflow post -> https://stackoverflow.com/a/50159452/7293149
def alpha_shape(points, alpha, only_outer=True):
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def concave_hull(img):
    image = img.copy()
    points = []
    for j in range(image.shape[1]):
        for i in range(image.shape[0]):
            if image[i, j] == 1:
                points.append((j, i))

    points = np.array(points)
    edges = alpha_shape(points, alpha=1.8, only_outer=True)
    edges = np.array([edge for edge in edges])
    r = edges[:, 1]
    c = edges[:, 0]
    rr, cc = polygon(r, c)
    image[rr, cc] = 1
    return image
# -------------------------------------------------------------------------------------
# using graham scan algorithm to construct convex hull and fill it
def right_turn(p1, p2, p3):
    if (p3[1]-p1[1])*(p2[0]-p1[0]) >= (p2[1]-p1[1])*(p3[0]-p1[0]):
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
    for i in range(len(p)-3, -1, -1):
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
def watershed_edges(img):
    image = img.copy()
    image = img_as_ubyte(image)
    markers = rank.gradient(image, disk(5)) < 20
    markers = ndi.label(markers)[0]
    gradient = rank.gradient(image, disk(2))
    labels = watershed(gradient, markers)
    # show_images([image, labels], ["edges", "edges' labels"])
    labels[labels <= 4] = 1
    labels[labels > 4] = 0
    labels = np.invert(labels)
    labels[labels == -1] = 1
    labels[labels == -2] = 0
    return labels


# def main():
#     IM_SIZE = 400
#     content = io.imread('../images/cow.jpg') / 255.0
#     content = (cv2.resize(content, (IM_SIZE, IM_SIZE))).astype(np.float32)
#     filled_edges = edge_segmentation(content)
#     show_images([content, filled_edges])

# main()
