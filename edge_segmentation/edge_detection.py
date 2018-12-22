import cv2
from numpy.linalg import eig
from .commonfunctions import *
from sklearn.feature_extraction.image import extract_patches
from numpy.linalg import multi_dot
import cv2
from numpy import pi, exp, sqrt
from scipy import ndimage as ndi
from skimage.draw import polygon
from skimage.filters import rank
from numpy.linalg import multi_dot
from skimage.util import img_as_ubyte
from skimage.morphology import watershed, disk
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


def main():
    IM_SIZE = 400
    content = io.imread('../images/emilia2.jpg') / 255.0
    content = (cv2.resize(content, (IM_SIZE, IM_SIZE))).astype(np.float32)
    root_n = 5
    edge_chull = edge_detection(content, root_n)  # root_n should be odd number
    edge_watershed = edge_chull.copy()

    # thresholding edges for watershed on edges with low threshold to include as much edges as possible
    edge_watershed[edge_watershed >= 0.2] = 1
    edge_watershed[edge_watershed < 0.2] = 0
    # thresholding edges for convex hull with threshold to remove as much noise as possible
    edge_chull[edge_chull >= 0.8] = 1
    edge_chull[edge_chull < 0.8] = 0

    show_images([content, edge_chull, edge_watershed], ["content", "edges convex hull", "edges watershed"])

    # running different filling algorithms
    chull = convex_hull(edge_chull)
    watershed_content_bin = watershed_content(content)
    watershed_edges_bin = watershed_edges(edge_watershed)

    show_images(
        [content, chull, watershed_content_bin, watershed_edges_bin],
        ["content", "convex hull filling", "watershed on content binaraized", "watershed on edges binaraized"]
    )


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


def watershed_content(img):
    image = img.copy()
    image = rgb2gray(image)
    image = img_as_ubyte(image)
    markers = rank.gradient(image, disk(5)) < 20
    markers = ndi.label(markers)[0]
    gradient = rank.gradient(image, disk(2))
    labels = watershed(gradient, markers)
    show_images([image, labels], ["content", "content's labels"])
    img_hist = histogram(labels, nbins=256)
    for i in range(img_hist[0].shape[0]):
        if img_hist[0][i] < int(round(0.00625 * labels.shape[0] * labels.shape[1])):
            labels[labels == img_hist[1][i]] = 0

    labels[labels <= 7] = 1
    labels[labels > 7] = 0
    h, w = labels.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(labels, mask, (0, 0), 255)
    return np.invert(labels)


def watershed_edges(img):
    image = img.copy()
    image = img_as_ubyte(image)
    markers = rank.gradient(image, disk(5)) < 20
    markers = ndi.label(markers)[0]
    gradient = rank.gradient(image, disk(2))
    labels = watershed(gradient, markers)
    show_images([image, labels], ["edges", "edges' labels"])
    labels[labels <= 7] = 1
    labels[labels > 7] = 0
    return np.invert(labels)

# main()
