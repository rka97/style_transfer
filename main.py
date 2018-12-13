import cv2 as cv2
from color_transfer.commonfunctions import *
from color_transfer.color_transfer import *
from domain_transform.domain_transform import *
from fast_nearest_neighbor.fast_nearest_neighbor import *
from sklearn.feature_extraction.image import extract_patches
from sklearn.neighbors import NearestNeighbors
# from sklearn.decomposition import PCA
from skimage.feature import canny
from skimage.segmentation import *
from scipy.ndimage import binary_fill_holes
from timeit import default_timer as timer

LMAX = 3
IM_SIZE = 400
PATCH_SIZES = np.array([33, 21, 13])
SAMPLING_GAPS = np.array([28, 18, 8])
IALG = 10
IRLS_it = 3
IRLS_r = 0.8


def build_gaussian_pyramid(img, L):
    img_arr = []
    img_arr.append(img)  # D_L (img) = img
    for i in range(L - 1):
        img_arr.append(cv2.pyrDown(img_arr[-1].astype(np.float32)).astype(np.float32))
    return img_arr


def segment_edges(img):
    return 4  # 100% edge segmentation


def segment_faces(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_img = (rgb2gray(img) * 255).astype(np.uint8)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    if len(faces) == 0:
        print("Found no faces.")
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        cv2.rectangle(gray_img, (x, y), (x + w, y + h), (255, 0, 0), -1)
    return gray_img / 255.0


def get_segmentation_mask(mode, img=None, c=1.0):
    if mode == 'none' or mode is None or img is None:
        return np.ones((IM_SIZE, IM_SIZE), dtype=np.float32) * c
    elif mode == 'edge':
        edge = (canny(rgb2gray(img), sigma=0.5, low_threshold=0.0, high_threshold=0.3) * 1.0).astype(np.float32)
        return binary_fill_holes(edge)
    elif mode == 'face':
        return segment_faces(img)
    elif mode == 'vese':
        segm = 1 - chan_vese(rgb2gray(img))
        return (segm * 1.0).astype(np.float32)


def solve_irls(X, X_patches_raw, p_index, style_patches, neighbors, projection_matrix):
    p_size = PATCH_SIZES[p_index]
    sampling_gap = SAMPLING_GAPS[p_index]
    current_size = X.shape[0]
    # Extracting Patches
    X_patches = X_patches_raw.reshape(-1, p_size * p_size * 3)
    npatches = X_patches.shape[0]

    # projecting X to same dimention as style patches
    if p_size == 13 or p_size == 21:
        X_patches = project(X_patches, projection_matrix)

    # Computing Nearest Neighbors
    distances, indices = neighbors.kneighbors(X_patches)
    distances += 0.0001
    # Computing Weights
    weights = np.power(distances, IRLS_r - 2)
    # Patch Accumulation
    R = np.zeros((current_size, current_size, 3), dtype=np.float32)
    Rp = extract_patches(R, patch_shape=(p_size, p_size, 3), extraction_step=sampling_gap)
    X[:] = 0
    t = 0
    for t1 in range(X_patches_raw.shape[0]):
        for t2 in range(X_patches_raw.shape[1]):
            nearest_neighbor = style_patches[indices[t, 0]]
            X_patches_raw[t1, t2, 0, :, :, :] += nearest_neighbor * weights[t]
            Rp[t1, t2, 0, :, :, :] += 1 * weights[t]
            t = t + 1
    R += 0.0001  # to avoid dividing by zero.
    X /= R


def style_transfer(content, style, segmentation_mask):
    content_arr = build_gaussian_pyramid(content, LMAX)
    style_arr = build_gaussian_pyramid(style, LMAX)
    segm_arr = build_gaussian_pyramid(segmentation_mask, LMAX)
    X = content_arr[LMAX - 1] + np.random.normal(0, 50, size=content_arr[LMAX - 1].shape) / 255.0
    X = np.clip(X, 0.0, 1.0).astype(np.float32)
    # Set up Content Fusion constants.
    fus_const1 = []
    fus_const2 = []
    for i in range(LMAX):
        sx, sy = segm_arr[i].shape
        curr_segm = segm_arr[i].reshape(sx, sy, 1)
        fus_const1.append(curr_segm * content_arr[i])
        fus_const2.append(1.0 / (curr_segm + 1))
    print('Starting Style Transfer..')
    for L in range(LMAX - 1, -1, -1):  # over scale L
        print('Scale ', L)
        current_size = style_arr[L].shape[0]
        Xbefore = X.copy()
        for n in range(PATCH_SIZES.size):  # over patch size n
            p_size = PATCH_SIZES[n]
            print('Patch Size', p_size)
            style_patches = extract_patches(style_arr[L], patch_shape=(p_size, p_size, 3),
                                            extraction_step=SAMPLING_GAPS[n])
            npatchx, npatchy, _, _, _, _ = style_patches.shape
            npatches = npatchx * npatchy
            # Preparing for NN
            style_patches = style_patches.reshape(-1, p_size * p_size * 3)
            njobs = 1
            if (L == 0) or (L == 1 and p_size <= 13):
                njobs = -1
            # neighbors = NearestNeighbors(n_neighbors=1, p=2, algorithm='brute', n_jobs=njobs).fit(style_patches)
            # style_patches = style_patches.reshape((-1, p_size, p_size, 3))

            projection_matrix = 0
            # for small patches perform PCA
            if p_size == 13 or p_size == 21:
                new_stlye_patches, projection_matrix = pca(style_patches)
                neighbors = NearestNeighbors(n_neighbors=1, p=2, algorithm='kd_tree', n_jobs=njobs).fit(new_stlye_patches)
            else:
                neighbors = NearestNeighbors(n_neighbors=1, p=2, algorithm='kd_tree', n_jobs=njobs).fit(style_patches)
            style_patches = style_patches.reshape((-1, p_size, p_size, 3))

            for k in range(IALG):  # over # of algorithm iterations IALG
                # Steps 1 & 2: Patch-Extraction and and Robust Patch Aggregation
                X_patches_raw = extract_patches(X, patch_shape=(p_size, p_size, 3), extraction_step=SAMPLING_GAPS[n])
                for i in range(IRLS_it):
                    solve_irls(X, X_patches_raw, n, style_patches, neighbors, projection_matrix)
                # Step 3: Content Fusion
                X = fus_const2[L] * (X + fus_const1[L])
                # Step 4: Color Transfer
                X = imhistmatch2(X, style)
                # Step 5: Denoising
                X = denoise(X, sigma_r=0.17, sigma_s=20)
        # show_images([Xbefore, X])
        # Upscale X
        if (L > 0):
            sizex, sizey, _ = content_arr[L - 1].shape
            X = cv2.resize(X, (sizex, sizey))
    return X


def main():
    content = io.imread('images/emilia2.jpg') / 255.0
    style = io.imread('images/van_gogh.jpg') / 255.0
    segm_mask = get_segmentation_mask('face', content, 0.0)
    content = (cv2.resize(content, (IM_SIZE, IM_SIZE))).astype(np.float32)
    style = (cv2.resize(style, (IM_SIZE, IM_SIZE))).astype(np.float32)
    segm_mask = (cv2.resize(segm_mask, (IM_SIZE, IM_SIZE))).astype(np.float32)
    show_images([content, segm_mask, style])
    content = imhistmatch(content, style)
    start = timer()
    X = style_transfer(content, style, segm_mask)
    end = timer()
    print("Style Transfer took ", end - start, " seconds!")
    # Finished. Just show the images
    show_images([content, segm_mask, style, X])

def mainGui(content_image, stlye_image):
    content = io.imread(content_image) / 255.0
    style = io.imread(stlye_image) / 255.0
    segm_mask = get_segmentation_mask('face', content, 0.0)
    content = (cv2.resize(content, (IM_SIZE, IM_SIZE))).astype(np.float32)
    style = (cv2.resize(style, (IM_SIZE, IM_SIZE))).astype(np.float32)
    segm_mask = (cv2.resize(segm_mask, (IM_SIZE, IM_SIZE))).astype(np.float32)
    # show_images([content, segm_mask, style])
    content = imhistmatch(content, style)
    start = timer()
    X = style_transfer(content, style, segm_mask)
    end = timer()
    print("Style Transfer took ", end - start, " seconds!")
    X_fixed = X * 255.0
    X_fixed = X_fixed.astype(np.uint8)
    # io.imsave("x.png", X_fixed)
    return X_fixed

# main()
