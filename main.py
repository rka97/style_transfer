import cv2 as cv2
from color_transfer.commonfunctions import *
from color_transfer.color_transfer import *
from domain_transform.domain_transform import *
from pca.pca import *
from sklearn.feature_extraction.image import extract_patches
from sklearn.neighbors import NearestNeighbors
# from sklearn.decomposition import PCA
from skimage.filters import gaussian
from skimage.feature import canny
from skimage.util import view_as_windows, pad, random_noise
from skimage.segmentation import *
from scipy.ndimage import binary_fill_holes
from timeit import default_timer as timer
from edge_segmentation.edge_detection import *
from skimage.segmentation import active_contour

LMAX = 3
IM_SIZE = 400
PATCH_SIZES = np.array([33, 21, 13, 9, 5])
SAMPLING_GAPS = np.array([28, 18, 8, 5, 3])
IALG = 10
IRLS_it = 3
IRLS_r = 0.8
PADDING_MODE = 'constant'


def build_gaussian_pyramid(img, L):
    img_arr = []
    img_arr.append(img)  # D_L (img) = img
    for i in range(L - 1):
        img_arr.append(cv2.pyrDown(img_arr[-1].astype(np.float32)).astype(np.float32))
    return img_arr


def segment_edges(img, sigma_c=2, low_threshold=0.1, high_threshold=0.2, sigma_post=2, num_dilation=1):
    gray_img = rgb2gray(img)
    gray_img = (gray_img - np.min(gray_img)) / (np.max(gray_img) - np.min(gray_img))
    E = canny(rgb2gray(img), sigma=sigma_c, low_threshold=low_threshold, high_threshold=high_threshold)
    for i in range(num_dilation):
        E = dilation(E)
    show_images([E])
    return 1.0 * (gaussian(1.0 * E + morphological_chan_vese(rgb2gray(img), iterations=35, init_level_set=E, smoothing=1), sigma=sigma_post) > 0)


def segment_faces(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_img = (rgb2gray(img) * 255).astype(np.uint8)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    temp_img = (img.copy() * 255).astype(np.uint8)
    mask = segment_edges(img)
    if len(faces) == 0:
        print("Found no faces. Will Hallucinate.")
        return np.zeros_like(gray_img)
    for (x, y, w, h) in faces:
        xs = x  # max([x - int(w / 2), 0])
        ys = y  # max([y - int(h / 2), 0])
        xe = x + w  # int(3 * w / 2)
        ye = y + h  # int(3 * h / 2)
        mask[xs:xe, ys:ye] += dilation(canny(gray_img[xs:xe, ys:ye], sigma=2))
        # show_images([mask])
        # mask = 1.0 * (gaussian(1.0 * mask + morphological_chan_vese(rgb2gray(img), iterations=35, init_level_set=mask, smoothing=1), sigma=0.5) > 0)
        # # mask[x:x + w, y:y + h] = segment_edges(gray_img[x:x + w, y:y + h], sigma_c=1, low_threshold=0.1, high_threshold=0.2, num_dilation=0)
        # show_images([mask])
        return mask
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(temp_img, mask=mask, rect=(x, y, w, h), bgdModel=bgdModel, fgdModel=fgdModel, iterCount=10, mode=cv2.GC_INIT_WITH_MASK)
        mask_min = np.min(mask)
        mask_max = np.max(mask)
        mask = (mask - mask_min) / (mask_max - mask_min)
        return gaussian(mask, 5)


def get_segmentation_mask(mode, img=None, c=1.0):
    if mode == 'none' or mode is None or img is None:
        return np.ones((IM_SIZE, IM_SIZE), dtype=np.float32) * c
    elif mode == 'edge':
        return edge_segmentation(img) * c
        # return segment_edges(img) * c
    elif mode == 'face':
        return segment_faces(img) * c


def solve_irls(X, X_patches_raw, p_index, style_patches, neighbors, projection_matrix):
    p_size = PATCH_SIZES[p_index]
    sampling_gap = SAMPLING_GAPS[p_index]
    current_size = X.shape[0]
    # Extracting Patches
    X_patches_raw_cpy = X_patches_raw.copy()
    X_patches = X_patches_raw.reshape(-1, p_size * p_size * 3)
    npatches = X_patches.shape[0]
    # Projecting X to same dimention as style patches
    if p_size <= 21:
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
    # X = content_arr[LMAX - 1] + np.random.normal(0, 50, size=content_arr[LMAX - 1].shape) / 255.0
    X = random_noise(content_arr[LMAX - 1], mode='gaussian', var=20)
    # X = np.clip(X, 0.0, 1.0).astype(np.float32)
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
        style_L_sx, style_L_sy, _ = style_arr[L].shape
        X = random_noise(X, mode='gaussian', var=20 / 250.0)
        # X = X + np.random.normal(0, np.max(X), size=X.shape)
        Xbefore = X.copy()
        for n in range(PATCH_SIZES.size):  # over patch size n
            p_size = PATCH_SIZES[n]
            print('Patch Size', p_size)
            # We should pad the image here to ensure dimension
            # SAMPLING_GAPS[n] * (npatchx - 1) <= style_L_sx - p_size
            npatchx = int((style_L_sx - p_size) / SAMPLING_GAPS[n] + 1)
            padding = p_size - (style_L_sx - npatchx * SAMPLING_GAPS[n])
            # new_size = style_L_sx + padding
            # padding = 0
            padding_arr = ((0, padding), (0, padding), (0, 0))
            current_style = pad(style_arr[L], padding_arr, mode=PADDING_MODE)
            X = pad(X, padding_arr, mode=PADDING_MODE)
            const1 = pad(fus_const1[L], padding_arr, mode=PADDING_MODE)
            const2 = pad(fus_const2[L], padding_arr, mode=PADDING_MODE)
            style_patches = extract_patches(current_style, patch_shape=(p_size, p_size, 3),
                                            extraction_step=SAMPLING_GAPS[n])
            npatchx, npatchy, _, _, _, _ = style_patches.shape
            npatches = npatchx * npatchy
            # Preparing for NN
            style_patches = style_patches.reshape(-1, p_size * p_size * 3)
            njobs = 1
            if (L == 0) or (L == 1 and p_size <= 13):
                njobs = -1
            projection_matrix = 0
            # for small patches perform PCA
            if p_size <= 21:
                new_style_patches, projection_matrix = pca(style_patches)
                neighbors = NearestNeighbors(n_neighbors=1, p=2, n_jobs=njobs).fit(new_style_patches)
            else:
                neighbors = NearestNeighbors(n_neighbors=1, p=2, n_jobs=njobs).fit(style_patches)
            style_patches = style_patches.reshape((-1, p_size, p_size, 3))
            for k in range(IALG):  # over # of algorithm iterations IALG
                # Steps 1 & 2: Patch-Extraction and and Robust Patch Aggregation
                X_patches_raw = extract_patches(X, patch_shape=(p_size, p_size, 3), extraction_step=SAMPLING_GAPS[n])
                for i in range(IRLS_it):
                    solve_irls(X, X_patches_raw, n, style_patches, neighbors, projection_matrix)
                # Step 3: Content Fusion
                X = const2 * (X + const1)
                # Step 4: Color Transfer
                X = color_transfer(X, style)
                # Step 5: Denoising
                # Xpden = X.copy()
                X[:style_L_sx, :style_L_sx, :] = denoise(X[:style_L_sx, :style_L_sx, :], sigma_r=0.17, sigma_s=15)
                # show_images([Xpden, X])
            X = X[:style_L_sx, :style_L_sx, :]
        show_images([Xbefore, X])
        # Upscale X
        if (L > 0):
            sizex, sizey, _ = content_arr[L - 1].shape
            X = cv2.resize(X, (sizex, sizey))
    return X


def main():
    content = io.imread('images/emilia2.jpg') / 255.0
    style = io.imread('images/paper_images/van_gogh.jpg') / 255.0
    segm_mask = edge_segmentation(content, 5, 0.6)
    content = (cv2.resize(content, (IM_SIZE, IM_SIZE))).astype(np.float32)
    style = (cv2.resize(style, (IM_SIZE, IM_SIZE))).astype(np.float32)
    segm_mask = (cv2.resize(segm_mask, (IM_SIZE, IM_SIZE))).astype(np.float32)
    show_images([content, segm_mask, style])
    original_content = content.copy()
    content = color_transfer(content, style)
    start = timer()
    X = style_transfer(content, style, segm_mask)
    end = timer()
    print("Style Transfer took ", end - start, " seconds!")
    # Finished. Just show the images
    show_images([original_content, segm_mask, style])
    show_images([X])


def main_gui(content_image, style_image):
    content = io.imread(content_image) / 255.0
    style = io.imread(style_image) / 255.0
    segm_mask = get_segmentation_mask('edge', content, 0.2)
    content = (cv2.resize(content, (IM_SIZE, IM_SIZE))).astype(np.float32)
    style = (cv2.resize(style, (IM_SIZE, IM_SIZE))).astype(np.float32)
    segm_mask = (cv2.resize(segm_mask, (IM_SIZE, IM_SIZE))).astype(np.float32)
    # show_images([content, segm_mask, style])
    content = color_transfer(content, style)
    start = timer()
    X = style_transfer(content, style, segm_mask)
    end = timer()
    print("Style Transfer took ", end - start, " seconds!")
    X_fixed = X * 255.0
    X_fixed = X_fixed.astype(np.uint8)
    # io.imsave("x.png", X_fixed)
    return X_fixed

main()
