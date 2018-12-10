import cv2
from color_transfer.commonfunctions import *
from color_transfer.color_transfer import *
from domain_transform.domain_transform import *
from sklearn.feature_extraction.image import extract_patches
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
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


# Computes np.sum(A * B, axis=0)
def SumOfProduct(A, B, num_times=1):
    sx, sy = A.shape
    C = np.zeros(sy, dtype=np.float32)
    # do it over 4 times
    step = int(sx / num_times)
    for i in range(0, sx, step):
        C += np.sum(A[i:i + step, :] * B[i:i + step, :], axis=0)
    return C


def build_gaussian_pyramid(img, L):
    img_arr = []
    img_arr.append(img)  # D_L (img) = img
    for i in range(L - 1):
        img_arr.append(cv2.pyrDown(img_arr[-1].astype(np.float32)).astype(np.float32))
    return img_arr


def get_segmentation_mask(mode, img=None, c=1.0):
    if mode == 'none' or mode is None or img is None:
        return np.ones((IM_SIZE, IM_SIZE), dtype=np.float32) * c
    else:
        if mode == 'edge':
            edge = (canny(rgb2gray(img), sigma=0.5, low_threshold=0.0, high_threshold=0.3) * 1.0).astype(np.float32)
            return binary_fill_holes(edge)
        elif mode == 'vese':
            segm = 1 - chan_vese(rgb2gray(img))
            return (segm * 1.0).astype(np.float32)



def solve_irls(X, X_patches_raw, p_index, style_patches, neighbors):
    p_size = PATCH_SIZES[p_index]
    sampling_gap = SAMPLING_GAPS[p_index]
    current_size = X.shape[0]
    # Extracting Patches
    X_patches = X_patches_raw.reshape(-1, p_size * p_size * 3)
    npatches = X_patches.shape[0]
    # Computing Nearest Neighbors
    distances, indices = neighbors.kneighbors(X_patches)
    distances += 0.0001
    # Computing Weights
    weights = np.power(distances, IRLS_r - 2)
    # Patch Accumulation
    R = np.zeros((current_size, current_size, 3), dtype=np.float32)
    Rp = extract_patches(R, patch_shape=(p_size, p_size, 3), extraction_step=sampling_gap)
    Z = np.zeros((current_size, current_size, 3), dtype=np.float32)
    Xp = X_patches_raw.copy()
    X[:] = 0
    t = 0
    for t1 in range(X_patches_raw.shape[0]):
        for t2 in range(X_patches_raw.shape[1]):
            nearest_neighbor = np.reshape(style_patches[indices[t, 0]], (p_size, p_size, 3))
            # show_images([Xp[t1, t2, 0, :, :, :], nearest_neighbor])
            X_patches_raw[t1, t2, 0, :, :, :] += nearest_neighbor * weights[t]
            Rp[t1, t2, 0, :, :, :] += 1 * weights[t]
            t = t + 1
    R += 0.0001
    #R[np.abs(R) < 0.001] = 0.001
    X /= R


def style_transfer(content, style, segmentation_mask):
    content_arr = build_gaussian_pyramid(content, LMAX)
    style_arr = build_gaussian_pyramid(style, LMAX)
    segm_arr = build_gaussian_pyramid(segmentation_mask, LMAX)
    X = content_arr[LMAX - 1] + np.random.normal(0, 50, size=content_arr[LMAX - 1].shape) / 255.0
    # X = np.abs(X).astype(np.float32)
    X = np.clip(X, 0.0, 1.0).astype(np.float32)
    # Set up IRLS constants.
    irls_const1 = []
    irls_const2 = []
    for i in range(LMAX):
        sx, sy = segm_arr[i].shape
        curr_segm = segm_arr[i].reshape(sx, sy, 1)
        irls_const1.append(curr_segm * content_arr[i])
        irls_const2.append(1.0 / (curr_segm + 1))
    print('Starting Style Transfer..')
    for L in range(LMAX - 1, -1, -1):  # over scale L
        print('Scale ', L)
        current_style = style_arr[L]
        current_size = style_arr[L].shape[0]
        Xbefore = X.copy()
        for n in range(PATCH_SIZES.size):  # over patch size n
            p_size = PATCH_SIZES[n]
            print('Patch Size', p_size)
            style_patches = extract_patches(current_style, patch_shape=(p_size, p_size, 3), extraction_step=SAMPLING_GAPS[n])
            npatchx, npatchy, _, _, _, _ = style_patches.shape
            npatches = npatchx * npatchy
            # print("Preparing for NN")
            style_patches = style_patches.reshape(-1, p_size * p_size * 3)
            njobs = 1
            if (L == 0) or (L == 1 and p_size <= 13):
                njobs = -1
            neighbors = NearestNeighbors(n_neighbors=1, p=2, algorithm='brute', n_jobs=njobs).fit(style_patches)
            for k in range(IALG):  # over # of algorithm iterations IALG
                # Steps 1 & 2: Patch-Matching and and Robust Patch Aggregation
                X_patches_raw = extract_patches(X, patch_shape=(p_size, p_size, 3), extraction_step=SAMPLING_GAPS[n])
                for i in range(IRLS_it):
                    solve_irls(X, X_patches_raw, n, style_patches, neighbors)
                # Step 3: Content Fusion
                current_segm = segm_arr[L].reshape((current_size, current_size, 1))
                X = irls_const2[L] * (X + irls_const1[L])
                # Step 4: Color Transfer
                X = imhistmatch2(X, style)
                # Step 5: Denoising
                # Xpn = X.copy()
                X = denoise(X, sigma_r=0.17, sigma_s=20)
                # show_images([Xpn, X])
        show_images([Xbefore, X])
        # Upscale X
        if (L > 0):
            sizex, sizey, _ = content_arr[L - 1].shape
            X = cv2.resize(X, (sizex, sizey))
    return X


def main():
    content = cv2.resize(io.imread('images/houses.jpg'), (IM_SIZE, IM_SIZE)) / 255.0
    content = content.astype(np.float32)[:, :, 0:3]
    # content[:] = 0.0
    # print(content.shape)
    style = cv2.resize(io.imread('images/van_gogh.jpg'), (IM_SIZE, IM_SIZE)) / 255.0
    style = style.astype(np.float32)
    content = imhistmatch(content, style)
    segmentation_mask = get_segmentation_mask('vese', content, 0.1)
    show_images([content, segmentation_mask, style])
    start = timer()
    X = style_transfer(content, style, segmentation_mask)
    end = timer()
    print("Style Transfer took ", end - start, " seconds!")
    # Finished. Just show the images
    show_images([content, segmentation_mask, style, X])

main()
