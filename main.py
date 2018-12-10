import cv2
from color_transfer.commonfunctions import *
from color_transfer.color_transfer import *
from domain_transform.domain_transform import *
from sklearn.feature_extraction.image import extract_patches
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from timeit import default_timer as timer
LMAX = 3
IM_SIZE = 400
PATCH_SIZES = np.array([33, 21, 13, 9])
SAMPLING_GAPS = np.array([28, 18, 8, 5])
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


def get_segmentation_mask(mode):
    return np.ones((IM_SIZE, IM_SIZE), dtype=np.float32) * 0


def solve_irls(X, X_patches_raw, p_index, style_patches, neighbors):
    p_size = PATCH_SIZES[p_index]
    sampling_gap = SAMPLING_GAPS[p_index]
    current_size = X.shape[0]
    # Extracting Patches
    X_patches = X_patches_raw.reshape(-1, p_size * p_size * 3)
    npatches = X_patches.shape[0]
    # Computing Nearest Neighbors
    distances, indices = neighbors.kneighbors(X_patches)
    # Computing Weights
    weights = np.power(distances, IRLS_r - 2)
    # Patch Accumulation
    R = np.zeros((current_size, current_size, 3), dtype=np.float32)
    Z = np.zeros((current_size, current_size, 3), dtype=np.float32)
    X[:] = 0
    t_i = -1 * sampling_gap
    t_j = 0
    for t in range(npatches):
        t_i += sampling_gap
        if t_i + p_size > current_size:
            t_i = 0
            t_j += sampling_gap
            if t_j + p_size > current_size:
                raise ValueError("We really should never be here.\n")
        nearest_neighbor = np.reshape(style_patches[indices[t, 0]], (p_size, p_size, 3))
        X[t_i:t_i + p_size, t_j:t_j + p_size, :] += nearest_neighbor * weights[t]
        R[t_i:t_i + p_size, t_j:t_j + p_size, :] += 1 * weights[t]
    R[np.abs(R) < 0.001] = 0.001
    X /= R


def style_transfer(content, style, segmentation_mask):
    content_arr = build_gaussian_pyramid(content, LMAX)
    style_arr = build_gaussian_pyramid(style, LMAX)
    segm_arr = build_gaussian_pyramid(segmentation_mask, LMAX)
    X = content_arr[LMAX - 1] + np.random.normal(0, 50, size=content_arr[LMAX - 1].shape) / 255.0
    # Set up IRLS constants.
    irls_const1 = []
    irls_const2 = []
    for i in range(LMAX):
        sx, sy = segm_arr[i].shape
        curr_segm = segm_arr[i].reshape(sx, sy, 1)
        irls_const1.append(curr_segm * content_arr[i])
        irls_const2.append(curr_segm + 1)
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
            neighbors = NearestNeighbors(n_neighbors=1, p=2, algorithm='ball_tree', n_jobs=njobs).fit(style_patches)
            for k in range(IALG):  # over # of algorithm iterations IALG
                # Steps 1 & 2: Patch-Matching and and Robust Patch Aggregation
                X_patches_raw = extract_patches(X, patch_shape=(p_size, p_size, 3), extraction_step=SAMPLING_GAPS[n])
                for i in range(IRLS_it):
                    solve_irls(X, X_patches_raw, n, style_patches, neighbors)
                # Step 3: Content Fusion
                current_segm = segm_arr[L].reshape((current_size, current_size, 1))
                X[:] = (X[:] + irls_const1[L]) / irls_const2[L]
                # Step 4: Color Transfer
                X = color_transfer_lab(X, style)
                # Step 5: Denoising
                # X = denoise(X)
        # show_images([Xbefore, X])
        # Upscale X
        if (L > 0):
            sizex, sizey, _ = content_arr[L - 1].shape
            X = cv2.resize(X, (sizex, sizey))
    return X


def main():
    content = cv2.resize(io.imread('images/ocean_day.jpg'), (IM_SIZE, IM_SIZE)) / 255.0
    content = content.astype(np.float32)
    style = cv2.resize(io.imread('images/ocean_sunset.jpg'), (IM_SIZE, IM_SIZE)) / 255.0
    style = style.astype(np.float32)
    content = color_transfer_lab(content, style)
    show_images([content, style])
    segmentation_mask = get_segmentation_mask(None)
    start = timer()
    X = style_transfer(content, style, segmentation_mask)
    end = timer()
    print("Style Transfer took ", end - start, " seconds!")
    # Finished. Just show the images
    show_images([content, style, X])

main()
