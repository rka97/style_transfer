import cv2
from color_transfer.commonfunctions import *
from color_transfer.color_transfer import *
from domain_transform.domain_transform import *
from sklearn.feature_extraction.image import extract_patches
from sklearn.neighbors import NearestNeighbors
LMAX = 3
IM_SIZE = 400
PATCH_SIZES = np.array([33, 21, 13, 9])
SAMPLING_GAPS = np.array([28, 18, 8, 5])
IALG = 1
IRLS_it = 1
IRLS_r = 0.8


# Computes np.sum(A * B, axis=0)
def SumOfProduct(A, B, num_times=1):
    sx, sy = A.shape
    C = np.zeros(sy, dtype=np.float16)
    # do it over 4 times
    step = int(sx / num_times)
    for i in range(0, sx, step):
        C += np.sum(A[i:i + step, :] * B[i:i + step, :], axis=0)
    return C


def build_gaussian_pyramid(img, L):
    img_arr = []
    img_arr.append(img)  # D_L (img) = img
    for i in range(L - 1):
        img_arr.append(cv2.pyrDown(img_arr[-1].astype(np.float32)).astype(np.float16))
    return img_arr


def get_segmentation_mask(mode):
    return np.zeros((IM_SIZE, IM_SIZE), dtype=np.float16)


def main():
    content = cv2.resize(io.imread('images/ocean_day.jpg'), (IM_SIZE, IM_SIZE)) / 255.0
    content = content.astype(np.float16)
    style = cv2.resize(io.imread('images/autumn.jpg'), (IM_SIZE, IM_SIZE)) / 255.0
    style = style.astype(np.float16)
    # show_images([content, style, color_transfer_lab(content, style)])
    content = color_transfer_lab(content, style)
    segmentation_mask = get_segmentation_mask(None)
    content_arr = build_gaussian_pyramid(content, LMAX)
    style_arr = build_gaussian_pyramid(style, LMAX)
    segm_arr = build_gaussian_pyramid(segmentation_mask, LMAX)
    X = content_arr[LMAX - 1] + np.random.normal(0, 50, size=content_arr[LMAX - 1].shape) / 255.0

    irls_const1 = []
    irls_const2 = []
    for i in range(LMAX):
        sx, sy = segm_arr[i].shape
        curr_segm = segm_arr[i].reshape(sx, sy, 1)
        irls_const1.append(curr_segm * content_arr[i])
        irls_const2.append(curr_segm + 1)
    print('Starting..')
    for L in range(LMAX - 1, -1, -1):  # over scale L
        print('Scale ', L)
        current_style = style_arr[L]
        current_size = style_arr[L].shape[0]
        for n in range(PATCH_SIZES.size):  # over patch size n
            p_size = PATCH_SIZES[n]
            print('Patch Size', p_size)
            style_patches = extract_patches(current_style, patch_shape=(p_size, p_size, 3), extraction_step=SAMPLING_GAPS[n])
            npatchx, npatchy, _, _, _, _ = style_patches.shape
            npatches = npatchx * npatchy
            R = np.zeros((npatches + 1, current_size, current_size, 3), dtype=np.float16)
            Z = np.zeros((npatches + 1, current_size, current_size, 3), dtype=np.float16)
            for t in range(npatches):
                t_i = int(t / npatches) * SAMPLING_GAPS[n]
                t_j = int(t % npatches) * SAMPLING_GAPS[n]
                R[t, t_i:t_i + p_size, t_j:t_j + p_size, :] = 1
                Z[t, t_i:t_i + p_size, t_j:t_j + p_size, :] = current_style[t_i:t_i + p_size, t_j:t_j + p_size, :]
            R = R.reshape(-1, current_size * current_size * 3)
            Z = Z.reshape(-1, current_size * current_size * 3)
            print("Preparing for NN")
            style_patches = style_patches.reshape(-1, p_size * p_size * 3)
            neighbors = NearestNeighbors(n_neighbors=1, p=2, algorithm='ball_tree').fit(style_patches)
            for k in range(IALG):  # over # of algorithm iterations IALG
                # Steps 1 & 2: Patch-Matching and and Robust Patch Aggregation
                print("k = ", k)
                X_patches_raw = extract_patches(X, patch_shape=(p_size, p_size, 3), extraction_step=SAMPLING_GAPS[n])
                for i in range(IRLS_it):
                    print("i = ", i)
                    print("Extracing Patches.")
                    X_patches = X_patches_raw.reshape(-1, p_size * p_size * 3)
                    print("Computing nearest neighbors..")
                    distances, indices = neighbors.kneighbors(X_patches)
                    print("Computing weights..")
                    weights = np.power(distances, IRLS_r - 2)
                    print("Computing Z_p")
                    print(indices.shape)
                    print(Z.shape)
                    Z_p = np.zeros((Z.shape[0] - 1, Z.shape[1]))
                    print(Z_p.shape)
                    for t in range(npatches):
                        Z[npatches] += Z[indices[t][0]]
                    print("Computing the overlapMatrix")
                    R_p = np.zeros((R.shape[0] - 1, R.shape[1]))
                    for t in range(npatches):
                        R_p += (R[t] * weights[t])
                    RxWeights = np.zeros((R.shape[1]), dtype=np.float16)
                    Xhat = np.zeros((R.shape[1]), dtype=np.float16)
                    """
                    num_times = LMAX - L
                    if p_size <= 10 and L <= 0:
                        num_times = R.shape[0]
                    RxWeights = SumOfProduct(R, weights, num_times)
                    Xhat = SumOfProduct(Z_p, weights)
                    """
                    print("Computing overlapMatrix Part 2")
                    overlapMatrix = RxWeights
                    overlapMatrix += (np.abs(overlapMatrix) < 0.01) * 1
                    print("Computing Xhat")
                    Xhat = Xhat / overlapMatrix
                    print("Altering X")
                    X[:] = (Xhat.reshape(current_size, current_size, 3))[:]
                print("Content Fusion..")
                # Step 3: Content Fusion
                current_segm = segm_arr[L].reshape((current_size, current_size, 1))
                X = (X + irls_const1[L]) / irls_const2[L]
                # X = (np.abs(X) <= 1) * X
                print("Color Transfer..")
                # Step 4: Color Transfer
                X = color_transfer_lab(X, style)
                print("Denoising.. ")
                # Step 5: Denoising
                X = denoise(X)
        # Upscale X
        if (L > 0):
            sizex, sizey, _ = content_arr[L - 1].shape
            X = cv2.resize(X, (sizex, sizey))
    # Finished. Just show the images
    show_images([content, style, X])



main()
