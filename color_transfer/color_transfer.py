from skimage import color
from scipy.linalg import fractional_matrix_power as fractional_power
from numpy.linalg import matrix_power
import cv2
from .commonfunctions import *
from skimage.exposure import cumulative_distribution


#color transfer with REAL histogram matching using interpolation
def color_transfer(content, style):
    transfered = np.copy(content)
    #for each channel of the content, match the cum_histogram with the style's one
    for i in range (0, content.shape[2]):    
        content_channel = content[:,:,i].flatten()
        style_channel = style[:,:,i].flatten()
         #calculate histogram for both content and style
        content_values, content_indices, content_counts = np.unique(content_channel, return_inverse=True, return_counts=True)
        style_values, style_counts = np.unique(style_channel, return_counts=True)
        #calculate cummulative histogram
        content_cumhist = np.cumsum(content_counts)
        style_cumhist = np.cumsum(style_counts)
        #normalize it
        content_cumhist = content_cumhist / max(content_cumhist) 
        style_cumhist = style_cumhist / max(style_cumhist)
        #match using interpolation
        matched = np.interp(content_cumhist, style_cumhist, style_values)
        transfered[:,:,i] = matched[content_indices].reshape(content[:,:,i].shape)
    return transfered

# does color transfer through converting an image to the LAB color space, changing
# the mean and variance there, then converting the image back into RGB
def color_transfer_lab(content, style):
    # convert images to LAB space
    style_lab = cv2.cvtColor((style * 255).astype("uint8"), cv2.COLOR_RGB2LAB).astype("float32")  # color.rgb2lab(style)
    content_lab = cv2.cvtColor((content * 255).astype("uint8"), cv2.COLOR_RGB2LAB).astype("float32")  # color.rgb2lab(content)
    # calculate mean
    content_mu = np.mean(content_lab, axis=tuple(range(2)))
    style_mu = np.mean(style_lab, axis=tuple(range(2)))
    # calculate standard deviation
    content_std = np.std(content_lab, axis=tuple(range(2)))
    style_std = np.std(style_lab, axis=tuple(range(2)))
    # transfer
    content_lab = (content_lab - content_mu) * (content_std / style_std) + style_mu
    content_lab = np.clip(content_lab, 0, 255)
    # convert back to RGB)
    content_rgb = cv2.cvtColor(content_lab.astype("uint8"), cv2.COLOR_LAB2RGB)  # color.lab2rgb(content_lab)
    content_rgb = np.clip(content_rgb, 0, 255) / 255.0
    return content_rgb.astype("float32")


# doe color transfer through histogram matching
def color_transfer_hm(content, style):
    original = content.copy()
    # calculate the mean
    content_mu = np.mean(content, axis=(0, 1))
    style_mu = np.mean(content, axis=(0, 1))
    # print(style.reshape(-1, 3).shape)
    # calculate the covariance
    style_cov = np.cov(style.reshape(-1, 3).T)
    content_cov = np.cov(content.reshape(-1, 3).T)
    # get A, b
    A = fractional_power(style_cov, 0.5).dot(fractional_power(content_cov, -0.5))
    b = (style_mu.T - A.dot(content_mu.T)).T
    x, y, z = np.shape(content)
    content = np.dot(A, original.reshape(-1, z).T)
    content = (content.T + b).reshape(x, y, z)
    content = np.clip(content, 0.0, 1.0)
    return content

# Taken from here
# https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
def hist_match(template, source):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def imhistmatch(content, style):
    img = np.zeros_like(content)
    for i in range(content.shape[2]):
        img[:, :, i] = hist_match(style[:, :, i], content[:, :, i])
    return img


def cdf(im):
    '''
    computes the CDF of an image im as 2D numpy ndarray
    '''
    c, b = cumulative_distribution(im) 
    # pad the beginning and ending pixels and their CDF values
    c = np.insert(c, 0, [0]*b[0])
    c = np.append(c, [1]*(255-b[-1]))
    return c

def hist_matching(style, im):
    '''
    c: CDF of input image computed with the function cdf()
    c_t: CDF of template image computed with the function cdf()
    im: input image as 2D numpy ndarray
    returns the modified pixel values
    ''' 
    c = cdf(im)
    c_t = cdf(style)
    pixels = np.arange(256)
    # find closest pixel-matches corresponding to the CDF of the input image, given the value of the CDF H of   
    # the template image at the corresponding pixels, s.t. c_t = H(pixels) <=> pixels = H-1(c_t)
    new_pixels = np.interp(c, c_t, pixels) 
    im = (np.reshape(new_pixels[im.ravel()], im.shape)).astype(np.uint8)
    return im


def imhistmatch2(content, style):
    img = (255 * content.copy()).astype(np.uint8)
    sty = (255 * style.copy()).astype(np.uint8)
    for i in range(content.shape[2]):
        img[:, :, i] = hist_matching(sty[:, :, i], img[:, :, i])
    return img / 255.0