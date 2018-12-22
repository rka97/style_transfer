import cv2
import numpy as np
from skimage.exposure import cumulative_distribution


# color transfer with histogram matching using interpolation
def color_transfer(content, style):
    transfered = np.copy(content)
    # for each channel of the content, match the cum_histogram with the style's one
    for i in range(0, content.shape[2]):
        content_channel = content[:, :, i].flatten()
        style_channel = style[:, :, i].flatten()
        # calculate histogram for both content and style
        content_values, content_indices, content_counts = np.unique(content_channel, return_inverse=True, return_counts=True)
        style_values, style_counts = np.unique(style_channel, return_counts=True)
        # calculate cummulative histogram
        content_cumhist = np.cumsum(content_counts)
        style_cumhist = np.cumsum(style_counts)
        # normalize it
        content_cumhist = content_cumhist / np.max(content_cumhist)
        style_cumhist = style_cumhist / np.max(style_cumhist)
        # match using interpolation
        matched = np.interp(content_cumhist, style_cumhist, style_values)
        transfered[:, :, i] = matched[content_indices].reshape(content[:, :, i].shape)
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