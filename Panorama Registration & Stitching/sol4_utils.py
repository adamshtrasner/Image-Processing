import numpy as np
from scipy import signal
from imageio import imread
from skimage.color import rgb2gray
from scipy.signal import convolve2d


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


###############################
#          From Ex1           #
###############################
def read_image(filename, representation):
    """
    This function reads an image into a given representation.
    :param filename: the file name of an image on disk
    :param representation: representation code, can be either 1 or 2
    when 1 defines a grayscale image, and 2 defines an RGB image.
    :return: an image, according to the given representation
    """
    im = imread(filename)
    im_float = np.true_divide(im, 255).astype(np.float64)

    if representation == 1:
        im_float = im.astype(np.float64)
        im_float /= 255
        # if the picture is rgb -> turn to gray scale
        if im_float.ndim == 3:
            im_gray = rgb2gray(im_float).astype(np.float64)
            return im_gray
        # if the picture is gray -> leave it gray
        return im_float
    else:
        return im_float


###############################
#          From Ex4           #
###############################
def filter_construction(filter_size):
    base_filter_vec = 0.5 * np.array([1, 1])
    if filter_size == 2:
        return np.array([base_filter_vec])
    filter_vec = base_filter_vec
    while len(filter_vec) != filter_size:
        filter_vec = signal.convolve(base_filter_vec, filter_vec)
    return np.array([filter_vec])


def blur(im, filter):
    return signal.convolve2d(im, signal.convolve2d(filter, filter.T), mode='same')


def reduce(im_blured):
    n_rows = im_blured.shape[0]
    n_cols = im_blured.shape[1]
    return im_blured[1:n_rows:2, 1:n_cols:2]


def expand(im):
    # pad zeros between every two rows and two columns
    im_pad = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    i, j = 0, 0
    while i < im.shape[0] and j < im_pad.shape[0]:
        im_pad[j][::2] = im[i]
        i += 1
        j += 2

    # blur after expansion
    return im_pad


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    This function constructs a Gaussian pyramid
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size:  the size of the Gaussian filter
    (an odd scalar that represents a squared filter) to be used in
    constructing the pyramid filter
    :return: pyr: the resulting pyramid as a list, with maximum length of max_levels,
                  where each element of the array is a grayscale image.
             filter_vec: row vector of shape (1, filter_size) used for the pyramid construction.
    """
    filter_vec = filter_construction(filter_size)
    pyr = list()
    pyr.append(im)
    if max_levels == 1:
        return pyr, filter_vec
    for i in range(1, max_levels):
        g_i = reduce(blur(pyr[i-1], filter_vec))
        if g_i.shape[0] < 16 or g_i.shape[1] < 16:
            break
        pyr.append(g_i)
    return pyr, filter_vec