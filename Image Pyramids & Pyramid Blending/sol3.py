import os
import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from scipy import signal
from matplotlib import pyplot as plt


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


# ------------------ 3.1 Gaussian and Laplacian pyramid construction ------------------


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


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    This function constructs a Laplacian pyramid
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
    (an odd scalar that represents a squared filter) to be used in
    constructing the pyramid filter
    :return: pyr: the resulting pyramid as a list, with maximum length of max_levels,
                  where each element of the array is a grayscale image.
             filter_vec: row vector of shape (1, filter_size) used for the pyramid construction.
    """
    gaussian_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = list()
    for i in range(len(gaussian_pyr) - 1):
        l_i = gaussian_pyr[i] - blur(expand(gaussian_pyr[i+1]), 2*filter_vec)
        pyr.append(l_i)
    pyr.append(gaussian_pyr[len(gaussian_pyr) - 1])
    return pyr, filter_vec


# ------------------ 3.2  Laplacian pyramid reconstruction ------------------


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: the Laplacian pyramid generated by build_laplacian_pyramid, a python list
    :param filter_vec: the filter vector generated by build_laplacian_pyramid
    :param coeff: a python list.
    :return: img - the original image.
    """
    im_orig_shape = lpyr[0].shape
    sum = lpyr[0] * coeff[0]
    for i in range(1, len(lpyr)):
        while lpyr[i].shape != im_orig_shape:
            lpyr[i] = blur(expand(lpyr[i]), 2*filter_vec)
        lpyr[i] *= coeff[i]
        sum += lpyr[i]
    return sum


# ------------------ 3.3  Pyramid display ------------------
def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width))
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa+wb] = imgb
    return new_img


def render_pyramid(pyr, levels):
    """
    Renders the pyramid after stretching all the images in pyr to the range [0,1]
    :param pyr: the pyramid (either gaussian or laplacian)
    :param levels: number of levels in the pyramid
    :return: an image of the pyramids images - stacked horizontally
    """
    output = None
    for i in range(levels):
        if i == 0:
            output = normalize_data(pyr[0])
        else:
            output = concat_images(output, normalize_data(pyr[i]))
    return output


def display_pyramid(pyr, levels):
    """
    Displays the pyramid in one figure.
    :param pyr: the pyramid (either gaussian or laplacian)
    :param levels: number of levels in the pyramid
    :return: the pyramid image in one figure
    """
    res = render_pyramid(pyr, levels)
    show_img(res)


def normalize_data(data):
    """
    This function stretches the values of data to the range [0,1]
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def show_img(img):
    """
    Helper function to show the image.
    """
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()


# ------------------ 4 Pyramid Blending ------------------
def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    :param im1: image to be blended
    :param im2: image to be blended
    :param mask: an np.bool image
    :param max_levels: the maximal number of levels in the pyramid.
    :param filter_size_im: size of the gaussian filter for im1 and im2
    :param filter_size_mask: size of the gaussian filter for the mask
    :return: an image where im1 is blended into im2 according to the mask
    """
    # (1) Construct Laplacian pyramids L1 and L2 for the input images im1 and im2, respectively.
    lpyr1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)[0]
    lpyr2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]

    # (2) Construct a Gaussian pyramid Gm for the provided mask
    gpyr_mask = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)[0]

    # (3) Construct the Laplacian pyramid Lout of the blended image for each level k by:
    lout = list()
    coeff = list()
    for k in range(len(lpyr1)):
        lout.append(gpyr_mask[k] * lpyr1[k] + (1 - gpyr_mask[k]) * lpyr2[k])
        coeff.append(1)

    # (4) Reconstruct the resulting blended image from the Laplacian pyramid Lout

    im_blend = laplacian_to_image(lout, filter_construction(filter_size_im), coeff)

    return im_blend

# ------------------ 4.1  Blending examples ------------------


def blending(im1, im2, mask):
    im_blend = np.zeros(im1.shape)
    for i in range(3):
        im_blend[:, :, i] = np.clip(pyramid_blending(im1[:, :, i], im2[:, :, i], mask, 5, 9, 5), 0, 1)

    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].imshow(im2)
    axarr[0, 1].imshow(im1)
    axarr[1, 0].imshow(mask, cmap=plt.cm.gray)
    axarr[1, 1].imshow(im_blend)

    plt.show()

    return im1, im2, mask.astype(np.bool), im_blend


def blending_example1():
    im1 = read_image(relpath("bibi_trump.jpg"), 2)
    im2 = read_image(relpath("chocolit.jpg"), 2)
    mask = np.round(read_image(relpath("bibi_trump_mask.jpg"), 1))
    return blending(im1, im2, mask)


def blending_example2():
    im1 = read_image(relpath("noga.jpg"), 2)
    im2 = read_image(relpath("eiffel.jpg"), 2)
    mask = np.round(read_image(relpath("noga_mask.jpg"), 1))
    return blending(im1, im2, mask)


# ------------------------------------------------------------------------------------------------

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