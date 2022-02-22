import numpy as np
from imageio import imread
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

###################################################
#                    Constants                    #
###################################################

RGB_TO_YIQ_MAT = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])
YIQ_TO_RGB_MAT = np.linalg.inv(RGB_TO_YIQ_MAT)

##################################################
#                 The Exercise                   #
##################################################

# 3.1 Toy Example
x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255]*6)[None, :]])
grad = np.tile(x, (256, 1))


# 3.2 Reading an image into a given representation


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


# 3.3 Displaying an image


def imdisplay(filename, representation):
    """
    This function utilizes the function read_image to display an image
    in a given representation.
    :param filename: the file name of an image on disk
    :param representation: representation code, can be either 1 or 2
    when 1 defines a grayscale image, and 2 defines an RGB image.
    """
    im = read_image(filename, representation)
    plt.imshow(im, cmap='gray')
    plt.show()


# 3.4 Transforming an RGB image to YIQ color space


def rgb2yiq(imRGB):
    """
    :param imRGB: an RGB image matrix of shape height x width x np.float64
    :return: a YIQ representation matrix of the given image matrix
    """
    im_original_shape = imRGB.shape
    return np.dot(imRGB.reshape(-1, 3), RGB_TO_YIQ_MAT.T).reshape(im_original_shape)


def yiq2rgb(imYIQ):
    """
    :param imYIQ: a YIQ image matrix of shape height x width x np.float64
    :return: an RGB representation matrix of the given image matrix
    """
    im_original_shape = imYIQ.shape
    return np.dot(imYIQ.reshape(-1, 3), YIQ_TO_RGB_MAT.T).reshape(im_original_shape)


# 3.5 Histogram Equalization


def stretch_function(c_k, c_m, c_255):
    """
    The stretch function according to lecture
    :param c_k: a cumulative normalized histogram
    :param c_m: first non zero gray level value
    :param c_255: last gray level value
    :return:
    """
    return ((c_k - c_m) / (c_255 - c_m)) * 255


def histogram_equalize(im_orig):
    """
    :param im_orig: The input grayscale or RGB float64 image with values in [0, 1]
    :return: a list [im_eq, hist_orig, hist_eq], where:
             im_eq - the image after equalization
             hist_orig - the histogram of the original image
             hist_eq - the equalized histogram of the original image
    """
    # (1) Compute the image histogram
    im_orig *= 255

    if im_orig.ndim == 3:
        im_yiq = rgb2yiq(im_orig)
        img = im_yiq[:, :, 0]  # Taking the Y channel
    else:
        img = im_orig

    hist_orig, bins = np.histogram(img.flatten(), 256, (0, 255))

    # (2) Compute the cumulative histogram
    cumulative_hist = np.cumsum(hist_orig)

    # (3) Normalize the cumulative histogram
    n_pixels = cumulative_hist[-1]
    cumulative_normalized_hist = cumulative_hist / n_pixels

    # (4) Multiply the normalized histogram by 255
    cumulative_normalized_hist *= 255


    # (5) Verify that the minimal value is 0 and that the maximal is 255,
    # otherwise stretch the result linearly in the range [0, 255].

    stretched = False
    first_non_zero = cumulative_hist[np.nonzero(cumulative_hist)[0][0]]
    last_grey_lvl = cumulative_hist[255]

    if hist_orig[0] == 0 or hist_orig[255] == 0:
        stretched = True
        cumulative_normalized_hist = stretch_function(cumulative_hist,
                                                      first_non_zero,
                                                      last_grey_lvl)

    # (6) Round the values to get integers
    lut = cumulative_normalized_hist.astype(int)

    # (7) Map the intensity values of the image using the result of step 6.
    if stretched:
        hist_eq = np.zeros((256,))
        hist_eq[lut] += hist_orig
    else:
        hist_eq = lut

    if im_orig.ndim == 3:
        im_yiq[:, :, 0] = lut[img.astype(int)]
        im_eq = yiq2rgb(im_yiq)
        im_eq = np.rint(im_eq)
    else:
        im_eq = lut[img.astype(int)]

    im_eq = im_eq.astype(np.float64) / 255

    return [im_eq, hist_orig, hist_eq]


# 3.6 Optimal Image Quantization


def error_func(h, q, z, n_quant):
    """
    The error function as specified in the recitation.
    :return: The error on the given q and z.
    """
    err = 0
    z = z.astype(int)
    for i in range(n_quant):
        g = np.arange(z[i] + 1, z[i + 1]).astype(int)
        err += ((np.full((z[i+1] - z[i] - 1).astype(int), q[i]).astype(int) - g)**2).T @ h[g]
    return err


def initialize_z(hist_orig, z, n_quant, n_pixels):
    """
    Does the initialization of the z components
    such that in each segment [z_(i),z_(i+1)] there will
    be approximately equal number of pixels.
    :return: z
    """
    i = 0
    for k in range(1, n_quant):
        hist_temp = hist_orig[i:]
        z[k] = np.where(np.cumsum(hist_temp) >= n_pixels)[0][0] + i
        i = z[k].astype(int)
    return z


def quantized_hist(q, z, n_quant):
    """
    Calculating the histogram of the image after quantization.
    :return: quantized hist
    """
    hist = np.zeros((256, ))
    for i in range(n_quant):
        hist[z[i]+1: z[i+1] + 1] = q[i]
    return hist.astype(int)


def quantization_iters(hist_orig, q, z, n_iter, n_quant):
    """
    Does the iterations of the error's minimization
    in the quantization functions.
    :return: z, q, error
    """
    error = list()
    for i in range(n_iter):
        for j in range(n_quant):
            g = np.arange(z[j] + 1, z[j + 1] + 1).astype(int)
            q[j] = (g.T @ hist_orig[g]) / np.sum(hist_orig[g])
        for j in range(1, n_quant):
            z[j] = (q[j - 1] + q[j]) / 2
        err = error_func(hist_orig, q, z, n_quant)
        if i == 0:
            error.append(err)
        else:
            if err != error[i - 1]:
                error.append(err)
            else:
                break
    return z, q, error


def quantize(im_orig, n_quant, n_iter):
    """
    :param im_orig: The input grayscale or RGB float64 image with values in [0, 1]
    :param n_quant: The number of intensities the output im_quant image should have
    :param n_iter:  The maximum number of iterations of the optimization procedure
    :return:  a list [im_quant, error] where
              im_quant - is the quantized output image.
              (float64 image with values in [0, 1]).
              error - is an array with shape (n_iter,) (or less) of the
              total intensities error for each iteration of the quantization procedure.
    """
    im_orig *= 255

    if im_orig.ndim == 3:
        im_yiq = rgb2yiq(im_orig)
        img = im_yiq[:, :, 0]  # Taking the Y channel
    else:
        img = im_orig

    hist_orig, bins = np.histogram(img.flatten(), 256, (0, 255))
    z = np.zeros((n_quant + 1, ))
    z[0] = -1
    z[n_quant] = 255
    q = np.zeros((n_quant, ))

    # first initialization of the segments

    z = initialize_z(hist_orig, z, n_quant, np.cumsum(hist_orig)[-1] / n_quant)
    for i in range(n_quant):
        q[i] = (z[i] + z[i+1]) / 2

    # The iterations of the quantization algorithm
    z, q, error = quantization_iters(hist_orig, q, z, n_iter, n_quant)

    # restore histogram of the quantized image
    hist_quant = quantized_hist(q, z.astype(int), n_quant)

    if im_orig.ndim == 3:
        im_yiq[:, :, 0] = hist_quant[img.astype(int)]
        im_quant = yiq2rgb(im_yiq)
        im_quant = im_quant.astype(int).astype(float) / 255
    else:
        im_quant = hist_quant[img.astype(int)].astype(int).astype(float) / 255

    return [im_quant, np.array(error)]
