import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from scipy.io import wavfile

###################################
#          The Exercise           #
###################################

# -------------------------------- 1.1 1D DFT --------------------------------


def DFT(signal):
    """
    :param signal: an array of dtype float64 with shape (N,) or (N,1)
    :return: complex Fourier signal, of the same shape as signal and of dtype complex128.
    """
    orig_shape = signal.shape
    signal = np.squeeze(signal)
    n = signal.shape[0]
    phi = np.exp(-2j * np.pi / n)
    phi_matrix = np.vander(np.power(phi, np.arange(n)), increasing=True)
    return (phi_matrix @ signal).astype(np.complex128).reshape(orig_shape)


def IDFT(fourier_signal):
    """
    :param fourier_signal:  an array of dtype complex128 with the shape (N,) or (N,1)
    :return: complex signal, of the same shape as fourier_signal, also of dtype complex128.
    """
    orig_shape = fourier_signal.shape
    fourier_signal = np.squeeze(fourier_signal)
    n = fourier_signal.shape[0]
    phi = np.exp(2j * np.pi / n)
    phi_matrix = np.vander(np.power(phi, np.arange(n)), increasing=True)
    return ((1/n) * (phi_matrix @ fourier_signal)).astype(np.complex128).reshape(orig_shape)


# -------------------------------- 1.2 2D DFT --------------------------------


def DFT2(image):
    """
    :param image: grayscale image of dtype float64, of shape (M,N) or (M,N,1).
    :return: complex fourier image, of the same shape as image and of dtype complex128.
    """
    orig_shape = image.shape
    image = np.squeeze(image)
    m = image.shape[0]
    n = image.shape[1]
    fourier_image_tmp = np.zeros(image.shape).astype(np.complex128)
    for u in range(m):
        fourier_image_tmp[u] = DFT(image[u])
    fourier_image = np.zeros(image.shape).astype(np.complex128)
    for v in range(n):
        fourier_image[:, v] = DFT(fourier_image_tmp[:, v])
    return fourier_image.reshape(orig_shape)


def IDFT2(fourier_image):
    """
    :param fourier_image: 2D array of dtype complex128, of shape (M,N) or (M,N,1).
    :return: complex image, of the same shape as fourier_signal, also of dtype complex128.
    """
    orig_shape = fourier_image.shape
    fourier_image = np.squeeze(fourier_image)
    m = fourier_image.shape[0]
    n = fourier_image.shape[1]
    image_tmp = np.zeros(fourier_image.shape).astype(np.complex128)
    for u in range(m):
        image_tmp[u] = IDFT(fourier_image[u])
    image = np.zeros(fourier_image.shape).astype(np.complex128)
    for v in range(n):
        image[:, v] = IDFT(image_tmp[:, v])
    return image.reshape(orig_shape)


# -------------------------------- 2.1 Fast forward by rate change --------------------------------
def change_rate(filename, ratio):
    """
    :param filename: a path to a wav file
    :param ratio: a float64 number, 0.25 < ratio < 4
    changes the sample rate of the given file by the given ratio.
    """
    samplerate, data = wavfile.read(filename)
    wavfile.write("change_rate.wav", np.floor(samplerate * ratio).astype(int), data)


# -------------------------------- 2.2 Fast forward using Fourier --------------------------------
def resize(data, ratio):
    """
    :param data: 1D ndarray of dtype float64 or complex128
                 representing the original sample points
    :param ratio: a float number, 0.25 < ratio < 4
    :return: 1D ndarray of the dtype of data representing the new sample points
    """
    if ratio == 1:
        return data

    # (1) Compute DFT
    n = data.shape[0]
    fourier_signal = DFT(data)

    # (2) Shift to center
    fourier_signal = np.fft.fftshift(fourier_signal)

    # (3) Crop fourier according to ratio
    new_n = np.floor(n / ratio).astype(int)
    if ratio < 1:
        # increase number of samples
        to_pad = new_n - n
        left = np.zeros((to_pad - round(to_pad / 2)).astype(int))
        right = np.zeros((round(to_pad / 2)).astype(int))
        fourier_signal = np.concatenate((left, fourier_signal, right))
    else:
        # reduce number of samples
        to_reduce = n - new_n
        fourier_signal = fourier_signal[(to_reduce - round(to_reduce / 2)).astype(int):]
        fourier_signal = fourier_signal[:(fourier_signal.size - round(to_reduce / 2)).astype(int)]

    # (4) Compute IDFT
    return IDFT(np.fft.ifftshift(fourier_signal))


def change_samples(filename, ratio):
    """
    :param filename: a path to a wav file
    :param ratio: a float number, 0.25 < ratio < 4
    uses resize function to resize the number of the data's samples of
    the given wav file by the given ratio.
    :return: the new data numpy array of dtype float64
    """
    samplerate, data = wavfile.read(filename)
    new_data = np.real(resize(data, ratio))
    wavfile.write("change_samples.wav", samplerate, new_data)
    return new_data.astype(np.float64)


# -------------------------------- 2.3 Fast forward using Spectrogram --------------------------------

def resize_spectrogram(data, ratio):
    """
    :param data: 1D ndarray of dtype float64 representing the original sample points
    :param ratio: a positive float64 representing the rate change of the WAV file
    :return: the new sample points according to ratio with the same datatype as data.
    """
    # (1) compute the spectrogram
    stft_mat = stft(data)
    new_n = np.floor(stft_mat.shape[1] / ratio).astype(int)
    resized_mat = np.zeros((stft_mat.shape[0], new_n))

    # (2) change the number of spectrogram's columns
    for row in range(stft_mat.shape[0]):
        resized_mat[row] = np.real(resize(stft_mat[row], ratio))

    # (3) creating back the audio
    return istft(resized_mat)


# --------------------------- 2.4 Fast forward using Spectrogram and phase vocoder ---------------------------
def resize_vocoder(data, ratio):
    """
    :param data: a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: a positive float64 representing the rate change of the WAV file
    :return:  the given data rescaled according to ratio with the same datatype as data
    """
    return istft(phase_vocoder(stft(data), ratio))


# -------------------------------- 3.1 Image derivatives in image space --------------------------------
def conv_der(im):
    """
    magnitude of the image, using convolution derivatives
    :param im: grayscale images of type float64
    :return: the magnitude of the derivative, with the same dtype and shape
    """
    dx_filter = np.array([[-0.5, 0, 0.5]])
    dy_filter = dx_filter.T
    dx = signal.convolve2d(im, dx_filter, mode='same')
    dy = signal.convolve2d(im, dy_filter, mode='same')
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude


# -------------------------------- 3.2  Image derivatives in Fourier space --------------------------------
def fourier_der(im):
    """
    magnitude of the image, using fourier derivatives
    :param im: grayscale images of type float64
    :return: the magnitude of the derivative, with the same dtype and shape
    """
    # compute DFT of im and shift to center
    n = im.shape[0]
    m = im.shape[1]
    fourier_signal = np.fft.fftshift(DFT2(im))
    dx = np.zeros(im.shape).astype(np.complex128)
    dy = np.zeros(im.shape).astype(np.complex128)

    # multiply each row by fourier
    u = np.arange(round(-n / 2), np.ceil(n/2).astype(int))
    v = np.arange(round(-m / 2), np.ceil(m/2).astype(int))
    for row in range(n):
        dx[row] = fourier_signal[row] * u[row]

    # multiply each column by fourier
    for col in range(m):
        dy[:, col] = fourier_signal[:, col] * v[col]

    # shift back and compute IDFT of both DFT's
    dx = (2j * np.pi / n) * IDFT2(np.fft.ifftshift(dx))
    dy = (2j * np.pi / m) * IDFT2(np.fft.ifftshift(dy))

    # compute magnitude
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude

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
    im_float = im.astype(np.float64)
    im_float /= 255

    if representation == 1:
        # if the picture is rgb -> turn to gray scale
        if im_float.ndim == 3:
            im_gray = rgb2gray(im_float).astype(np.float64)
            return im_gray
        # if the picture is gray -> leave it gray
        return im_float


####################################
#          ex2_helper.py           #
####################################
def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec
