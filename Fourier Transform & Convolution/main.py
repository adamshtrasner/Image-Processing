from sol2 import *
from matplotlib import pyplot as plt
import ex2_presubmit


def test_DFT1_and_IDFT1():
    arr = np.arange(10)
    assert np.all(np.isclose(DFT(arr), np.fft.fft(arr)))
    assert np.all(np.isclose(IDFT(arr), np.fft.ifft(arr)))


def test_DFT2_and_IDFT2(im):
    assert np.all(np.isclose(np.fft.fft2(im), DFT2(im)))
    assert np.all(np.isclose(np.fft.ifft2(im), IDFT2(im)))


def show_img(img):
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()


def resize_test(data, ratio):
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
    return np.real(IDFT(fourier_signal))


def test_resize_spectogram(filename, ratio):
    samplerate, data = wavfile.read(filename)
    new_data = np.real(resize_spectrogram(data, ratio)).astype(np.int16)
    wavfile.write("resize_spectrogram_test_0.5.wav", samplerate, new_data)


def test_resize_vocoder(filename, ratio):
    samplerate, data = wavfile.read(filename)
    new_data = np.real(resize_vocoder(data, ratio)).astype(np.int16)
    wavfile.write("resize_vocoder_test_0.5.wav", samplerate, new_data)


if __name__ == '__main__':
    # Testing DFT and IDFT
    test_DFT1_and_IDFT1()
    im = read_image("monkey.jpg", 1)
    test_DFT2_and_IDFT2(im)

    # Testing changing rate, resizing spectrogram, resizing vocoder
    # and changing samples
    change_rate("external/aria_4kHz.wav", 1.5)
    test_resize_spectogram("gettysburg10.wav", 2)
    test_resize_vocoder("gettysburg10.wav", 2)
    change_samples("external/aria_4kHz.wav", 1.5)

    # Testing derivatives
    show_img(im)
    show_img(conv_der(im))
    show_img(fourier_der(im))
