from sol1 import *


def histeq(im,nbr_bins=256):
    # get image histogram
    imhist,bins = np.histogram(im.flatten(),nbr_bins, (0, 255))
    cdf = np.cumsum(imhist)  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    im2 = im2.reshape(im.shape) / 255

    return im2.astype(np.float64)


def show_img(img):
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()


def show_hist(img):
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.show()


if __name__ == '__main__':

    #########################################
    #     Histogram Equalization Testing    #
    #########################################

    images = [read_image("jerusalem.jpg", 1),
              read_image("monkey.jpg", 2),
              read_image("Unequalized_Hawkes_Bay_NZ.jpg", 1)]
    for im in images:
        # Original image and histogram
        show_img(im)
        show_hist(im * 255)

        # Image and histogram after equalization
        im_eq = histogram_equalize(im)[0]
        show_img(im_eq)
        show_hist(im_eq * 255)

        # Quantization of equalized image
        lst = quantize(im_eq, 4, 10)
        show_img(lst[0])
        show_hist(lst[0] * 255)

    # # im = read_image("Unequalized_Hawkes_Bay_NZ.jpg", 1)
    # im = read_image("dog.jpg", 2)
    # # im = grad / 255
    # show_img(im)
    # show_hist(im * 255)
    # im_eq = histogram_equalize(im)[0]
    # show_img(im_eq)
    # show_hist(im_eq * 255)
    # # #print("Equalized image matrix:\n", histogram_equalize_test(grad)[0])
    # # #print("Original hist:\n", histogram_equalize_test(grad)[1])
    # #
    # # #########################################
    # # #         Quantization Testing          #
    # # #########################################
    # #
    # #
    # lst = quantize(im_eq, 5, 10)
    # show_img(lst[0])
    # show_hist(lst[0] * 255)







