# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/03
#

# This script provides a test on generating
# SIFT descriptors on images which consist
# of digits. This script and relative
# functions are tested under Python 3.5.2.


import numpy as np
import scipy.io as sio
import scipy.signal as ss
import scipy.ndimage as sn


def get_patch(img, x, y, patch_radius):
    ''' GET_PATCH

        Get partial image at the centre of (x, y) with the radius.

        Input arguments:

        - img : a full scale image contains the wanted patch
        - x, y  : position of the patch's centre in full scale image
        - patch_radius : the distance between patch's centre and patch's edge

        Output:

        - patch : a square image with the width of (2 * patch_radius + 1)

    '''

    if x - 1 + patch_radius > img.shape[1] or x - 1 - patch_radius < 0:
        # Edge of patch is out of the range of image in X direction
        print("Patch outside image border in X direction.")
    elif y - 1 + patch_radius > img.shape[0] or y - 1 - patch_radius < 0:
        # Edge of patch is out of the range of image in Y direction
        print("Patch outside image border in Y direction.")
    else:
        # Extract patial image from full scale image
        return img[y - 1 - patch_radius: y + patch_radius,
                   x - 1 - patch_radius: x + patch_radius]


def gaussian_filter(img, sigma=1.0):
    ''' GAUSSIAN_FILTER

        Filtering image with gaussian filter.

        Input argements:

        - img : an image to be filtered
        - sigma : standard deviation determins the width of normal
                  distribution and the size of the filter

        Output:

        - img_gau : the filtered image

    '''

    # Initialize output
    img_gau = np.zeros(img.shape)

    # Filtering image by a Gaussian filter, the image boundary is
    # padded by its mirror-reflecting
    sn.gaussian_filter(img, sigma=sigma, output=img_gau, mode='mirror')

    return img_gau


def gaussian_gradients(img, sigma=1.0):
    ''' GAUSSIAN_GRADIENTS

        Obtain the gradients in X and Y direction respectively of the
        image that is smoothed by a Gaussian filter.

        Input argements:

        - img : the image to be extracted gradients
        - sigma : standard deviation determins the width of normal
                  distribution and the size of the filter

        Outputs:

        - grad_x : the gradient in X direction
        - grad_y : the gradient in Y direction

    '''

    # Obtain filtered image first to keep information that will not
    # change with scale, which is also known as scale-invariant features
    img_gau = gaussian_filter(img, sigma)

    # Define a kernal for convolution
    kernal = np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]])

    # Calculate gradients in X and Y direction, the image boundary
    # is padded by its mirror-reflecting
    grad_y = ss.convolve2d(img_gau, kernal.T, 'same', 'symm')
    grad_x = ss.convolve2d(img_gau, kernal, 'same', 'symm')

    return grad_x, grad_y


def gradient_histogram(grad_x, grad_y):
    ''' GRADIENT_HISTOGRAM

        Calculate the gradient histogram for given gradients in
        X and Y directions.

        Input arguments:

        - grad_x : X direction gradient
        - grad_y : Y direction gradient

        Output:

        - histogram : 8 by 1 vector consists of the sum of length
                      in 8 gradient directions, shown as follows:

               y
            \ 6|7 /      Group & Interval:
            5\ | /8      1: [-1, 0)   5: [3, 4)
          ____\|/____x   2: [-2, -1)  6: [2, 3)
              /|\        3: [-3, -2)  7: [1, 2)
            4/ | \1      4: [-4, -3)  8: [0, 1)
            / 3|2 \

    '''

    # Compute the direction of the sum vector
    # The negative sign of grad_y: since the orientation of y axis in
    # an image is downward, the direction of grad_y in odinary coordinate
    # shoule be the opposite to the direction in image coordinate
    # The tan^-1 of -grad_y is devided by (pi / 4), resulting in
    # eight intervals that can be presented by intergers as shown above
    all_atan = np.arctan2(-grad_y, grad_x) / (np.pi / 4)

    # Compute the length of sum vector of each point
    all_length = np.sqrt(np.power(grad_x, 2) + np.power(grad_y, 2))

    # Initialize the vector to hold histogram
    hist_num = 8
    histogram = np.zeros([hist_num, 1])

    # There are four iterations, in
    # 1st: obtain group 1 & 8
    # 2nd: obtain group 2 & 7
    # 3rd: obtain group 3 & 6
    # 4th: obtain group 4 & 5
    for i in range(int(hist_num / 2)):
        temp = np.logical_and(all_atan >= i, all_atan < i + 1)
        histogram[hist_num - 1 - i] = \
            np.sum(np.multiply((temp == 1) * 1., all_length))

        temp = np.logical_and(all_atan >= -(i + 1), all_atan < -i)
        histogram[i] = np.sum(np.multiply((temp == 1) * 1., all_length))

    return histogram


def read_data(path):
    ''' READ_DATA

        Load training image set and validation image set from .mat file.

        Input argument:

        - path : the .mat file path

        Outputs:

        - train:set : the training set
        - validate_set : the validation set

    '''

    # Load data from given path
    data = sio.loadmat(path)

    # Read matrix by their variable name
    train_set = data['digits_training']
    validate_set = data['digits_validation']

    # Visit first image
    # print(validate_set[0, 0][0])
    # Visit first label
    # print(validate_set[0, 0][1])

    return train_set, validate_set