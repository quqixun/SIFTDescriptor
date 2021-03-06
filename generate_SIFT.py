# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/03
#

# This script provides functions for generating
# SIFT descriptors on images which consist of
# digits. This script and relative functions
# are tested under Python 3.5.2.


import numpy as np
import scipy.io as sio
import scipy.signal as ss
import scipy.ndimage as sn
import matplotlib.pyplot as plt


# Standard deviation scale to determin
# the size of gaussian filter
STD_SCALE = 0.05

# The number of grids in one patch
GRIDES_NUM = 9

# The number of gradient group in one grid
DESC_PER_GRID = 8

# The number of all gradient descriptors
# of one patch
DESC_ALL_NUM = GRIDES_NUM * DESC_PER_GRID


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


def plot_gradients(img, grad_x, grad_y):
    ''' PLOT_GRADIENT

        Plot gradient arrows on original image.

        Input arguments:

        - img : original image
        - grad_x : gradient in X direction
        - grad_y : gradient in Y direction

    '''

    X, Y = np.meshgrid(np.arange(0, grad_x.shape[0]),
                       np.arange(0, grad_x.shape[1]))
    U = grad_x
    V = grad_y

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.quiver(X, Y, U, -V, facecolor='red')
    plt.axis('image')
    plt.axis('off')
    plt.show()

    return


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
    hist_num = DESC_PER_GRID
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


def plot_bouqute(histogram):
    ''' PLOT_BOUQUTE

        Plot gradient histogram as a bouqute.

        Input argement:

        - histogram : the gradient histogram of an image

    '''

    angles = np.pi / DESC_PER_GRID + \
        np.arange(0, 8) * np.pi / (DESC_PER_GRID / 2)
    max_val = 0.1

    plt.figure()

    for i in range(DESC_PER_GRID):
        vec = histogram[i] * [np.cos(angles[i]), np.sin(angles[i])]
        plt.quiver(0, 0, vec[0], -vec[1], scale=1, units='y', facecolor='r')
        max_val = np.maximum(max_val, np.max(np.abs(vec)))

    plt.plot([-max_val, max_val], [0, 0], 'k:')
    plt.plot([0, 0], [-max_val, max_val], 'k:')

    plt.axis('off')
    plt.show()

    return


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

    # Visit ith image
    # print(validate_set[0, i - 1][0])
    # Visit ith label
    # print(validate_set[0, i - 1][1])

    return np.asanyarray(train_set), np.asanyarray(validate_set)


def place_regions(position, scale):
    ''' PLACE_REGION

        Return center's position and radius of each small square of a
        patch. Center's position indicates the coordinate of each small
        square in the full scale image.

        Input arguments:
        - position : position of the patch's center in full scale image
        - scale  : the size of the patch

        Outputs:
        - centres : 2 x 9 matrix records nine centers of 9 small squares
        - radius  : a number show the radius of each small square

        Input arguments and outputs could be shown as below

        |<---- scale ---->|
        |-----|-----|-----|----|--> radius
        |  *  |  *  |  *  |----|
        |-----|-----|-----|         X : position
        |  *  |  X  |  *  |
        |-----|-----|-----|         * & X : centres
        |  *  |  *  |  *  |
        |-----|-----|-----|

    '''

    # Calculate radius for each small square
    # (scale - 3) : since the center is not a part of radius, each row
    # and each column of patch consists of 6 radius and 3 centers
    # using function floor() is trying to ensure that the edge of image
    # will not be exceeded in the follwing operation
    radius = np.floor((scale - 3) / 6)

    # Set the relative center position, the relative center is
    # the centre of the patch, not in the full scale image
    c = 3 * radius + 1
    rc = np.array([[c, c]])

    # Set three numbers that can form all nine centers' position
    # all centers (except the midpoint) converge 1 pixel to the
    # midpoint toprepare for the overlap
    c_pos = np.array([c - 2 * radius, c, c + 2 * radius]) - 1
    x, y = np.meshgrid(c_pos, c_pos)

    # Calculate the distance between each center of samll squrae
    # and therelative center of the patch
    dis = np.array([y.flatten(), x.flatten()]) - rc.T
    # Calculate the real position of each small square in full scale image
    centres = dis + position.T

    # Now, radius pluses 1 that achives 2-pixel overlap
    radius += 1

    return np.int32(centres), np.int32(radius)


def plot_grides(img, obj_pos):
    ''' PLOT_GRIDES

        Plot nine small squares in full scale image.

        Input arguments:

        - img : the full scale image that will be plotted nine squares in
        - obj_pos : a n x 3 matrix, first two columns store the position of the
          patch's center, the third column stores the scale of the patch

    '''

    # Generate a new figure, plot the full scale image first
    plt.imshow(img, cmap='gray')

    # Obtain the number of patches, all patches are about to be segmented in
    # nine samll squares and also be plotted in full scale image
    for i in range(obj_pos.shape[0]):
        # For each patch, calculate the nine centers of samll squares and the
        # radius as well, plotting all nine squares in full scale image
        centres, radius = place_regions(obj_pos[i, 0], obj_pos[i, 1])

        for j in range(centres.shape[1]):
            cols = centres[0, j] + radius * np.array([-1, 1, 1, -1, -1])
            rows = centres[1, j] + radius * np.array([-1, -1, 1, 1, -1])
            plt.plot(cols, rows, 'r-')

        plt.axis('off')
        plt.show()

    return


def gradient_descriptor(img, obj_pos):
    ''' GRADIENT_DESCRIPTOR

        Return the SIFT_like descriptor of a patch in image.

        Input arguments:

        - img : a full scale image that contains the patch
        - obj_pos : a n x 3 matrix, first two columns store the position of the
          patch's center, the third column stores the scale of the patch

        Output:

        - desc : 72 x 1 vector indicates the SIFT-like descriptor of the patch

    '''

    # Standard deviation is propotional to the scale of patch
    stddev = np.round(obj_pos[0, 1] * STD_SCALE)

    # Calculate the gradients in x and y direction
    igx, igy = gaussian_gradients(img, stddev)

    # The patch is about to be devided into nine samll squares,
    # according to the position and the scale of patch, compute
    # nine centers and one radius for all small squares
    centres, radius = place_regions(obj_pos[0, 0], obj_pos[0, 1])

    # Initialize a vector to store descriptor
    # each small square has 8 descriptors, all
    # nine small squares give 72 descriptors
    desc = np.zeros([DESC_ALL_NUM, 1])
    for i in range(GRIDES_NUM):
        # Obtain X gradient and Y gradient for the ith small squares
        pgx = get_patch(igx, centres[0, i], centres[1, i], radius - 1)
        pgy = get_patch(igy, centres[0, i], centres[1, i], radius - 1)

        # Compute the descriptors for the ith small square
        desc[i * DESC_PER_GRID: (i + 1) * DESC_PER_GRID] = \
            gradient_histogram(pgx, pgy)

        # gs.plot_bouqute(desc[i * 8: (i + 1) * 8])

    # Normalized all values in descriptor
    desc /= np.linalg.norm(desc)

    return desc
