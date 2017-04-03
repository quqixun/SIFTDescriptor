# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/03
#

# This script provides a test on generating
# SIFT descriptors on images which consist
# of digits. This script and relative
# functions are tested under Python 3.5.2.


import numpy as np
import generateSIFT as gs
import matplotlib.pyplot as plt


# Generate a simple image to test
simple_test_img = np.arange(start=11, stop=101).reshape([9, 10]).T

# Get a patch from test image
test_patch = gs.get_patch(simple_test_img, 5, 5, 3)

# Filtering the test patch with gaussian filter
sigma = 3.0
test_patch_filt = gs.gaussian_filter(test_patch, sigma)

# Compute the gradient of test patch in x ang y direction
patch_grad_x, patch_grad_y = gs.gaussian_gradients(test_patch, sigma)

# Plot gradients on test patch
# X, Y = np.meshgrid(np.arange(0, 7), np.arange(0, 7))
# U = patch_grad_x
# V = patch_grad_y

# plt.figure()
# plt.imshow(test_patch, cmap='gray')
# plt.quiver(X, Y, U, -V, facecolor='red')
# plt.axis('off')
# plt.show()

# Compute the gradient histogram of test patch
histogram = gs.gradient_histogram(patch_grad_x, patch_grad_y)

# The image set in digits,mat has 100 training images and
# 50 validation images, all images are in grayscale and lack of
# SIFT descriptors, the scale of each image is 39 by 39
train_set, validate_set = gs.read_data('Data/digits.mat')
