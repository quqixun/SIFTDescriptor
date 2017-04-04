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

# -------------------------------------------------
# Stage 1: Test some functions using a simple case
# -------------------------------------------------

# Generate a simple image to test
simple_test_img = np.arange(start=11, stop=101).reshape([9, 10]).T
# simple_test_img = np.random.normal(size=[9, 10]).T

# Get a patch from test image
test_patch = gs.get_patch(simple_test_img, 5, 5, 3)

# Filtering the test patch with gaussian filter
sigma = 3.0
test_patch_filt = gs.gaussian_filter(test_patch, sigma)

# Compute the gradient of test patch in x ang y direction
patch_grad_x, patch_grad_y = gs.gaussian_gradients(test_patch, sigma)

# Plot gradients on test patch
# gs.plot_gradients(test_patch, patch_grad_x, patch_grad_y)

# Compute the gradient histogram of test patch
histogram = gs.gradient_histogram(patch_grad_x, patch_grad_y)

# Plot bouqute of gradients histogram
# gs.plot_bouqute(histogram)

# -------------------------------------------------------------
# Stage 2: Test some functions using one image which has digit
# -------------------------------------------------------------

# The image set in digits,mat has 100 training images and
# 50 validation images, all images are in grayscale and lack of
# SIFT descriptors, the scale of each image is 39 by 39
train_set, validate_set = gs.read_data('Data/digits.mat')

# Set the index of an image in training set, this image is used
# int next few steps, the range of idx is from 1 to 100
idx = 27

# Extract the image from training set
train_img = train_set[0, idx - 1][0]
# print(train_set[0, idx][1])

# Set the position of the train image's centre
# Set the scale of the intersted patch that needs to be processed
# in this case, the entire training image is interested patch
position = [20, 20]
scale = 39

# Plot nine grids on training image
gs.plot_grides(train_img, position, scale)
