# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/05
#

# This script provides functions for generating
# SIFT descriptors on images which consist of
# digits. This script and relative functions
# are tested under Python 3.5.2.


import numpy as np
import generate_SIFT as gs


def prepare_digits(data_set, obj_pos):
    ''' PREPARE_DIGITS

        This function is to compute SIFT-like descriptors
        for each image in training set.

        Input argements:

        - data_set : a training set with 100 images
        - obj_pos : a matrix consists of the size of each
        training image and the position of centre

        Outputs:

        - data_desc : all descriptors of training images
        - data_labels : all labels for training images

    '''

    # Get the number of images in training set
    obj_num = data_set.shape[1]

    # Initialize outputs, in this case, there are 72
    # descriptors of every training image
    data_desc = np.zeros([1, obj_num, gs.DESC_ALL_NUM])
    data_labels = np.zeros([1, obj_num])

    for i in range(obj_num):
        # Calculate each training image's descriptor
        temp = gs.gradient_descriptor(data_set[0, i][0], obj_pos)
        data_desc[0, i] = temp.flatten()
        data_labels[0, i] = data_set[0, i][1]

    return data_desc, np.int32(data_labels)


def classify_digit(test, train_desc, train_labels, obj_pos):
    ''' CLASSIFY_DIGIT

        Get the label of one test image which consists of a digit.

        Input arguments:

        - test : a testing image contains one digit
        - train_desc, train_label : SIFT descriptors and labels
        for training images
        - obj_pos : a matrix consists of the size of each training
        image and the position of centre

        Output:

        - label : a number shows the digit in test set

    '''

    # Calculate the descriptor for test image
    # desc will be a 72 x 1 vector
    desc = gs.gradient_descriptor(test, obj_pos).flatten()

    # Initialize the distance to the maximum value, since the descriptor
    # vector has 72 elements that are normalized from 0 to 1, the maximum
    # distance between two descriptors is 72 according to the distance
    # equation: sum((desc1 - desc2)^2)
    dis = gs.DESC_ALL_NUM

    # Initialize the label for one test image, -1 is not included in
    # any posible results, if -1 appears in classification, which
    # means there are mistakes in the process of computing descriptors
    label = -1

    for i in range(train_desc.shape[1]):
        # Compare the descriptor of test image to each descriptor of
        # training image, calculate the distance between two descriptors
        dis_tmp = np.sum(np.power(desc - train_desc[0, i], 2))

        # Reserve the label of one training image that has
        # the minimum distance with the test image
        if dis_tmp < dis:
            dis = dis_tmp
            label = train_labels[0, i]

    return int(label)


def classify_all_digit(validate, train_desc, train_label, obj_pos):
    ''' CLASSIFY_ALL_DIGIT

        Classify all validation images in validation set and show
        classification result.

        Input argements:

        - validate : an image set contains 50 images with labels
        - train_desc, train_label : SIFT descriptors and labels
        for training images
        - obj_pos : a matrix consists of the size of each training
        image and the position of centre

    '''

    # Obtain the number of validation image
    obj_num = validate.shape[1]

    # Initialize the number of digits that
    # can be classified correctly
    right_pred = 0

    for i in range(obj_num):
        # Get label of ith validation image
        label = classify_digit(validate[0, i][0],
                               train_desc, train_label, obj_pos)

        # Accumelate the number of digits that
        # can be classified correctly
        if label == validate[0, i][1]:
            right_pred += 1

    # Compute classification accuracy and print result
    accuracy = right_pred / obj_num * 100

    print("Validate all {} images,".format(obj_num))
    print("{0:.2f}% validation images can be recognized.".format(accuracy))

    return
