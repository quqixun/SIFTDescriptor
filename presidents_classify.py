# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/08
#


# This script is a demo to recogmize presidents
# of five contries by matching SIFT descriptors.
# First, generate SIFT descriptor from 50 training
# images. Given a test image, extract its SIFT
# and compare with all known SIFT descriptors to
# find out the best match.


# Import OpenCV library
# NOTE: OpenCV library consists of opencv_contrib
# Access to git: https://github.com/opencv
# Install instruction:
# http://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html
import cv2
import numpy as np
import pandas as pd


# Each feature point has 128 SIFT descriptors
SIFT_DES_NUM = 128

# There are 5 groups to be classified
GROUP_NUM = 5

# The ration of the best match over second best match
#      distance of best match
# ------------------------------- <= MATCH_RATIO
#  distance of second best match
MATCH_RATIO = 0.7


def extract_SIFT(path):
    ''' EXTRACT_SIFT

        Extract SIFT descriptors from the image whose file path
        has been given.

        Input argument:

        - path : the file path of iname to be processed

        Out put:

        - desc : all SIFT descriptors of the image, its dimension
        will be n by 128 where n is the number of feature points

    '''

    # Load the image in grayscale
    img = cv2.imread(path, 0)

    # Extract SIFT descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    _, desc = sift.detectAndCompute(img, None)

    return desc


def create_train_set():
    ''' CREATE_TRAIN_SET

        This function is used to extract all SIFT descriptors
        from 50 training images to form the training set for
        matching new image.

        The output is a file contains descriptors and their labels.
        You may find the model file in the folder "Data" which names
        as "train.npz".

    '''

    # Load paths of training images
    # The label of each image can be found in president.csv
    folder_path = 'Data/president/'
    data = pd.read_csv('Data/president.csv')
    files = data['Image']
    index_code = data['Index']

    # Initialize a matrix to keep descriptors
    all_desc = np.array([]).reshape(0, SIFT_DES_NUM)

    # Initialize a vector to save label for each descriptor
    all_labs = np.array([]).reshape(0, 1)

    for i in range(len(index_code)):
        # For each image, extract its SIFT descriptors first
        file_path = folder_path + files[i]

        desc = extract_SIFT(file_path)
        all_desc = np.concatenate((all_desc, desc), axis=0)

        # Put labels into vector
        all_labs = np.append(all_labs, np.ones(
            [desc.shape[0], 1]) * index_code[i])

    # Before writting the file, convert the formation
    # of two variables
    all_desc = np.float32(all_desc)
    all_labs = np.int32(all_labs)

    # Write the file
    np.savez('Data/train.npz', desc=all_desc, labs=all_labs)

    return


def match_SIFT(desc):
    ''' MATCH_SIFT

        Given a set of SIFT descriptors of one image, match in
        training set, find out the person in the image.

        Input argument:

        - desc : the descriptors of image

        Output:

        - the name of the person in the image

    '''

    # Load traing set data, including descriptors and labels
    data1 = np.load('Data/train.npz')
    all_desc = data1['desc']
    all_labs = data1['labs']

    # Read a csv that indicates the relation between label and
    # person, such as:
    # 1 for Xi Jinping
    # 2 for Donald John Trump
    # 3 for Vladimir Putin
    # 4 for Francois Hollande
    # 5 for Theresa Mary May
    data2 = pd.read_csv('Data/index_name.csv')

    # Match the given descriptors and obtain two best matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc, all_desc, k=2)

    # Obtain the good match if the ration id smaller than 0.7
    good = []
    for m, n in matches:
        if m.distance <= MATCH_RATIO * n.distance:
            good.append(all_labs[m.trainIdx])

    # Count the best matches, the result of this section
    # looks like:
    # Label      Occurrence
    #   1			15
    #   2           129
    #   3           29
    #   4           40
    #   5           9
    good_count = np.zeros(GROUP_NUM)
    for i in range(GROUP_NUM):
        good_count[i] = good.count(i + 1)

    # Obtain the label that has the largest occurrence number
    # this number is the label for the given image
    fit_idx = np.argmax(good_count)
    print("This is {}.".format(data2['Name'][fit_idx]))

    return data2['Name'][fit_idx]


def display_result(path, name):
    ''' DISPLAY_RESULT

        Plot name on the given image.

        Input arguments:

        - path : the path of given image
        - name : the name of the person in image

    '''

    # Read the image
    img = cv2.imread(path)

    # Plot the name on image
    pos = (10, img.shape[0] - 30)
    font = cv2.FONT_HERSHEY_DUPLEX
    color = (100, 100, 100)
    cv2.putText(img, name, pos, font, 1, color)

    cv2.imshow(name, img)
    cv2.waitKey()

    return


def who_is_this(path):
    ''' WHO_IS_THIS

        Recognize the person in the given image.

        Input argument:

        - path : the path of given image

    '''

    # Extract SIFT descriptors form image
    desc = extract_SIFT(path)

    # Obtain the name of the person in image
    name = match_SIFT(desc)

    # Display name in image
    display_result(path, name)

    return


if __name__ == '__main__':
    # Creat traing model of 50 known images
    # create_train_set()

    # Test a new image to find out the person in the image
    file_path = 'Data/president_test/5.jpg'
    who_is_this(file_path)
