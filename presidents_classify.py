import cv2
import numpy as np
import pandas as pd


def extract_SIFT(path):
    img = cv2.imread(path, 0)

    sift = cv2.xfeatures2d.SIFT_create()
    _, desc = sift.detectAndCompute(img, None)

    return desc


def create_train_set():
    folder_path = 'Data/president/'
    data = pd.read_csv('Data/president.csv')
    files = data['Image']
    index_code = data['Index']

    all_desc = np.array([]).reshape(0, 128)
    all_labs = np.array([]).reshape(0, 1)

    for i in range(len(index_code)):
        file_path = folder_path + files[i]

        desc = extract_SIFT(file_path)
        all_desc = np.concatenate((all_desc, desc), axis=0)

        all_labs = np.append(all_labs, np.ones(
            [desc.shape[0], 1]) * index_code[i])

    all_desc = np.float32(all_desc)
    all_labs = np.int32(all_labs)

    np.savez('Data/train.npz', desc=all_desc, labs=all_labs)

    return


def match_SIFT(desc):
    data1 = np.load('Data/train.npz')
    all_desc = data1['desc']
    all_labs = data1['labs']

    data2 = pd.read_csv('Data/index_name.csv')

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc, all_desc, k=2)

    good = []
    for m, n in matches:
        if m.distance <= 0.7 * n.distance:
            good.append(all_labs[m.trainIdx])

    good_count = np.zeros(5)
    for i in range(5):
        good_count[i] = good.count(i + 1)

    fit_idx = np.argmax(good_count)
    print("This is {}.".format(data2['Name'][fit_idx]))

    return data2['Name'][fit_idx]


def display_result(path, name):
    img = cv2.imread(path)

    pos = (10, img.shape[0] - 30)
    font = cv2.FONT_HERSHEY_DUPLEX
    color = (50, 50, 50)
    cv2.putText(img, name, pos, font, 1, color)

    cv2.imshow(name, img)
    cv2.waitKey()

    return


def who_is_this(path):
    desc = extract_SIFT(path)
    name = match_SIFT(desc)

    display_result(path, name)

    return


if __name__ == '__main__':
    # create_train_set()
    file_path = 'Data/president_test/11.jpg'
    who_is_this(file_path)
