import cv2
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
import pandas as pd


def pickle_caltech101_images(data_path, pickle_path):
    print("starting pickling")

    # region variables and constants
    top_folder = os.listdir(data_path)

    picnum = 0
    N = 50
    SIZE = 250  # new size
    data = []  # should be SIZE x SIZE x picnum
    labels = []
    dataDict = {}  # should be 101 X SIZE x SIZE x picnum
    i = 0
    # endregion

    for dir in top_folder:
        dir_path = os.path.join(data_path, dir)
        pic_list = os.listdir(dir_path)
        for pic in pic_list:
            image = cv2.imread(os.path.join(dir_path, pic))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            dsize = (SIZE, SIZE)
            image_resized = cv2.resize(gray, dsize)
            data.append(image_resized)
            labels.append(dir)
            picnum += 1
            # if finished iterating through class images, append images to data dictionary
            if picnum == min(len(pic_list), N):
                data_array = np.array(data)
                labels_array = np.array(labels)
                dataDict[f"{i}"] = (data_array, labels_array)
                i += 1
                picnum = 0
                data = []
                labels = []
                break

    # for vals in dataDict.values():
    #     print(f"{vals[0].shape},\t{vals[1].shape}")

    file = open(pickle_path, 'wb')
    pickle.dump(dataDict, file, protocol=2)
    file.close()
    print("finished pickling")


def load_data(pickle_path):
    file = open(pickle_path, 'rb')
    dataDict = pickle.load(file)
    file.close()

    return dataDict


def create_sets(data_dict):
    trainSetDict = {}
    testSetDict = {}
    split_dict = {}
    i = 0
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    images = []
    labelsss = []
    sizesss = []
    for picClass in data_dict.values():
        # picClass is a touple (data, labels)
        data = picClass[0]
        labels = picClass[1]
        N = data.shape[0]

        # print(f"Class pictures array size: {data.shape},\tclass label array size: {labels.shape}")
        tst_size = int(0.5 * N)  # this always outputs half of N, rounded down TODO: to put it outside the loop. it's override the N again and again.
        print(f"Total class pictures: {N}\n\tTraining: {N - tst_size}\n\tTest: {tst_size}\n")
        images.append(data)
        labelsss.append(labels)
        # split_dict = {"Train": {'Data': X_train, 'Labels': y_train}, "Test": {'Data': X_test, 'Labels': y_test}}
        i += 1
        # for pic in X_train:
        #     cv2.imshow("pic", pic)
        #     cv2.waitKey(0)
    X_train, X_test, y_train, y_test = train_test_split(images, labelsss, test_size=0.2, shuffle=False)
    split_dict = {"Train": {'Data': X_train, 'Labels': y_train}, "Test": {'Data': X_test, 'Labels': y_test}}

    return split_dict


def GetDefaultParameters():  # too add more parameters
    class_indices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    image_size = (150, 150)
    split = 0.2
    clusters = 40
    svm_c = 200
    degree = 3
    kernel = 'rbf'
    gamma = 5
    step_size = 6
    bins = clusters
    validate = False
    parameters = {"class_indices": class_indices, "validate": validate, "image_size": image_size, "Split": split,
                  "clusters": clusters, "step_size": step_size, "bins": bins, "svm_c": svm_c, "kernel": kernel,
                  "gamma": gamma, 'degree': degree}
    return parameters


def train_kmeans(data, params):
    sift_vec = []  # define a list of sift
    for img in data:
        sift = cv2.xfeatures2d.SIFT_create()  # Creating Sifts
        step_size = params['step_size']  # use the step size as defined in params
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in
              range(0, img.shape[1], step_size)]  # compute key points
        points, sifts = sift.compute(img, kp)  # computing the sifsts from the keypoints.
        sift_vec.append(sifts)  # sift_vec: array of all the sifsts.

    # transfer the list to np.array
    all_sifts_array = list(sift_vec[0])
    for value in sift_vec[1:]:
        all_sifts_array = np.append(all_sifts_array, value, axis=0)
    # compute and return k_means
    model = MiniBatchKMeans(n_clusters=params["clusters"], random_state=42,
                            batch_size=params['clusters'] * 4)  # Kmenas model parameters - TODO: need to check in hyperparameters tuning
    kmeans = model.fit(all_sifts_array)  # Fitting the moddel on SIFT
    print('Kmeans trained')

    return kmeans


def prepare(kmeans, data, params):
    histograms_vector = []  # defining a vector
    for img in data:
        sift = cv2.xfeatures2d.SIFT_create()  # creating sifts
        step_size = params['step_size']  # Taking the step size from the params
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in
              range(0, img.shape[1], step_size)]  # computing keypoints
        points, sifts = sift.compute(img, kp)  # computing sifts from key points
        img_predicts = kmeans.predict(sifts)  # computing k-means predictions for the computed sifts
        img_hist, bin_size = np.histogram(img_predicts, bins=params['bins'])  # histograms for each sift bins parameter.
        normalized_hist = img_hist / sum(img_hist)
        histograms_vector.append(normalized_hist)  # add the histogram to histograms vector

    return histograms_vector



if __name__ == '__main__':
    data_path = r"C:\Users\Alina\OneDrive\Desktop\Studies\Learning, representation, and Computer Vision\Homework\Task 1\101_ObjectCategories"
    pickle_file_path = r'C:\Users\Alina\PycharmProjects\CVCourseT1\data.pkl'

    # pickle_caltech101_images(data_path, pickle_file_path)

    dataDict = load_data(pickle_file_path)

    splitdict = create_sets(dataDict)

    params = GetDefaultParameters()

    kmeans_model = train_kmeans(splitdict['Train']['Data'], params)

    TrainDataRep = prepare(kmeans_model, splitdict['Train']['Data'], params)


