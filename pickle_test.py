import cv2
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
import pandas as pd


def GetDefaultParameters():  # to add more parameters
    # data_path = r"C:\Users\Alina\OneDrive\Desktop\Studies\Learning, representation, and Computer Vision\Homework\Task 1\101_ObjectCategories"
    # pickle_file_path = r'C:\Users\Alina\PycharmProjects\CVCourseT1\data.pkl'
    data_path = r"C:\Users\razdo\Documents\_Dor\Second Degree documents\Courses\Semester 1\Learning, representation, and Computer Vision\Homework\Task 1 misc\101_ObjectCategories"
    pickle_file_path = r"C:\Users\razdo\Documents\_Dor\Second Degree documents\Courses\Semester 1\Learning, representation, and Computer Vision\Homework\Task 1 misc\data.pkl"

    class_indices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    image_size = (150, 150)
    split_ratio = 0.2
    clusters = 40
    svm_c = 200
    degree = 3
    kernel = 'rbf'
    gamma = 5
    step_size = 6
    bins = clusters
    validate = False

    parameters = {"Data": {'CalTechData': data_path, 'PickleData': pickle_file_path, "image_size": image_size},
                  "Prepare": {"step_size": step_size, "bins": bins},
                  "Train": {"step_size": step_size, "clusters": clusters},
                  "Split": {'split_ratio': split_ratio, 'class_indices': class_indices},
                  "validate": validate,
                  "svm_c": svm_c,
                  "kernel": kernel,
                  "gamma": gamma,
                  'degree': degree}

    return parameters


def pickle_caltech101_images(dataParams):
    """ Saves images and labels in pickle file for quick loading

        loads the images
        iterates over the requested folders and the images in each folder
        saves them to a dictionary
        pickles the dictionary

        """

    print("starting pickling")

    # region variables and constants
    data_path = dataParams['CalTechData']
    pickle_path = dataParams['PickleData']
    SIZE = dataParams['image_size'] # new size

    top_folder = os.listdir(data_path)

    picnum = 0
    N = 50
    data = []  # should be SIZE x SIZE x picnum
    labels = []
    dataDict = {"Data": {}, "Labels": {}}  # should be 101 X SIZE x SIZE x picnum
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
                dataDict["Data"][i] = data_array
                dataDict["Labels"][i] = labels_array
                i += 1
                picnum = 0
                data = []
                labels = []
                break

    # for vals in dataDict.values():
    #     print(f"{vals[0].shape},\t{vals[1].shape}")

    file = open(pickle_path, 'wb')
    pickle.dump(dataDict, file, protocol=2) # dictionary is of type { Data: {1: data1, 2: data2, ...},
                                                                   #  Labels: {1: labels1, 2: labels2, ...} }
    file.close()
    print("finished pickling")


def GetData(params):
    pickle_path = params['PickleData']

    file = open(pickle_path, 'rb')
    dataDict = pickle.load(file)
    file.close()

    return dataDict


def TrainTestSplit(dataDict, labelDict, splitParams):
    """ Splits the data from the dictionary to train sets and test sets

        each class has between 32 and 50 images_list
        the train set and test set are each half of the total number of images_list
        returns a dictionary of the split data sets

        """
    split_ratio = splitParams['split_ratio']
    class_indices = splitParams['class_indices']

    images_list = []
    labels_list = []

    for classNum in dataDict.keys(): #iterates over all classes [1,101]
        if classNum in class_indices:
            data = dataDict[classNum] # array of pictures with dimensions [N, SIZE, SIZE]
            labels = labelDict[classNum] # array with dimensions [N, 1]
            N = data.shape[0]  # depends on number of pictures in class, number between [32, 50]

            # print(f"Class pictures array size: {data.shape},\tclass label array size: {labels.shape}")
            tst_size = int(0.5 * N)  # this always outputs half of N, rounded down TODO: to put it outside the loop. it's override the N again and again.
            # print(f"Total class pictures: {N}\n\tTraining: {N - tst_size}\n\tTest: {tst_size}\n")

            images_list.append(data)
            labels_list.append(labels)

    X_train, X_test, y_train, y_test = train_test_split(images_list, labels_list, test_size=split_ratio, shuffle=False)
    split_dict = {"Train": {'Data': X_train, 'Labels': y_train}, "Test": {'Data': X_test, 'Labels': y_test}}

    return split_dict


def prepare(kmeans, data, prepareParams):
    histograms_vector = []  # defining a vector
    for img in data:
        sift = cv2.xfeatures2d.SIFT_create()  # creating sifts
        step_size = prepareParams['step_size']  # Taking the step size from the params
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in
              range(0, img.shape[1], step_size)]  # computing keypoints
        points, sifts = sift.compute(img, kp)  # computing sifts from key points
        img_predicts = kmeans.predict(sifts)  # computing k-means predictions for the computed sifts
        img_hist, bin_size = np.histogram(img_predicts, bins=prepareParams['bins'])  # histograms for each sift bins parameter.
        normalized_hist = img_hist / sum(img_hist)
        histograms_vector.append(normalized_hist)  # add the histogram to histograms vector

    return histograms_vector


def train_kmeans(data, trainParams):
    print('Begin Kmeans training')

    sift_vec = []  # define a list of sift
    for img in data:
        sift = cv2.xfeatures2d.SIFT_create()  # Creating Sifts
        step_size = trainParams['step_size']  # use the step size as defined in params
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in
              range(0, img.shape[1], step_size)]  # compute key points
        points, sifts = sift.compute(img, kp)  # computing the sifts from the keypoints.
        sift_vec.append(sifts)  # sift_vec: array of all the sifts.

    # transfer the list to np.array
    all_sifts_array = list(sift_vec[0])
    for value in sift_vec[1:]:
        all_sifts_array = np.append(all_sifts_array, value, axis=0)
    # compute and return k_means
    model = MiniBatchKMeans(n_clusters=trainParams["clusters"], random_state=42,
                            batch_size=trainParams['clusters'] * 4)  # Kmenas model parameters - TODO: need to check in hyperparameters tuning
    kmeans = model.fit(all_sifts_array)  # Fitting the moddel on SIFT
    print('Kmeans trained')

    return kmeans


if __name__ == '__main__':

    Params = GetDefaultParameters()

    np.random.seed(0)

    # pickle_caltech101_images(Params['Data'])

    DandL = GetData(Params["Data"])

    SplitData = TrainTestSplit(DandL["Data"], DandL["Labels"], Params["Split"])

    kmeans_model = train_kmeans(SplitData["Train"]['Data'], Params["Train"])

    TrainDataRep = prepare(kmeans_model, SplitData["Train"]['Data'], Params["Prepare"])




