import cv2
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from tqdm import tqdm
import itertools
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from more_itertools import unique_everseen
import seaborn as sns
import matplotlib.pyplot as plt
import inspect

def GetDefaultParameters():  # to add more parameters
    dir_path = r"C:\Users\Alina\OneDrive\Desktop\Studies\Learning, representation, and Computer Vision\Homework\Task 1"
    dir_path = r"C:\Users\razdo\Documents\_Dor\Second Degree documents\Courses\Semester 1\Learning, representation, and Computer Vision\Homework\Task 1 misc"

    data_path = os.path.join(dir_path, "101_ObjectCategories")
    data_pickle_file_path = os.path.join(dir_path, "data.pkl")
    kmeans_pickle_file_path = os.path.join(dir_path, "kmeans.pkl")
    SVM_pickle_file_path = os.path.join(dir_path, "SVM.pkl")
    Results_pickle_file_path = os.path.join(dir_path, "Results.pkl")

    Use_cache = True
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

    parameters = {"Data": {'CalTechData': data_path, 'data_pickle_file_path': data_pickle_file_path, "image_size": image_size, "Use_cache": Use_cache},
                  "Kmeans": {"step_size": step_size, "clusters": clusters, "kmeans_pickle_file_path": kmeans_pickle_file_path},
                  "Prepare": {"step_size": step_size, "bins": bins},
                  "Train": {"svm_c": svm_c, "SVM_pickle_file_path": SVM_pickle_file_path, "Use_cache": Use_cache, "kernel": kernel, "gamma": gamma},
                  "Split": {'split_ratio': split_ratio, 'class_indices': class_indices},
                  "Test" : {"SVM_pickle_file_path": SVM_pickle_file_path},
                  "Results" : {"Results_pickle_file_path": Results_pickle_file_path},
                  "validate": validate,
                  'degree': degree}

    return parameters


def pickle_caltech101_images(dataParams):
    """ Saves images and labels in pickle file for quick loading

        loads the images
        iterates over the requested folders and the images in each folder
        saves them to a dictionary
        pickles the dictionary

        """

    if dataParams['Use_cache'] == False:
        DandL = GetData(dataParams['data_pickle_file_path'])
        return DandL
    else:
        DandL = GetData(dataParams['data_pickle_file_path'])
        return DandL

    print("starting pickling Caltech data")

    # region variables and constants
    data_path = dataParams['CalTechData']
    pickle_path = dataParams['data_pickle_file_path']
    dsize = dataParams['image_size'] # new size

    top_folder = os.listdir(data_path)

    picnum = 0
    N = 50
    data = []  # should be SIZE x SIZE x picnum
    labels = []
    dataDict = {"Data": {}, "Labels": {}}  # should be 101 X SIZE x SIZE x picnum
    i = 0
    # endregion

    for dir in tqdm(top_folder, desc="Caltech 101 classes"):
        dir_path = os.path.join(data_path, dir)
        pic_list = os.listdir(dir_path)
        for pic in pic_list:
            image = cv2.imread(os.path.join(dir_path, pic))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
    print("finished pickling Caltech data")
    return dataDict


def GetData(pickle_file_path):

    file = open(pickle_file_path, 'rb')
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

    X_train_all = []
    X_test_all = []
    y_train_all = []
    y_test_all = []

    for classNum in dataDict.keys(): #iterates over all classes [1,101]

        if classNum in class_indices:
            data = dataDict[classNum] # array of pictures with dimensions [N, SIZE, SIZE]
            labels = labelDict[classNum] # array with dimensions [N, 1]

            N = data.shape[0]  # depends on number of pictures in class, number between [32, 50]
            tst_size = int(0.5 * N)  # this always outputs half of N, rounded down
            # print(f"Class pictures array size: {data.shape},\tclass label array size: {labels.shape}")
            # print(f"Total class pictures: {N}\n\tTraining: {N - tst_size}\n\tTest: {tst_size}\n")

            images_list = []
            labels_list = []

            for i in range(len(data)):
                images_list.append(data[i])
                labels_list.append(labels[i])

            X_train, X_test, y_train, y_test = train_test_split(images_list, labels_list, test_size=tst_size, shuffle=False)

            X_train_all.append(X_train)
            X_test_all.append(X_test)
            y_train_all.append(y_train)
            y_test_all.append(y_test)

    X_train_all_flat = list(itertools.chain(*X_train_all))
    X_test_all_flat = list(itertools.chain(*X_test_all))
    y_train_all_flat = list(itertools.chain(*y_train_all))
    y_test_all_flat = list(itertools.chain(*y_test_all))

    split_dict = {"Train": {'Data': X_train_all_flat, 'Labels': y_train_all_flat}, "Test": {'Data': X_test_all_flat, 'Labels': y_test_all_flat}}

    return split_dict


def TrainKmeans(images, KmeansParams):
    sift_vec = []  # define a list of sift
    sift = cv2.xfeatures2d.SIFT_create()  # Creating Sifts

    for img in images:
        step_size = KmeansParams['step_size']  # use the step size as defined in params
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in
              range(0, img.shape[1], step_size)]  # compute key points
        points, sifts = sift.compute(img, kp)  # computing the sifsts from the keypoints.
        sift_vec.append(sifts)  # sift_vec: array of all the sifsts.

    # sift vec has 385 images x 625 keypoints per image x 128 features per keypoint
    # This next line creates one long list with all sifts in it of all images
    alll_sifts = [keypoint for imgFeatures in sift_vec for keypoint in imgFeatures] # imgFeatures is a 625x128 list
                                                                                    # keypoint is a 1x128 vector
    # compute and return k_means
    model = MiniBatchKMeans(n_clusters=KmeansParams["clusters"], random_state=42,
                            batch_size=KmeansParams['clusters'] * 4)  # Kmenas model parameters - TODO: need to check in hyperparameters tuning

    kmeans = model.fit(alll_sifts)  # Fitting the model on SIFT
    print('Kmeans trained')

    return kmeans


def Prepare(kmeans, data, prepareParams):
    histograms_vector = []  # defining a vector
    sift = cv2.xfeatures2d.SIFT_create()  # creating sifts

    for img in data:
        step_size = prepareParams['step_size']  # Taking the step size from the params
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size)
              for x in range(0, img.shape[1], step_size)]  # computing keypoints
        points, sifts = sift.compute(img, kp)  # computing sifts from key points

        sifts = np.float64(sifts)

        img_predicts = kmeans.predict(sifts)  # computing k-means predictions for the computed sifts
        img_hist, bin_size = np.histogram(img_predicts, bins=prepareParams['bins'])  # histograms for each sift bins parameter.
        normalized_hist = img_hist / sum(img_hist)
        histograms_vector.append(normalized_hist)  # add the histogram to histograms vector

    return histograms_vector


def TrainModel(data, labels, trainParams):
    '''
    Train the model with SVM
    :param data: the train data
    :param labels: the train labels
    :param params:dictionary of parameters
    :return: the computed SVM of the trained data
    '''

    data = np.array(data)
    labels = np.array(labels)

    Model = OneVsRestClassifier(SVC(C=trainParams['svm_c'], kernel=trainParams['kernel'], gamma=trainParams['gamma'], probability=True)).fit(data, labels)  # fitting the SVM on the data
    print('SVM Trained')

    file = open(trainParams['SVM_pickle_file_path'], 'wb')
    pickle.dump(Model, file, protocol=2)
    file.close()
    print("finished pickling SVM")

    return Model


def Test(model, test_data):

    predictions = model.predict(test_data) #computing the predictions from the model
    probabilities = model.predict_proba(test_data) # computing the probabilities

    return predictions,probabilities


def evaluate(predicts, real_labels):

    error = 1 - accuracy_score(predicts, real_labels) # Compute error
    accuracy_score_of_all = accuracy_score(predicts, real_labels)

    print(f"\nTotal accuracy of predictions: {accuracy_score_of_all}")

    cnf_mat = confusion_matrix(real_labels, predicts, list(unique_everseen(real_labels)))  # Create confusion matrix
    print(cnf_mat)

    df_cm = pd.DataFrame(cnf_mat, index=list(unique_everseen(real_labels)), columns=list(unique_everseen(real_labels)),)
    col_sum = df_cm.sum(axis=1)
    df_cm = df_cm.div(col_sum,axis = 0)
    plt.figure(figsize=(20,20))
    heatmap = sns.heatmap(df_cm, annot=True)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=8)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=8)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title("Confusion Matrix, Normalized")
    plt.show()
    return error,cnf_mat


def Exp(params):

    print("Getting data and labels")
    DandL = pickle_caltech101_images(Params['Data'])

    print("Splitting train and test sets")
    splitdict = TrainTestSplit(DandL["Data"], DandL["Labels"], Params["Split"])

    print("Training kmeans using training images' Sifts")
    kmeans = TrainKmeans(splitdict['Train']['Data'], Params["Kmeans"])

    print("Creating histograms for training set using kmeans")
    TrainDataRep = Prepare(kmeans, splitdict['Train']['Data'], Params["Prepare"])

    print("Training SVM model")
    SVM_model = TrainModel(TrainDataRep, splitdict['Train']['Labels'], Params["Train"])

    print("Creating histograms for test set using kmeans")
    TestDataRep = Prepare(kmeans, splitdict['Test']['Data'], Params["Prepare"])

    print("Showing results")
    Results, probabilities = Test(SVM_model, TestDataRep)

    Error, conf_mat = evaluate(Results, splitdict['Test']['Labels'])

    file = open(params["Results"]['Results_pickle_file_path'], 'wb')
    pickle.dump([Error, conf_mat], file, protocol=2)  # dictionary is of type { Data: {1: data1, 2: data2, ...},
    file.close()

    print(f"finished pickling {inspect.stack()[0][3]} results.")


if __name__ == '__main__':
    Params = GetDefaultParameters()

    np.random.seed(0)

    # Running experiment 1 with default parameters
    Params["Data"]["image_size"] = (100, 100)

    Params["Kmeans"]['Clusters'] = '50'

    Exp(Params)

    Params["Kmeans"]['Clusters'] = '30'
    Params["Data"]["image_size"] = (300, 300)
    # Params["Train"]['Kernel'] = 'linear'

    Exp(Params)










