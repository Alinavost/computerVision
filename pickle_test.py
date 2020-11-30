import cv2
import os
import pickle
import numpy as np
# import sklearn.model_selection as sk
from sklearn.model_selection import train_test_split


def pickle_caltech101_images(data_path, pickle_path):
    print("starting pickling")

    #region variables and constants
    top_folder = os.listdir(data_path)

    picnum = 0
    N = 50
    SIZE = 250  # new size
    data = [] # should be SIZE x SIZE x picnum
    labels = []
    dataDict = {} # should be 101 X SIZE x SIZE x picnum
    i = 0
    #endregion

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
    pickle.dump(dataDict, file, protocol=pickle.HIGHEST_PROTOCOL)
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
    i = 0

    for picClass in data_dict.values():
        # picClass is a touple (data, labels)
        data = picClass[0]
        labels = picClass[1]
        N = data.shape[0]

        # print(f"Class pictures array size: {data.shape},\tclass label array size: {labels.shape}")
        tst_size = int(0.5 * N)  # this always outputs half of N, rounded down
        print(f"Total class pictures: {N}\n\tTraining: {N - tst_size}\n\tTest: {tst_size}\n")

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=tst_size, shuffle=False)
        trainSetDict[f"{i}"] = (X_train, y_train)
        testSetDict[f"{i}"] = (X_test, y_test)
        i += 1
        # for pic in X_train:
        #     cv2.imshow("pic", pic)
        #     cv2.waitKey(0)

    return trainSetDict, testSetDict


if __name__ == '__main__':
    data_path = r"C:\Users\Alina\OneDrive\Desktop\Studies\Learning, representation, and Computer Vision\Homework\Task 1\101_ObjectCategories"
    pickle_file_path = r'C:\Users\Alina\PycharmProjects\CVCourseT1\data.pkl'

    # pickle_caltech101_images(data_path, pickle_file_path)

    dataDict = load_data(pickle_file_path)

    train_set_dict, test_set_dict = create_sets(dataDict)

