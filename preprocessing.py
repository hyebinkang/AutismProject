import random
import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

data_dir = "AutismDataset/consolidated/"
autism_data = os.listdir(os.path.join(data_dir, "Autistic/"))
non_autism_data = os.listdir(os.path.join(data_dir, "Non_Autistic/"))

train_dir = "./AutismDataset/train/"
train_file = os.listdir(train_dir)

test_dir = "./AutismDataset/test/"
test_file = os.listdir(test_dir)

valid_dir = "AutismDataset/valid/"
valid_autism_data = os.listdir(os.path.join(valid_dir, "Autistic/"))
valid_non_autism_data = os.listdir(os.path.join(valid_dir, "Non_Autistic/"))

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNEL = 3

autism_face_name = []
non_autism_face_name = []

def train_set():
#####train set

    train_non_autistic = []
    train_autistic = []

    for i in train_file:
        if 'Non_Autistic' in i:
            train_non_autistic.append(i)
        else:
            train_autistic.append(i)

    train_imgs = train_autistic + train_non_autistic
    random.shuffle(train_imgs)

    X = []
    y = []

    for image in train_imgs:
        img = cv2.imread(train_dir+image, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        print(img.shape)
        img = img/255.0
        X.append(img)

        if 'Non_Autistic' in image:
            y.append(0)                     #0이 정상

        else:
            y.append(1)                     #1이 자폐


    X_train = np.array(X)
    y_train = np.array(y)

    return X_train, y_train

def test_set():
    ####test data set
    test_imgs = [i for i in test_file]

    X_test= []
    y_test = []

    for image in test_imgs:
        img = cv2.imread(test_dir+image, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img/255.0
        X_test.append(img)

        if 'Non_Autistic' in image:
            y_test.append(0)                                # 0 정상
            non_autism_face_name.append(test_dir + image)
        else:
            y_test.append(1)                                # 1 자폐
            autism_face_name.append(test_dir + image)


    X_test = np.array(X_test)
    y_test = np.array(y_test)



    return X_test, y_test

def valid_set():

    #####valid set
    valid_autism_imgs = ["Autistic/{}".format(i) for i in valid_autism_data]
    valid_non_autism_imgs = ["Non_Autistic/{}".format(i) for i in valid_non_autism_data]

    val_imgs = valid_autism_imgs + valid_non_autism_imgs
    random.shuffle(val_imgs)

    X1 = []
    y1 = []

    for image in val_imgs:
        img = cv2.imread(valid_dir+image, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img/255.0
        X1.append(img)

        if 'Non_Autistic' in image:
            y1.append(0)  # 0이 정상

        else:
            y1.append(1)  # 1이 자폐

    X1_val = np.array(X1)
    y1_val = np.array(y1)

    return X1_val, y1_val

train_set()
test_set()
valid_set()


# print(X_train.shape, "X_train")
# print(y_train.shape, "y_train")
# print(X1_val.shape, "X1val")
# print(y1_val.shape, "y1val")
# print()

