"""
This file contains all the functions used to load data

"""


import numpy as np
import pandas as pd
import os
import cv2
import logging
from pprint import pprint
import mlflow
from sklearn.preprocessing import StandardScaler

from src import costants

logging.basicConfig(format="%(message)s", level=logging.INFO)


def load_X_Y_train_test(dataset, label, image_dimensions, scaling=True):
    """
    Reads the ds folders and the .csv labels files and returns
    X_train, Y_train, X_test and Y_test (X_train and X_test in a flatten format).

    Args:
        - dataset (str): could be "cartoon", "cropped_eye_cartoon", "celeba", "cropped_mouth_celeba" or "cropped_eyes_celeba" and indicates which
        dataset we want to load
        - labels (str): label of Y_train and Y_test that we want to load
        (both datasets have two labels)
        - image_dimensions (tuple): represents the shape that we want for the image 
        of the dataset (in the case it's smaller then the original, resizing is perfomed)

    Returns:
        - X_train (np.ndarray): Contains the training samples in a flatten shape
        - Y_train (np.ndarray): Contains the training labels
        - X_test (np.ndarray): Contains the testing samples in a flatten shape
        - Y_test (np.ndarray): Contains the testing labels
    """

    if dataset == "cartoon":
        path_train_img = costants.PATH_CARTOON_TRAIN_IMG
        path_train_labels = costants.PATH_CARTOON_TRAIN_LABELS
        path_test_img = costants.PATH_CARTOON_TEST_IMG
        path_test_labels = costants.PATH_CARTOON_TEST_LABELS
    elif dataset == "cropped_eye_cartoon":
        path_train_img = costants.PATH_CARTOON_TRAIN_CROPPED_EYE_IMG
        path_train_labels = costants.PATH_CARTOON_TRAIN_LABELS
        path_test_img = costants.PATH_CARTOON_TEST_CROPPED_EYE_IMG
        path_test_labels = costants.PATH_CARTOON_TEST_LABELS
    elif dataset == "celeba":
        path_train_img = costants.PATH_CELEBA_TRAIN_IMG
        path_train_labels = costants.PATH_CELEBA_TRAIN_LABELS
        path_test_img = costants.PATH_CELEBA_TEST_IMG
        path_test_labels = costants.PATH_CELEBA_TEST_LABELS
    elif dataset == "cropped_mouth_celeba":
        path_train_img = costants.PATH_CELEBA_TRAIN_CROPPED_MOUTH_IMG
        path_train_labels = costants.PATH_CELEBA_TRAIN_LABELS
        path_test_img = costants.PATH_CELEBA_TEST_CROPPED_MOUTH_IMG
        path_test_labels = costants.PATH_CELEBA_TEST_LABELS
    elif dataset == "cropped_eyes_celeba":
        path_train_img = costants.PATH_CELEBA_TRAIN_CROPPED_EYES_IMG
        path_train_labels = costants.PATH_CELEBA_TRAIN_LABELS
        path_test_img = costants.PATH_CELEBA_TEST_CROPPED_EYES_IMG
        path_test_labels = costants.PATH_CELEBA_TEST_LABELS
    else:
        raise Exception(
            f'The dataset parameter of the function load_X_Y_train_test has a non possible value ({dataset}).\nThe dataset parameter has to be "cartoon", "cropped_eye_cartoon", "celeba", "cropped_mouth_celeba" or "cropped_eyes_celeba".'
        )

    # load flatten train ds
    X_train = load_flatten_images_from_folder(path_train_img, image_dimensions)
    Y_train = load_ds_labels_from_csv(path_train_labels)
    Y_train = Y_train[label]

    # load flatten test ds
    X_test = load_flatten_images_from_folder(path_test_img, image_dimensions)
    Y_test = load_ds_labels_from_csv(path_test_labels)
    Y_test = Y_test[label]

    if scaling:
        scaler = StandardScaler()
        # fitting the scaler on the training set
        # and scaling train and test sets with the same
        # mean and variance (without refitting the scaler
        # on the test set)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, Y_train, X_test, Y_test


def load_flatten_images_from_folder(ds_path, image_dimensions):
    """
    Reads the images from a folder and collects them in a flatten array form.

    Args:
        - ds_path (str): Path of the dataset folser
        - image_dimensions (tuple): represents the shape that we want for the 
        image of the dataset (in the case it's smaller then the original, 
        resizing is perfomed)

    Returns:
        - np.array(array_flatten_images) (np.ndarray): Numpy array that contains
        all the images in a flatten array form
    """
    logging.info(f"Collecting images from the folder {ds_path}")
    # the list of images is sorted beacuse in the labels file the dataframe is sorted by the number in the name
    # EXAMPLE OF IMAGE NAME: 987.png
    def sorting_lambda(image_name): return int(image_name.split(".")[0])
    images_list = sorted(os.listdir(ds_path), key=sorting_lambda)
    array_flatten_images = []
    for image_name in images_list:
        image_absolute_path = os.path.join(ds_path, image_name)
        img_array_form = cv2.imread(image_absolute_path)
        # resize image
        resized_img_array_form = cv2.resize(img_array_form, image_dimensions)
        img_flatten_array_form = resized_img_array_form.flatten()
        array_flatten_images.append(img_flatten_array_form)

    return np.array(array_flatten_images)


def load_ds_labels_from_csv(csv_path, separator=costants.SEPARATOR):
    """
    Reads a csv file that contains labels of a dataset and returns
    a dictionary containing the numpy arrays of the columns as values
    and labels as keys.

    Args:
        - csv_path (str): path of the csv file
        - separator (str): used by the function pd.read_csv

    Returns:
        - labels_dict (dict): Dictionary that has the labels as keys and
        numpy array of the labels as values

    """
    labels_dict = {}
    labels_df = pd.read_csv(csv_path, sep=separator)
    # we eliminate the index column and the img_name column
    # selecting only the labels columns
    labels_df = labels_df.iloc[:, 2:]
    labels_dict = {
        label: labels_df[label].to_numpy() for label in labels_df.keys()
    }

    return labels_dict


def main():
    #cartoon_X_train = load_flatten_images_from_folder(PATH_CARTOON_TRAIN_IMG)
    # pprint(cartoon_X_train)
    # pprint(load_ds_labels_from_csv(PATH_CELEBA_TRAIN_LABELS))
    pass


if __name__ == "__main__":
    main()
