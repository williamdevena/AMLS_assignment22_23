"""
This file contains all the functions used to load data

"""


import numpy as np
import pandas as pd
import os
import cv2
import logging
from pprint import pprint
from sklearn.preprocessing import StandardScaler

from src import costants, image_manipulation

logging.basicConfig(format="%(message)s", level=logging.INFO)


def load_X_Y_train_test(dataset_object, scaling=True):
    """
    Reads the ds folders and the .csv labels files and returns
    X_train, Y_train, X_test and Y_test (X_train and X_test in a flatten format).

    Args:
        - dataset_object (Dataset): dataset object
        - scaling (bool): indicates whether to scale or not the datasets (normally has to be True)

    Returns:
        - X_train (np.ndarray): Contains the training samples in a flatten shape
        - Y_train (np.ndarray): Contains the training labels
        - X_test (np.ndarray): Contains the testing samples in a flatten shape
        - Y_test (np.ndarray): Contains the testing labels
    """

    path_train_img, path_train_labels, path_test_img, path_test_labels, use_dominant_color_dataset = retrieve_img_and_labels_paths(
        dataset_object=dataset_object)

    if not use_dominant_color_dataset:
        # load flatten train ds
        X_train = load_flatten_images_from_folder(
            path_train_img, dataset_object.image_dimensions)
        Y_train = load_ds_labels_from_csv(path_train_labels)
        Y_train = Y_train[dataset_object.label]

        # load flatten test ds
        X_test = load_flatten_images_from_folder(
            path_test_img, dataset_object.image_dimensions)
        Y_test = load_ds_labels_from_csv(path_test_labels)
        Y_test = Y_test[dataset_object.label]
    else:
        X_train, Y_train = image_manipulation.create_dominant_colors_dataset(
            costants.PATH_CARTOON_TRAIN_CROPPED_EYE_IMG,
            costants.PATH_CARTOON_TRAIN_LABELS
        )
        X_test, Y_test = image_manipulation.create_dominant_colors_dataset(
            costants.PATH_CARTOON_TEST_CROPPED_EYE_IMG,
            costants.PATH_CARTOON_TEST_LABELS
        )

    if scaling:
        X_train, X_test = scale_train_test_datasets(
            X_train=X_train, X_test=X_test)

    logging.info(
        f"- Loaded {dataset_object.name.upper()} dataset " +
        f"with {dataset_object.label.upper()} label with images of dimensions {dataset_object.image_dimensions}\n\n")

    return X_train, Y_train, X_test, Y_test


def retrieve_img_and_labels_paths(dataset_object):
    """
    Returns the specific paths of the images folder and of the label file
    of the training and testing dataset 

    Args:
        - dataset_object (Dataset): dataset object

    Returns:
        - path_train_img (str): path of the images folder for the training dataset
        - path_train_labels (str): path of the labels file for the training dataset
        - path_test_img (str): path of the images folder for the testing dataset
        - path_test_labels (str): path of the labels file for the testing dataset
        - use_dominant_color_dataset (bool): indicates whether we are using the 'dominan_color'
        version of the dataset 'cartoon'
    """

    use_dominant_color_dataset = False

    if dataset_object.name == "cartoon":
        path_train_img = costants.PATH_CARTOON_TRAIN_IMG
        path_train_labels = costants.PATH_CARTOON_TRAIN_LABELS
        path_test_img = costants.PATH_CARTOON_TEST_IMG
        path_test_labels = costants.PATH_CARTOON_TEST_LABELS
    elif dataset_object.name == "cropped_eye_cartoon":
        path_train_img = costants.PATH_CARTOON_TRAIN_CROPPED_EYE_IMG
        path_train_labels = costants.PATH_CARTOON_TRAIN_LABELS
        path_test_img = costants.PATH_CARTOON_TEST_CROPPED_EYE_IMG
        path_test_labels = costants.PATH_CARTOON_TEST_LABELS
    elif dataset_object.name == "dyn_cropped_eye_cartoon":
        path_train_img = costants.PATH_CARTOON_TRAIN_DYN_CROPPED_EYE_IMG
        path_train_labels = costants.PATH_CARTOON_TRAIN_LABELS
        path_test_img = costants.PATH_CARTOON_TEST_DYN_CROPPED_EYE_IMG
        path_test_labels = costants.PATH_CARTOON_TEST_LABELS
    elif dataset_object.name == "celeba":
        path_train_img = costants.PATH_CELEBA_TRAIN_IMG
        path_train_labels = costants.PATH_CELEBA_TRAIN_LABELS
        path_test_img = costants.PATH_CELEBA_TEST_IMG
        path_test_labels = costants.PATH_CELEBA_TEST_LABELS
    elif dataset_object.name == "cropped_mouth_celeba":
        path_train_img = costants.PATH_CELEBA_TRAIN_CROPPED_MOUTH_IMG
        path_train_labels = costants.PATH_CELEBA_TRAIN_LABELS
        path_test_img = costants.PATH_CELEBA_TEST_CROPPED_MOUTH_IMG
        path_test_labels = costants.PATH_CELEBA_TEST_LABELS
    elif dataset_object.name == "cropped_eyes_celeba":
        path_train_img = costants.PATH_CELEBA_TRAIN_CROPPED_EYES_IMG
        path_train_labels = costants.PATH_CELEBA_TRAIN_LABELS
        path_test_img = costants.PATH_CELEBA_TEST_CROPPED_EYES_IMG
        path_test_labels = costants.PATH_CELEBA_TEST_LABELS
    elif dataset_object.name == "dyn_cropped_eyes_celeba":
        path_train_img = costants.PATH_CELEBA_TRAIN_DYN_CROPPED_EYES_IMG
        path_train_labels = costants.PATH_CELEBA_TRAIN_LABELS
        path_test_img = costants.PATH_CELEBA_TEST_DYN_CROPPED_EYES_IMG
        path_test_labels = costants.PATH_CELEBA_TEST_LABELS
    elif dataset_object.name == "dyn_cropped_mouth_celeba":
        path_train_img = costants.PATH_CELEBA_TRAIN_DYN_CROPPED_MOUTH_IMG
        path_train_labels = costants.PATH_CELEBA_TRAIN_LABELS
        path_test_img = costants.PATH_CELEBA_TEST_DYN_CROPPED_MOUTH_IMG
        path_test_labels = costants.PATH_CELEBA_TEST_LABELS
    elif dataset_object.name == "dominant_color":
        path_train_img = None
        path_train_labels = None
        path_test_img = None
        path_test_labels = None
        use_dominant_color_dataset = True
    else:
        raise Exception(
            f'The dataset parameter of the function load_X_Y_train_test has a non possible value ({dataset_object.name}).'
        )

    return path_train_img, path_train_labels, path_test_img, path_test_labels, use_dominant_color_dataset


def load_images_from_folder(ds_path, image_dimensions):
    """
    Reads the images from a folder and collects them in a multidimensional array form.

    Args:
        - ds_path (str): Path of the dataset folser
        - image_dimensions (tuple): represents the shape that we want for the 
        image of the dataset (in the case it's smaller then the original, 
        resizing is perfomed)

    Returns:
        - array_images (np.ndarray): Numpy array that contains
        all the images in an array form
    """
    logging.info(f"- Collecting NON-FLAT images from the folder {ds_path}")
    # the list of images is sorted beacuse in the labels file the dataframe is sorted by the number in the name
    # EXAMPLE OF IMAGE NAME: 987.png
    def sorting_lambda(image_name): return int(image_name.split(".")[0])
    images_list = sorted(os.listdir(ds_path), key=sorting_lambda)
    array_images = []
    for image_name in images_list:
        # print(image_name)
        image_absolute_path = os.path.join(ds_path, image_name)
        img_array_form = cv2.imread(image_absolute_path)
        # resize image
        resized_img_array_form = cv2.resize(img_array_form, image_dimensions)
        #img_flatten_array_form = resized_img_array_form.flatten()
        # print(resized_img_array_form.shape)
        array_images.append(resized_img_array_form)

    array_images = np.array(array_images)
    return array_images


def load_flatten_images_from_folder(ds_path, image_dimensions):
    """
    Reads the images from a folder and collects them in a flatten array form.

    Args:
        - ds_path (str): Path of the dataset folser
        - image_dimensions (tuple): represents the shape that we want for the 
        image of the dataset (in the case it's smaller then the original, 
        resizing is perfomed)

    Returns:
        - array_flatten_images (np.ndarray): Numpy array that contains
        all the images in a flatten array form
    """
    logging.info(f"- Collecting FLAT images from the folder {ds_path}")
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

    array_flatten_images = np.array(array_flatten_images)

    return array_flatten_images


def load_entire_ds_from_csv(csv_path, separator=costants.SEPARATOR):
    """
    Reads a csv containg a dataset (both X and Y) where the last two
    column are the labels

    Args:
        - csv_path (str): path of the csv file
        - separator (str, optional): separator fro the function pd.read_csv.
        Defaults to costants.SEPARATOR.

    Returns:
        - dataset_dict (Dict): contains the dataset (both X and Y)
    """
    logging.info(f"- Loading dataset from {csv_path}")
    dataset_dict = {}
    dataset_df = pd.read_csv(csv_path, sep=separator)
    dataset = dataset_df.to_numpy()
    dataset_dict['X'] = dataset[:, 1:-2]
    dataset_dict['Y'] = {}

    for label in dataset_df.columns[-2:]:
        dataset_dict['Y'][label] = dataset_df[label].to_numpy()

    return dataset_dict


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


def scale_train_test_datasets(X_train, X_test):
    """
    Scales and transforms the training and testing datasets with the following procedure:

    z = (x - u) / s

    where u is the mean of the data distribution and s is the standard deviation.

    Args:
        - X_train (np.ndarray): Contains the training samples in a flatten shape
        - X_test (np.ndarray): Contains the testing samples in a flatten shape

    Returns:
        - X_train_scaled (np.ndarray): Scaled version of X_train
        - X_test_scaled (np.ndarray): Scaled version of X_test
    """
    scaler = StandardScaler()
    # fitting the scaler on the training set
    # and scaling train and test sets with the same
    # mean and variance (without refitting the scaler
    # on the test set)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def scale_dataset(X):
    """
    Scales and transforms a dataset with the following procedure:

    z = (x - u) / s

    where u is the mean of the data distribution and s is the standard deviation.

    Args:
        - X (np.ndarray): Contains the samples in a flatten shape

    Returns:
        - X_scaled (np.ndarray): Scaled version of X
    """
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    return X_scaled


def main():
    #cartoon_X_train = load_flatten_images_from_folder(PATH_CARTOON_TRAIN_IMG)
    # pprint(cartoon_X_train)
    # pprint(load_ds_labels_from_csv(PATH_CELEBA_TRAIN_LABELS))
    pass


if __name__ == "__main__":
    main()
