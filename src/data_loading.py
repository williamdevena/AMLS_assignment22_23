"""
This file contains all the functions used to load data

"""


import numpy as np
import pandas as pd
import os
import cv2
import logging

from costants import (
    PATH_CARTOON_TRAIN_IMG,
    PATH_CARTOON_TRAIN_LABELS,
    SEPARATOR,
)

logging.basicConfig(format="%(message)s", level=logging.INFO)


def load_flatten_images_from_folder(ds_path):
    """
    Reads the images from a folder and collects them in a flatten array form.

    Args:
        - ds_path (str): Path of the dataset folser

    Returns:
        - np.array(array_flatten_images) (np.ndarray): Numpy array that contains
        all the images in a flatten array form
    """
    logging.info(f"Collecting images from the folder {ds_path}")
    # the list of images is sorted beacuse in the labels file the dataframe is sorted by the number in the name
    # EXAMPLE OF IMAGE NAME: 987.png
    sorting_lambda = lambda image_name: int(image_name.split(".")[0])
    images_list = sorted(os.listdir(ds_path), key=sorting_lambda)
    array_flatten_images = []
    for image_name in images_list:
        image_absolute_path = os.path.join(ds_path, image_name)
        img_array_form = cv2.imread(image_absolute_path)
        img_flatten_array_form = img_array_form.flatten()
        array_flatten_images.append(img_flatten_array_form)

    return np.array(array_flatten_images)


def load_ds_labels_from_csv(csv_path, separator=SEPARATOR):
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
    cartoon_X_train = load_flatten_images_from_folder(PATH_CARTOON_TRAIN_IMG)
    print(cartoon_X_train.shape)
    load_ds_labels_from_csv(PATH_CARTOON_TRAIN_LABELS)


if __name__ == "__main__":
    main()
