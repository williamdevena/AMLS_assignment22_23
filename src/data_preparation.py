"""
This file contains all the functions used to prepare/clean data

"""

import utilities.logging_utilities as logging_utilities
import logging
import os
import sys
from pprint import pprint, pformat

import cv2
import numpy as np
import pandas as pd
from colorthief import ColorThief

from src.costants import (
    PATH_CARTOON_TEST_IMG,
    PATH_CARTOON_TEST_LABELS,
    PATH_CARTOON_TRAIN_IMG,
    PATH_CARTOON_TRAIN_LABELS,
    PATH_CELEBA_TEST_IMG,
    PATH_CELEBA_TEST_LABELS,
    PATH_CELEBA_TRAIN_IMG,
    PATH_CELEBA_TRAIN_LABELS,
    SEPARATOR,
)

sys.path.append("../")

logging.basicConfig(format="%(message)s", level=logging.INFO)


def data_preparation():
    """
    Performs data stage of data preparation on both the cartoon dataset and the celeba dataset.
    (Check the functions check_values_from_csv and check_shape_images for more details)

    Args: None

    Returns:
        - None
    """
    logging_utilities.print_name_stage_project("DATA PREPARATION")
    check_values_from_csv(PATH_CARTOON_TRAIN_LABELS)
    check_values_from_csv(PATH_CELEBA_TRAIN_LABELS)
    check_values_from_csv(PATH_CARTOON_TEST_LABELS)
    check_values_from_csv(PATH_CELEBA_TEST_LABELS)
    check_shape_images(PATH_CELEBA_TRAIN_IMG)
    check_shape_images(PATH_CELEBA_TEST_IMG)
    check_shape_images(PATH_CARTOON_TRAIN_IMG)
    check_shape_images(PATH_CARTOON_TEST_IMG)


def check_values_from_csv(path_csv):
    """
    Reads a csv and prints for every variable the possible values and their counts
    (useful to check if there are NaN values, empty values or other types of values we might want to check)

    Args:
        - path_csv (str): path of the csv file

    Returns:
        - df.dtypes (pandas.core.series.Series): contains the types of all the variables
        - dict_counts (dict): dictionary of the form {variable : ([all values in the distribution], [counts of the values in the distribution])}
    """
    logging.info(f"READING FILE {path_csv}")
    df = pd.read_csv(path_csv, sep=SEPARATOR)
    df_only_variables = df.iloc[:, 2:]
    dict_counts = {}
    for key in df_only_variables.keys():
        counts = np.unique(df_only_variables[key], return_counts=True)
        dict_counts[key] = {
            "values": list(counts[0]),
            "counts": list(counts[1]),
        }
    logging.info(f"\nTYPES OF DATA IN THE DATAFRAME:\n{pformat(df.dtypes)}")
    logging.info(
        f"\nCOUNTS OF VALUES IN THE VARIABLES:\n{pformat(dict_counts)}"
    )
    logging.info("\n\n\n")

    return df.dtypes, dict_counts


def check_shape_images(path_folder_images):
    """
    Reads a folder of images and returns every type of shape present
    (useful to check what shape the image have and to check if they have all the same shape)

    Args:
        - path_folder_images (str): path of the csv file

    Returns:
         - returns a tuple of the form ([shapes present in the folder], [counts of the shapes in the folder])
    """
    logging.info(f"READING THE DATASET IN THE FOLDER {path_folder_images}")
    images = os.listdir(path_folder_images)
    array_shapes = [
        cv2.imread(os.path.join(path_folder_images, image)).shape
        for image in images
    ]
    shapes_np_unique = np.unique(array_shapes, axis=0, return_counts=True)

    shapes = {
        "shapes": [tuple(shape) for shape in shapes_np_unique[0]],
        "counts": [count for count in shapes_np_unique[1]],
    }

    logging.info(f"\nSHAPES IN THE DATASET:\n{shapes}")
    logging.info("\n\n\n")

    return shapes


def reformat_csv_cartoon(original_path, new_path):
    """
    Function used to reformat the csv of the cartoon dataset to match it with the celeba datase:
    The file_name column becomes img_name and it is the first column ( not the last as originally)

    Args:
        - original_path (str): path of the original csv file
        - old_path (str): path of the new csv file

    Returns:
         - None
    """
    df = pd.read_csv(original_path, sep=SEPARATOR)
    df = df.rename(columns={"file_name": "img_name"})
    columns_titles = ["img_name", "eye_color", "face_shape"]
    df = df.reindex(columns=columns_titles)
    df.to_csv(new_path, sep=SEPARATOR)
    print(df)


def count_dark_glasses(path_folder_images):
    raise NotImplementedError


# def count_pupil_colors(path_folder_images, pixel_y, pixel_x):
#     """
#     Tries to count the number of colors in the pupil to find out
#     if the task of predicting the feature eye_color is possible
#     to solve with a hard coded solution.

#     Args:
#         - path_folder_images (str): path of the folder that contains 
#         the images of the dataset
#         - pixel_y (int): y of the pixel we want to select
#         - pixel_x (int): x of the pixel we want to select

#     Returns: None

#     """
#     if not os.path.isdir(path_folder_images):
#         raise FileNotFoundError(
#             f"The directory {path_folder_images} does not exist.")

#     logging.info(f"READING THE IMAGES IN THE FOLDER {path_folder_images}")
#     images = os.listdir(path_folder_images)

#     dict_count_pupil_colors = {}
#     dict_images_pupil_colors = {}
#     for file in os.listdir(path_folder_images):
#         if file.endswith(".png"):
#             img = cv2.imread(os.path.join(path_folder_images, file))
#             pupil_color = img[pixel_y, pixel_x]
#             tuple_pupil_color = tuple(pupil_color)
#             # in this case get() returns dict_pupil_colors[pupil_color] if pupil_color
#             # is in dict_pupil_colors 0 otherwise
#             dict_count_pupil_colors[tuple_pupil_color] = dict_count_pupil_colors.get(
#                 tuple_pupil_color, 0) + 1
#             # dict_images_pupil_colors[tuple_pupil_color] =
#             dict_images_pupil_colors.setdefault(
#                 tuple_pupil_color, []).append(file)

#     # logging.info(
#     #     dict(sorted(dict_count_pupil_colors.items(), key=lambda item: item[1])))
#     # pprint(sorted(dict_images_pupil_colors.items(),
#     #        key=lambda item: dict_count_pupil_colors[item[0]]))

#     return (
#         dict(sorted(dict_count_pupil_colors.items(),
#              key=lambda item: item[1], reverse=True)),
#         dict(sorted(dict_images_pupil_colors.items(),
#                     key=lambda item: dict_count_pupil_colors[item[0]], reverse=True))
#     )


def count_dominant_pupil_colors(path_folder_images):
    """
    Returns the dominant colors of the pupils in the dataset.

    Args:
        - path_folder_images (str): path of the folder that contains 
        the images of the dataset

    Returns:
        - dict_dominant_colors (dict): a dictionary of this form {dominant_color: count}

    """
    if not os.path.isdir(path_folder_images):
        raise FileNotFoundError(
            f"The directory {path_folder_images} does not exist.")

    logging.info(f"READING THE IMAGES IN THE FOLDER {path_folder_images}")
    images = os.listdir(path_folder_images)

    dict_dominant_colors = {}
    i = 0
    for file in os.listdir(path_folder_images):
        if file.endswith(".png"):
            print(f"Progress: {i} \ 10000", end='\r')
            i += 1
            # if i > 200:
            #     break
            color_thief = ColorThief(os.path.join(path_folder_images, file))
            # get the dominant color
            dominant_color = color_thief.get_color(quality=10)
            # in this case get() returns dict_pupil_colors[pupil_color] if pupil_color
            # is in dict_pupil_colors 0 otherwise
            dict_dominant_colors[dominant_color] = dict_dominant_colors.get(
                dominant_color, 0) + 1
    dict_dominant_colors = dict(sorted(dict_dominant_colors.items(),
                       key=lambda item: item[1], reverse=True))
    
    return dict_dominant_colors


def main():
    # check_values_from_csv(PATH_CARTOON_TRAIN_LABELS)

    return


if __name__ == "__main__":
    main()
