import os
import cv2
import logging
from src import data_loading
from colorthief import ColorThief
import numpy as np


def transform_ds_rgba_to_rgb(path_folder, new_folder):
    logging.info(
        f"\n- Trasforming the images from {path_folder} from RGB-A to RGB and writing them in {new_folder}")
    images = os.listdir(path_folder)

    if not os.path.exists(new_folder):
        os.mkdir(new_folder)

    for image_name in images:
        image = cv2.imread(os.path.join(
            path_folder, image_name), cv2.IMREAD_COLOR)
        cv2.imwrite(filename=os.path.join(new_folder, image_name), img=image)
        # print(image)


def crop_images_dataset(path_original_folder, path_cropped_folder, extension_image, crop_y, crop_x):
    """
    It reads a folder with images and creates a new folder with inside the cropped version 
    of the images.

    Args:
        - path_original_folder (str): path of the original folder that contains the images
        that we want to crop
        - path_cropped_folder (str): path of the folder were we want to write the cropped images
        - extension_image (str): extension of the image (".jpg", "jpeg", ".png", ....)
        - crop_x (slice): represents the cropped area (the pixels we want to keep) on the x axis
        - crop_y (slice): represents the cropped area (the pixels we want to keep) on the y axis

    Returns: None

    """
    logging.info(
        f"\n- Cropping manually the images in {path_original_folder} and writing them in {path_cropped_folder}")
    if not os.path.isdir(path_cropped_folder):
        os.mkdir(path_cropped_folder)
    for file in os.listdir(path_original_folder):
        if file.endswith(extension_image):
            img = cv2.imread(os.path.join(
                path_original_folder, file), cv2.IMREAD_COLOR)
            cropped_image = img[crop_y, crop_x]
            cv2.imwrite(os.path.join(path_cropped_folder, file), cropped_image)


def create_dominant_colors_dataset(path_original_dataset, path_original_labels):
    """
    It reads a dataset and creates a transformed version of it where the transformation 
    consists in assigning every image of the dataset to it's dominant color.
    So it reduces the dataset to a 3 dimension one (the RGB dimensions of the dominant color).
    This purpose of this function is mainly to be applied on the 'cropped eye cartoon' dataset.

    Args:
        - path_original_dataset (str): path of the original dataset to transform (X_train)
        - path_original_labels (str): path of the file with original labels (Y_train)

    Returns:
        - X_train_dominant_colors (np.ndarray)
        - Y_train_dominant_colors (np.ndarray)
    """
    if not os.path.isdir(path_original_dataset):
        raise FileNotFoundError(
            f"The directory {path_original_dataset} does not exist.")
    logging.info(
        f"READING THE DATASET IN THE FOLDER {path_original_dataset} AND TRANSFORMING IT IN THE DOMINANT COLOR VERSION")

    Y_train_original = data_loading.load_ds_labels_from_csv(
        path_original_labels)
    Y_train_original = Y_train_original['eye_color']
    X_train_dominant_colors = []
    Y_train_dominant_colors = []

    def sorting_lambda(image_name): return int(image_name.split(".")[0])
    files = sorted(os.listdir(path_original_dataset), key=sorting_lambda)

    i = 0
    for file in files:
        if file.endswith(".png"):
            # if i > 1000:
            #     break
            print(f"Progress: {i} \ {len(files)}", end='\r')
            color_thief = ColorThief(
                os.path.join(path_original_dataset, file))
            # get the dominant color
            dominant_color = color_thief.get_color(quality=10)
            label = Y_train_original[i]
            X_train_dominant_colors.append(dominant_color)
            Y_train_dominant_colors.append(label)
            #print(file, label)
            i += 1
    print("")

    return np.array(X_train_dominant_colors), np.array(Y_train_dominant_colors)
