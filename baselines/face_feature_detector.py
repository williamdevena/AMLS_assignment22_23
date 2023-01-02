import cv2
import numpy as np
import os
import logging

from src import data_loading


def crop_images_dinamycally_using_face_features(path_original_folder, path_cropped_folder, extension_image, detector, predictor, range_features,
                                                resized_shape):
    """
    reads the images in a dataset and uses a face feature detector
    to crop a key part (like mouth or eyes)

    Args:
        - path_original_folder (str): path of the original folder that contains the images
        that we want to crop
        - path_cropped_folder (str): path of the folder were we want to write the cropped images
        - extension_image (str): extension of the image (".jpg", "jpeg", ".png", ....)
        - detector (str): face detector
        - predictor (str): face key features predictor
        - range_features (str): range of features that we want to select (mouth, eyes, ...)
        - resized-shape (tuple): final shape of the images

     Returns: None

    """
    if not os.path.isdir(path_cropped_folder):
        os.mkdir(path_cropped_folder)
    for file in os.listdir(path_original_folder):
        if file.endswith(extension_image):
            image = cv2.imread(os.path.join(path_original_folder, file))
            features = extract_face_features(
                image=image, detector=detector, predictor=predictor, range_features=range_features)
            cropping_coordinates = calculate_cropping_coordinates(features)
            slice_y = slice(
                cropping_coordinates[2], cropping_coordinates[3], 1)
            slice_x = slice(
                cropping_coordinates[0], cropping_coordinates[1], 1)
            cropped_image = image[slice_y, slice_x]
            
            
            cropped_resized_image = cv2.resize(cropped_image, resized_shape)
            
            
            cv2.imwrite(os.path.join(path_cropped_folder, file), cropped_resized_image)


def calculate_cropping_coordinates(features):
    """
    Calculates the coordinates of where to perform the cropping based
    on the points of keyface features, extracted using a face feature
    detector

    Args:
        - features (np.ndarray): contains the coordinates of key face points

    Returns:
        - croppint_points (np.ndarray): contains the four coordinates where
        to crop
    """
    max_width = max([point[0] for point in features])
    max_height = max([point[1] for point in features])
    min_width = min([point[0] for point in features])
    min_height = min([point[1] for point in features])

    crop_point_x_1 = max([min_width - 5, 0])
    crop_point_x_2 = max([max_width + 5, 0])
    crop_point_y_1 = max([min_height - 12, 0]) # -5 for mouth and -12 for eyes (to get the eyebrows)
    crop_point_y_2 = max([max_height + 5, 0])

    return np.array([
        crop_point_x_1,
        crop_point_x_2,
        crop_point_y_1,
        crop_point_y_2,
    ])


def create_face_features_dataset(ds_path, csv_path, label, image_dimensions, detector, predictor, range_features):
    """
    Reads a dataset of images and creates a dataset that has
    the key face points coordinates of the images as features

    Args:
        ds_path (_type_): _description_
        csv_path (_type_): _description_
        label (_type_): _description_
        image_dimensions (_type_): _description_
        detector (_type_): _description_
        predictor (_type_): _description_
        range_features (_type_): _description_

    Returns:
        _type_: _description_
    """
    X = []

    images = data_loading.load_images_from_folder(ds_path, image_dimensions)
    labels = data_loading.load_ds_labels_from_csv(csv_path)[label]

    logging.info("- Extracting coordinates of face features")
    for image in images:
        features = extract_face_features(
            image=image, detector=detector, predictor=predictor, range_features=range_features)
        # print(features.shape)
        X.append(features.flatten())

    X = np.array(X)
    labels = np.array(labels)

    return X, labels


def extract_face_features(image, detector, predictor, range_features):
    """

    Args:
        image (_type_): _description_
        detector (_type_): _description_
        predictor (_type_): _description_

    Returns:
        landmarks_array: contains the face features
        of the image in the form of 2D coordinates (x,y)
    """
    # # Convert image into grayscale
    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    landmarks_array = []

    # Use detector to find landmarks
    faces = detector(gray)
    # print(len(faces))

    # If it didn't detect any features
    if len(faces) == 0:
        for n in range_features:
            landmarks_array.append([0, 0])
    else:
        face = faces[0]
        # for face in faces:
        # Create landmark object
        landmarks = predictor(image=gray, box=face)
        # Loop through all the points
        for n in range_features:
            # print(n)
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            #print(x, y)
            landmarks_array.append([x, y])

    return np.array(landmarks_array)
