import logging
import torch
import torch.optim as optim
from torch import nn
from torchvision import models
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import dlib
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from src import (
    data_loading,
    data_preparation,
    data_visualization,
    costants,
    image_manipulation,
    seeds,
    training
)

from baselines import face_feature_detector

from baselines.knn import knn_for_every_dataset
from baselines.svm import svm_for_every_dataset
from baselines import svm, knn

from models.smile_cnn import SmileCNN

from assignment_dataset import AssignmentDataset
from pytorch_dataset import PytorchDataset

logging.basicConfig(format="%(message)s", level=logging.INFO)


def main():
    '''
    SET SEEDS FOR REPRODUCABILITY
    '''
    seeds.set_seeds()

    '''
    DATA PREPARATION
    '''

    # name_new_folder = "img_transform"
    # path_new_folder_cartoon_train = os.path.join(
    #     costants.PATH_CARTOON_TRAIN_FOLDER,
    #     name_new_folder
    # )
    # path_new_folder_cartoon_test = os.path.join(
    #     costants.PATH_CARTOON_TEST_FOLDER,
    #     name_new_folder
    # )

    # # Trasforming the cartoon set images from from RGB-A to RGB
    # # and writing them in a new folder
    # image_manipulation.transform_ds_rgba_to_rgb(
    #     costants.PATH_CARTOON_TRAIN_IMG, new_folder=path_new_folder_cartoon_train)

    # image_manipulation.transform_ds_rgba_to_rgb(
    #     costants.PATH_CARTOON_TEST_IMG, new_folder=path_new_folder_cartoon_test)

    """
    STATIC CROPPED DATASET CREATION
    """

    # Cropped eye Cartoon Set dataset
    path_original_train_folder = costants.PATH_CARTOON_TRAIN_IMG
    path_cropped_train_folder = costants.PATH_CARTOON_TRAIN_CROPPED_EYE_IMG
    path_original_test_folder = costants.PATH_CARTOON_TEST_IMG
    path_cropped_test_folder = costants.PATH_CARTOON_TEST_CROPPED_EYE_IMG
    crop_y = slice(248, 277)
    crop_x = slice(283, 308)
    extension_image = ".png"
    image_manipulation.crop_images_dataset(
        path_original_train_folder, path_cropped_train_folder, extension_image, crop_y, crop_x)
    image_manipulation.crop_images_dataset(
        path_original_test_folder, path_cropped_test_folder, extension_image, crop_y, crop_x)

    # Cropped eyes CelebA dataset
    path_original_train_folder = costants.PATH_CELEBA_TRAIN_IMG
    path_cropped_train_folder = costants.PATH_CELEBA_TRAIN_CROPPED_EYES_IMG
    path_original_test_folder = costants.PATH_CELEBA_TEST_IMG
    path_cropped_test_folder = costants.PATH_CELEBA_TEST_CROPPED_EYES_IMG
    crop_y = slice(90, 128)
    crop_x = slice(40, 138)
    extension_image = ".jpg"
    image_manipulation.crop_images_dataset(
        path_original_train_folder, path_cropped_train_folder, extension_image, crop_y, crop_x)
    image_manipulation.crop_images_dataset(
        path_original_test_folder, path_cropped_test_folder, extension_image, crop_y, crop_x)

    # Cropped mouth CelebA dataset
    path_original_train_folder = costants.PATH_CELEBA_TRAIN_IMG
    path_cropped_train_folder = costants.PATH_CELEBA_TRAIN_CROPPED_MOUTH_IMG
    path_original_test_folder = costants.PATH_CELEBA_TEST_IMG
    path_cropped_test_folder = costants.PATH_CELEBA_TEST_CROPPED_MOUTH_IMG
    crop_y = slice(140, 173)
    crop_x = slice(60, 118)
    extension_image = ".jpg"
    image_manipulation.crop_images_dataset(
        path_original_train_folder, path_cropped_train_folder, extension_image, crop_y, crop_x)
    image_manipulation.crop_images_dataset(
        path_original_test_folder, path_cropped_test_folder, extension_image, crop_y, crop_x)

    """
    DYNAMIC CROP (WITH FACE DETECTOR)
    """
    # Load the detector
    detector = dlib.get_frontal_face_detector()

    # Load the predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Mouth (CELEBA)
    range_features = range(49, 68)
    resized_shape = (48, 25)
    extension = ".jpg"

    face_feature_detector.crop_images_dinamycally_using_face_features(costants.PATH_CELEBA_TRAIN_IMG,
                                                                      costants.PATH_CELEBA_TRAIN_DYN_CROPPED_MOUTH_IMG,
                                                                      extension,
                                                                      detector,
                                                                      predictor,
                                                                      range_features,
                                                                      resized_shape
                                                                      )

    face_feature_detector.crop_images_dinamycally_using_face_features(costants.PATH_CELEBA_TEST_IMG,
                                                                      costants.PATH_CELEBA_TEST_DYN_CROPPED_MOUTH_IMG,
                                                                      extension,
                                                                      detector,
                                                                      predictor,
                                                                      range_features,
                                                                      resized_shape
                                                                      )

    # Eyes (CELEBA)
    range_features = range(37, 48)
    resized_shape = (65, 25)
    extension = ".jpg"

    face_feature_detector.crop_images_dinamycally_using_face_features(costants.PATH_CELEBA_TRAIN_IMG,
                                                                      costants.PATH_CELEBA_TRAIN_DYN_CROPPED_EYES_IMG,
                                                                      extension,
                                                                      detector,
                                                                      predictor,
                                                                      range_features,
                                                                      resized_shape
                                                                      )

    face_feature_detector.crop_images_dinamycally_using_face_features(costants.PATH_CELEBA_TEST_IMG,
                                                                      costants.PATH_CELEBA_TEST_DYN_CROPPED_EYES_IMG,
                                                                      extension,
                                                                      detector,
                                                                      predictor,
                                                                      range_features,
                                                                      resized_shape
                                                                      )

    # Eye (CARTOON)
    range_features = range(42, 48)
    resized_shape = (30, 30)
    extension = ".png"

    face_feature_detector.crop_images_dinamycally_using_face_features(costants.PATH_CARTOON_TRAIN_IMG,
                                                                      costants.PATH_CARTOON_TRAIN_DYN_CROPPED_EYE_IMG,
                                                                      extension,
                                                                      detector,
                                                                      predictor,
                                                                      range_features,
                                                                      resized_shape
                                                                      )

    face_feature_detector.crop_images_dinamycally_using_face_features(costants.PATH_CARTOON_TEST_IMG,
                                                                      costants.PATH_CARTOON_TEST_DYN_CROPPED_EYE_IMG,
                                                                      extension,
                                                                      detector,
                                                                      predictor,
                                                                      range_features,
                                                                      resized_shape
                                                                      )

    ### Face (CELEBA and CARTOON)
    range_features = range(0, 68)
    resized_shape = (125, 140)

    extension = ".png"
    face_feature_detector.crop_images_dinamycally_using_face_features(costants.PATH_CARTOON_TRAIN_IMG,
                                                                      costants.PATH_CARTOON_TRAIN_DYN_CROPPED_FACE_IMG,
                                                                      extension,
                                                                      detector,
                                                                      predictor,
                                                                      range_features,
                                                                      resized_shape
                                                                      )

    face_feature_detector.crop_images_dinamycally_using_face_features(costants.PATH_CARTOON_TEST_IMG,
                                                                      costants.PATH_CARTOON_TEST_DYN_CROPPED_FACE_IMG,
                                                                      extension,
                                                                      detector,
                                                                      predictor,
                                                                      range_features,
                                                                      resized_shape
                                                                      )

    extension = ".jpg"
    face_feature_detector.crop_images_dinamycally_using_face_features(costants.PATH_CELEBA_TRAIN_IMG,
                                                                      costants.PATH_CELEBA_TRAIN_DYN_CROPPED_FACE_IMG,
                                                                      extension,
                                                                      detector,
                                                                      predictor,
                                                                      range_features,
                                                                      resized_shape
                                                                      )

    face_feature_detector.crop_images_dinamycally_using_face_features(costants.PATH_CELEBA_TEST_IMG,
                                                                      costants.PATH_CELEBA_TEST_DYN_CROPPED_FACE_IMG,
                                                                      extension,
                                                                      detector,
                                                                      predictor,
                                                                      range_features,
                                                                      resized_shape
                                                                      )

    """
    CREATE AND WRITE LANDMARK FEATURES DATASETS
    """
    # Load the detector
    detector = dlib.get_frontal_face_detector()

    # Load the predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    range_features = range(0, 68)  # ALL

    # CREATE AND WRITE FACE FEATURES DATASETS
    face_feature_detector.create_and_write_face_features_datasets(detector=detector,
                                                                  predictor=predictor,
                                                                  range_features=range_features)

    """
    SVM AND KNN ON FACE FEATURES
    """

    # LOAD DS FROM CSV OF FEATURE FACE

    # FACE SHAPE
    logging.info(
        "\n\n- Face shape recognition with landmark features (jaw features)")
    label = 'face_shape'
    train_csv_path = costants.PATH_CARTOON_TRAIN_FACE_FEATURES
    test_csv_path = costants.PATH_CARTOON_TEST_FACE_FEATURES
    # # (0, 136) for all features
    # # (0, 34) for jaw features

    feature_slice = slice(0, 34)
    # #feature_slice = slice(0, 136)

    train_dataset_dict = data_loading.load_entire_ds_from_csv(
        csv_path=train_csv_path)
    test_dataset_dict = data_loading.load_entire_ds_from_csv(
        csv_path=test_csv_path)

    X_train = train_dataset_dict['X'][:, feature_slice]
    X_test = test_dataset_dict['X'][:, feature_slice]
    #print(X_train.shape, X_test.shape)

    Y_train = train_dataset_dict['Y'][label]
    Y_test = test_dataset_dict['Y'][label]

    Y_pred, training_acc, testing_acc = svm.svm(kernel='rbf', C=1.0,
                                                X_train=X_train,
                                                Y_train=Y_train,
                                                X_test=X_test,
                                                Y_test=Y_test)

    logging.info(f"\n- SVM score: {testing_acc}")

    Y_pred, training_acc, testing_acc = knn.knn(
        30, X_train, Y_train, X_test, Y_test)

    logging.info(f"\n- KNN score: {testing_acc}")

    # SMILE DETECTION
    logging.info("\n\n- Smile detection with landmark features")
    label = 'smiling'
    train_csv_path = costants.PATH_CELEBA_TRAIN_FACE_FEATURES
    test_csv_path = costants.PATH_CELEBA_TEST_FACE_FEATURES
    # # (0, 136) for all features
    # # (0, 34) for jaw features

    feature_slice = slice(0, 136)
    # #feature_slice = slice(0, 136)

    train_dataset_dict = data_loading.load_entire_ds_from_csv(
        csv_path=train_csv_path)
    test_dataset_dict = data_loading.load_entire_ds_from_csv(
        csv_path=test_csv_path)

    X_train = train_dataset_dict['X'][:, feature_slice]
    X_test = test_dataset_dict['X'][:, feature_slice]
    #print(X_train.shape, X_test.shape)

    Y_train = train_dataset_dict['Y'][label]
    Y_test = test_dataset_dict['Y'][label]

    Y_pred, training_acc, testing_acc = svm.svm(kernel='rbf', C=1.0,
                                                X_train=X_train,
                                                Y_train=Y_train,
                                                X_test=X_test,
                                                Y_test=Y_test)

    logging.info(f"\n- SVM score: {testing_acc}")

    Y_pred, training_acc, testing_acc = knn.knn(
        30, X_train, Y_train, X_test, Y_test)

    logging.info(f"\n- KNN score: {testing_acc}")

    # GENDER DETECTION
    logging.info("\n\n- Gender detection with landmark features")
    label = 'gender'
    train_csv_path = costants.PATH_CELEBA_TRAIN_FACE_FEATURES
    test_csv_path = costants.PATH_CELEBA_TEST_FACE_FEATURES
    # # (0, 136) for all features
    # # (0, 34) for jaw features

    feature_slice = slice(0, 136)
    # #feature_slice = slice(0, 136)

    train_dataset_dict = data_loading.load_entire_ds_from_csv(
        csv_path=train_csv_path)
    test_dataset_dict = data_loading.load_entire_ds_from_csv(
        csv_path=test_csv_path)

    X_train = train_dataset_dict['X'][:, feature_slice]
    X_test = test_dataset_dict['X'][:, feature_slice]
    #print(X_train.shape, X_test.shape)

    Y_train = train_dataset_dict['Y'][label]
    Y_test = test_dataset_dict['Y'][label]

    Y_pred, training_acc, testing_acc = svm.svm(kernel='rbf', C=1.0,
                                                X_train=X_train,
                                                Y_train=Y_train,
                                                X_test=X_test,
                                                Y_test=Y_test)

    logging.info(f"\n- SVM score: {testing_acc}")

    Y_pred, training_acc, testing_acc = knn.knn(
        30, X_train, Y_train, X_test, Y_test)

    logging.info(f"\n- KNN score: {testing_acc}")

    """
    KNN
    """
    knn_for_every_dataset()

    '''
    KNN CROSS VALIDATION
    '''
    # k_fold_params = range(5, 11)
    # array_possible_hyperparameter = range(10, 101, 10)
    # k_fold_params = range(5, 7)
    # array_possible_hyperparameter = range(10, 31, 10)
    # knn.knn_k_fold_cross_validation_for_every_dataset(
    #     k_fold_params, array_possible_hyperparameter
    # )

    """
    SVM ON ALL DATASETS
    """
    svm_for_every_dataset()

    """
    CNN TESTING
    """

    # CNN ON FACE SHAPE RECOGNITION

    dataset_name = "dyn_cropped_face_cartoon"
    label_name = "face_shape"
    test_transform = A.Compose(
        [
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    test_set = PytorchDataset(
        dataset_name=dataset_name, label_name=label_name, train_or_test="test",
        transform=test_transform,
        validation_split=7500)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.efficientnet_b0(weights='DEFAULT')
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=5, bias=True)
    )
    model = model.to(device)

    test_dataloader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=1,
                                                  shuffle=False)

    model.load_state_dict(torch.load(
        "./weights_cnn/weights_cnn_face_shape", map_location=torch.device(device)))

    test_acc = training.testing(model, device, test_dataloader, binary=False)
    logging.info(
        f"\n- EfficientNet B0 accuracy on face shape recognition: {test_acc}")

    #####
    # CNN ON GENDER DETECTION
    #####

    dataset_name = "dyn_cropped_face_celeba"
    label_name = "gender"
    test_transform = A.Compose(
        [
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    test_set = PytorchDataset(
        dataset_name=dataset_name, label_name=label_name, train_or_test="test",
        transform=test_transform,
        validation_split=4000)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.efficientnet_b0(weights='DEFAULT')
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=1, bias=True)
    )
    model = model.to(device)

    test_dataloader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=1,
                                                  shuffle=False)

    model.load_state_dict(torch.load(
        "./weights_cnn/weights_cnn_gender", map_location=torch.device(device)))

    test_acc = training.testing(model, device, test_dataloader, binary=True)
    logging.info(
        f"\n- EfficientNet B0 accuracy on gender detection: {test_acc}")


if __name__ == "__main__":
    main()
