import os

PROJECT_PATH = "/Users/william.devena/Desktop/UCL/COURSES/AML1/ASSIGNMENT_PROJECT/AMLS_assignment22_23"
DATASET_FOLDER_NAME = "../Datasets"
DATASETS_PATH = os.path.join(PROJECT_PATH, DATASET_FOLDER_NAME)


"""
CELEBA
"""

# TRAINING
PATH_CELEBA_TRAIN_IMG = os.path.join(DATASETS_PATH, "celeba/img")
PATH_CELEBA_TRAIN_CROPPED_MOUTH_IMG = os.path.join(
    DATASETS_PATH, "celeba/img_cropped_mouth")
PATH_CELEBA_TRAIN_CROPPED_EYES_IMG = os.path.join(
    DATASETS_PATH, "celeba/img_cropped_eyes")
PATH_CELEBA_TRAIN_DYN_CROPPED_MOUTH_IMG = os.path.join(
    DATASETS_PATH, "celeba/img_dyn_cropped_mouth")
PATH_CELEBA_TRAIN_DYN_CROPPED_EYES_IMG = os.path.join(
    DATASETS_PATH, "celeba/img_dyn_cropped_eyes")
PATH_CELEBA_TRAIN_LABELS = os.path.join(DATASETS_PATH, "celeba/labels.csv")

PATH_CELEBA_TRAIN_FACE_FEATURES = os.path.join(
    DATASETS_PATH, "face_features/celeba_train.csv")

# TESTING
PATH_CELEBA_TEST_IMG = os.path.join(DATASETS_PATH, "celeba_test/img")
PATH_CELEBA_TEST_CROPPED_MOUTH_IMG = os.path.join(
    DATASETS_PATH, "celeba_test/img_cropped_mouth")
PATH_CELEBA_TEST_CROPPED_EYES_IMG = os.path.join(
    DATASETS_PATH, "celeba_test/img_cropped_eyes")
PATH_CELEBA_TEST_DYN_CROPPED_MOUTH_IMG = os.path.join(
    DATASETS_PATH, "celeba_test/img_dyn_cropped_mouth")
PATH_CELEBA_TEST_DYN_CROPPED_EYES_IMG = os.path.join(
    DATASETS_PATH, "celeba_test/img_dyn_cropped_eyes")
PATH_CELEBA_TEST_LABELS = os.path.join(DATASETS_PATH, "celeba_test/labels.csv")

PATH_CELEBA_TEST_FACE_FEATURES = os.path.join(
    DATASETS_PATH, "face_features/celeba_test.csv")


"""
CARTOON
"""

# TRAINING

PATH_CARTOON_TRAIN_IMG = os.path.join(DATASETS_PATH, "cartoon_set/img")
PATH_CARTOON_TRAIN_CROPPED_EYE_IMG = os.path.join(
    DATASETS_PATH, "cartoon_set/img_eye")
PATH_CARTOON_TRAIN_DYN_CROPPED_EYE_IMG = os.path.join(
    DATASETS_PATH, "cartoon_set/dyn_img_eye")
PATH_CARTOON_TRAIN_LABELS = os.path.join(
    DATASETS_PATH, "cartoon_set/labels.csv")

PATH_CARTOON_TRAIN_FACE_FEATURES = os.path.join(
    DATASETS_PATH, "face_features/cartoon_train.csv")


# TESTING
PATH_CARTOON_TEST_IMG = os.path.join(DATASETS_PATH, "cartoon_set_test/img")
PATH_CARTOON_TEST_CROPPED_EYE_IMG = os.path.join(
    DATASETS_PATH, "cartoon_set_test/img_eye")
PATH_CARTOON_TEST_DYN_CROPPED_EYE_IMG = os.path.join(
    DATASETS_PATH, "cartoon_set_test/dyn_img_eye")
PATH_CARTOON_TEST_LABELS = os.path.join(
    DATASETS_PATH, "cartoon_set_test/labels.csv")

PATH_CARTOON_TEST_FACE_FEATURES = os.path.join(
    DATASETS_PATH, "face_features/cartoon_test.csv")


COLS_CELEBA = ["img_name", "gender", "smiling"]
COLS_CARTOON = ["img_name", "eye_color", "face_shape"]

SEPARATOR = "\t"

PATH_PLOTS_LOGS_FOLDER = os.path.join(PROJECT_PATH, "plots_and_logs")

# DICT_COMBINATIONS_DATASETS = {
#     "cartoon": (['eye_color', 'face_shape'], (500, 500)),
#     # "cartoon": (['eye_color', 'face_shape'], (250, 250)),
#     "celeba": (['gender', 'smiling'], (178, 218)),
#     "cropped_eye_cartoon": (['eye_color'], (25, 29)),
#     "cropped_mouth_celeba": (['gender', 'smiling'], (58, 33)),
#     "cropped_eyes_celeba": (['gender', 'smiling'], (98, 38)),
#     # "dominant_color": (['eye_color'], None)
# }
