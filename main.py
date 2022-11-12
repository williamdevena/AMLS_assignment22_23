import logging

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
from src.data_preparation import *
from src.data_visualization import *
from utilities.logging_utilities import *

logging.basicConfig(format="%(message)s", level=logging.INFO)


def data_preparation():
    print_name_stage_project("DATA PREPARATION")
    check_values_from_csv(PATH_CARTOON_TRAIN_LABELS)
    check_values_from_csv(PATH_CELEBA_TRAIN_LABELS)
    check_values_from_csv(PATH_CARTOON_TEST_LABELS)
    check_values_from_csv(PATH_CELEBA_TEST_LABELS)
    check_shape_images(PATH_CELEBA_TRAIN_IMG)
    check_shape_images(PATH_CELEBA_TEST_IMG)
    check_shape_images(PATH_CARTOON_TRAIN_IMG)
    check_shape_images(PATH_CARTOON_TEST_IMG)


def data_visualization():
    print_name_stage_project("DATA VISUALIZATION")
    visualize_hist_distribution_csv(PATH_CARTOON_TRAIN_LABELS)
    visualize_hist_distribution_csv(PATH_CELEBA_TRAIN_LABELS)


def main():
    data_preparation()
    data_visualization()


if __name__ == "__main__":
    main()
