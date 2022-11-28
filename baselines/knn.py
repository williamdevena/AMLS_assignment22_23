from src import (
    data_loading,
    costants
)

from utilities import logging_utilities

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import sys
import os
import logging
import json
from pprint import pformat


sys.path.append("../")


logging.basicConfig(format="%(message)s", level=logging.INFO)


def knn_for_every_dataset():
    """"
    Train and tests knn and writes the results for every combination of dataset
    and target label:
        - 'celeba' dataset and 'gender' label
        - 'celeba' dataset and 'smiling' label
        - 'cropped mouth celeba' dataset and 'gender' label
        - 'cropped mouth celeba' dataset and 'smiling' label
        - 'cropped eyes celeba' dataset and 'gender' label
        - 'cropped eyes celeba' dataset and 'smiling' label
        - 'cartoon' dataset and 'eye_color' label
        - 'cartoon' dataset and 'face_shape' label
        - 'cropped eye cartoon' dataset and 'eye_color' label
        - 'cropped eye cartoon' dataset and 'face_shape' label

    Args: None

    Returns: None

    """
    logging_utilities.print_name_stage_project("KNN")
    path_directory = os.path.join(
        costants.PATH_PLOTS_LOGS_FOLDER, "knn_results")
    if not os.path.isdir(path_directory):
        os.mkdir(path_directory)
    array_k = [20, 30, 40]
    for key, value in costants.DICT_COMBINATIONS_DATASETS_LABELS_DIMENSIONS.items():
        dataset = key
        array_labels = value[0]
        image_dimensions = value[1]
        for label in array_labels:
            file_name = f"knn_{dataset}_{label}"
            final_path = os.path.join(path_directory, file_name)
            scores = []
            X_train, Y_train, X_test, Y_test = data_loading.load_X_Y_train_test(
                dataset, label, image_dimensions
            )
            for k in array_k:
                score = knn(k, X_train, Y_train, X_test, Y_test)
                scores.append((k, score))
            write_knn_results(final_path, dataset, label, scores)


def knn(k, X_train, Y_train, X_test, Y_test):
    """
    Uses KNN on a dataset. It first fits the training data (X_train) and then returns the score
    on a test dataset (X_test).

    Args:
        - k (int): the k paramater for the KNN model
        - X_train (np.ndarray): training dataset (the dataset that is divided in k subdatasets)
        - Y_train (np.ndarray): training labels
        - X_test (np.ndarray): testing dataset (for the final evaluation of the best hyperparameter selected)
        - Y_test (np.ndarray): testing labels (for the final evaluation of the best hyperparameter selected)

    Returns:
        - score (float): accuracy of the model on the test dataset (X_test)
    """
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, Y_train)
    score = knn_model.score(X_test, Y_test)

    return score


def knn_k_fold_cross_validation_for_every_dataset(
    k_fold_params, array_possible_hyperparameter
):
    """ 
    Performs k-fold cross validation multiple times for multiple k's (k-fold parameter)
    for every for every combination of dataset and target label:
        - 'celeba' dataset and 'gender' label
        - 'celeba' dataset and 'smiling' label
        - 'cropped mouth celeba' dataset and 'gender' label
        - 'cropped mouth celeba' dataset and 'smiling' label
        - 'cropped eyes celeba' dataset and 'gender' label
        - 'cropped eyes celeba' dataset and 'smiling' label
        - 'cartoon' dataset and 'eye_color' label
        - 'cartoon' dataset and 'face_shape' label
        - 'cropped eye cartoon' dataset and 'eye_color' label
        - 'cropped eye cartoon' dataset and 'face_shape' label

    Args:
        - k_fold_params (list): list of k's (k-fold parameters) to test
        - array_possible_hyperparameter (list): list of hyperparameters (k of the knn model) to test

    Returns: None
    """
    logging_utilities.print_name_stage_project("KNN MULTIPLE K-FOLD CROSS VALIDATION")
    path_directory = os.path.join(
        costants.PATH_PLOTS_LOGS_FOLDER, "knn_cross_validation")
    if not os.path.isdir(path_directory):
        os.mkdir(path_directory)

    for key, value in costants.DICT_COMBINATIONS_DATASETS_LABELS_DIMENSIONS.items():
        dataset = key
        array_labels = value[0]
        image_dimensions = value[1]
        for label in array_labels:
            X_train, Y_train, X_test, Y_test = data_loading.load_X_Y_train_test(
                dataset, label, image_dimensions
            )
            results_of_every_k_fold_param = knn_k_fold_cross_validation_with_multiple_k(
                k_fold_params,
                array_possible_hyperparameter,
                X_train,
                Y_train,
                X_test,
                Y_test,
            )

            # plot and write results of cross validation
            descriptive_string_for_logs_and_plots = f"k_fold_{dataset}_{label}"
            path_plot = os.path.join(
                path_directory,
                "".join(["plot_", descriptive_string_for_logs_and_plots]),
            )
            path_file = os.path.join(
                path_directory,
                "".join(["log_", descriptive_string_for_logs_and_plots, ".txt"]),
            )
            plot_knn_k_fold_cross_validation(
                path_plot,
                dataset,
                label,
                results_of_every_k_fold_param
            )
            write_knn_k_fold_cross_validation(
                path_file,
                dataset,
                label,
                results_of_every_k_fold_param
            )


def knn_k_fold_cross_validation_with_multiple_k(
    k_fold_params,
    array_possible_hyperparameter,
    X_train,
    Y_train,
    X_test,
    Y_test,
):
    """
    Performs k-fold cross validation multiple times for multiple k's (k-fold parameter).

    Args:
        - k_fold_params (list): list of k fold parameters to test
        - array_possible_hyperparameter (list): list of hyperparameters (k of the knn model) to test
        - X_train (np.ndarray): training dataset (the dataset that is divided in k subdatasets)
        - Y_train (np.ndarray): training labels
        - X_test (np.ndarray): testing dataset (for the final evaluation of the best hyperparameter selected)
        - Y_test (np.ndarray): testing labels (for the final evaluation of the best hyperparameter selected)

    Returns:
        - results_of_every_k_fold_param (dict): for very k in k_fold_params contains a list of three 
        elements, that are array_hyperparams_and_scores, best_hyperparam, final_score_best_model 
        (see the documentation of the function knn_k_fold_cross_validation for more details)
    """
    results_of_every_k_fold_param = {}
    for k_fold_param in k_fold_params:
        # perform k-fold cross validation
        (
            array_hyperparams_and_scores,
            best_hyperparam,
            final_score_best_model,
        ) = knn_k_fold_cross_validation(
            k=k_fold_param,
            array_possible_hyperparameter=array_possible_hyperparameter,
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
        )
        results_of_every_k_fold_param[k_fold_param] = [
            array_hyperparams_and_scores,
            best_hyperparam,
            final_score_best_model
        ]

    return results_of_every_k_fold_param


def knn_k_fold_cross_validation(
    k, array_possible_hyperparameter, X_train, Y_train, X_test, Y_test
):
    """
    Performs k-fold cross validation with KNN model

    Args:
        - k (int): k-fold parameter (in how many subdataset is the training dataset divided)
        - array_possible_hyperparameter (list): list of hyperparameters (k of the knn model) to test
        - X_train (np.ndarray): training dataset (the dataset that is divided in k subdatasets)
        - Y_train (np.ndarray): training labels
        - X_test (np.ndarray): testing dataset (for the final evaluation of the best hyperparameter selected)
        - Y_test (np.ndarray): testing labels (for the final evaluation of the best hyperparameter selected)

    Returns:
        - array_hyperparameter_and_scores (list): contains tuple of the form (hyperparam, score) that represent
        all the hyperparameters tested and their results
        - best_hyperparam (int): best hyperparameter selected (with the highest score)
        - final_score_best_hyperparam (float): score of the best hyperparameter selected
    """
    array_hyperparameter_and_scores = []
    logging.info(f"Calculating {k}-fold cross validation scores")

    for hyperparameter in array_possible_hyperparameter:
        logging.info(f"Calculating score for k = {hyperparameter}")
        model = KNeighborsClassifier(n_neighbors=hyperparameter)
        scores = cross_val_score(model, X_train, Y_train, cv=k)
        mean_score = scores.mean()
        logging.info(f"Score: {mean_score}")
        array_hyperparameter_and_scores.append((hyperparameter, mean_score))

    best_hyperparam = max(
        array_hyperparameter_and_scores, key=lambda x: x[1]
    )[0]
    best_model = KNeighborsClassifier(n_neighbors=best_hyperparam)
    best_model.fit(X_train, Y_train)
    final_score_best_hyperparam = best_model.score(X_test, Y_test)

    return (
        array_hyperparameter_and_scores,
        best_hyperparam,
        final_score_best_hyperparam,
    )


def plot_knn_k_fold_cross_validation(
    path_plot, dataset, label, results_of_every_k_fold_param
):
    """
    Plots the scores of all the k-fold cross validations (with multiple k's).
    For every k-fold execution it plots the score of the different hyperparameters
    tested and saves the plot locally.

    Args:
        - path_plot (str): path where to save the plot
        - dataset (str): dataset tested
        - label (str): target variable tested
        - results_of_every_k_fold_param: results of every k-fold execution
        (see documentation of the function knn_k_fold_cross_validation_for_every_dataset
        for more details)

    Returns:
        - None
    """
    for k_fold_param, array_hyperparams_and_scores, best_hyperparam, final_score_best_model in results_of_every_k_fold_param:
        hyperparams, scores = zip(*array_hyperparams_and_scores)
        plt.plot(hyperparams, scores)
        plt.title(f"k-fold on {dataset} with {label} target")
    plt.legend(
        [f"k = {k}" for k in k_fold_param],
        loc="best"
    )
    plt.savefig(path_plot)
    logging.info(f"Saved plot of results in {path_plot}")


def write_knn_k_fold_cross_validation(
    path_file, dataset, label, results_of_every_k_fold_param
):
    """
    Writes the scores of all the k-fold cross validations (with multiple k's).
    For every k-fold execution it writes the score of the different hyperparameters
    tested.

    Args:
        - path_file (str): path where to save the file
        - dataset (str): dataset tested
        - label (str): target variable tested
        - results_of_every_k_fold_param: results of every k-fold execution
        (see documentation of the function knn_k_fold_cross_validation_for_every_dataset
        for more details)

    Returns:
        - None


    """
    best_k_fold = max(results_of_every_k_fold_param.values(),
                      key=lambda x: x[2])[0, 2]
    with open(path_file, "w") as f:
        lines = [
            f"This file contains the scores of multiple executions of k-fold cross validation (for different k's) done on {dataset} dataset and {label} target variable.\n",
            f"The best k (k-fold parameter) was {best_k_fold[0]}, obtaining a score of {best_k_fold[1]}\n\n"

        ]

        for k_fold_param, array_hyperparams_and_scores, best_hyperparam, final_score_best_model in results_of_every_k_fold_param:
            single_k_fold_lines = [
                f"{k_fold_param}-FOLD\n",
            ]
            for hyperparam, score in array_hyperparams_and_scores:
                single_k_fold_lines.append(
                    f"- k = {hyperparam}, score = {score}\n"
                )
            single_k_fold_lines.append(
                f"\nThe best model found was the one with k = {best_hyperparam}.\n"
            )
            single_k_fold_lines.append(
                f"The best model tested on the entire test dataset has an accuracy of {final_score_best_model}.\n\n"
            )
            lines = lines + single_k_fold_lines
    logging.info(f"Wrote the results in {path_file}")


def write_knn_results(
    file_path, dataset, label, scores
):
    """
    Writes the reults on knn experiments done on one dataset trying to predict
    a certain label with multiple k parameters.

    Args:
        file_path (str): path of the file we want to write
        dataset (str): represents which dataset are the results on that we
        are writing
        label (str): represents the label that the results regard
        scores (list): contains the accuracies of the knn models on the testing data

    Returns: None
    """
    logging.info(f"Writing results on {file_path}")
    best_k = max(scores, key=lambda x: x[1])
    with open(file_path, "w") as f:
        lines = [
            f"This file contains the scores calculated for KNN models applied on:\n",
            "\n",
            f"- DATASET: {dataset}\n",
            f"- LABEL: {label}\n",
            "\n"
        ]
        score_lines = [f"- k = {k}, score = {score}\n" for k, score in scores]
        lines = lines + score_lines
        lines.append(
            f"\nBEST MODEL: k = {best_k[0]}, score = {best_k[1]}"
        )
        f.writelines(lines)


def main():
    pass


if __name__ == "__main__":
    main()
