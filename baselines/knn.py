import src.data_loading as data_loading
import src.costants as costants

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn

import sys
import os
import logging


sys.path.append("../")


logging.basicConfig(format="%(message)s", level=logging.INFO)


def knn_k_fold_cross_validation_with_multiple_k(
    k_fold_params,
    array_possible_hyperparameter,
    dataset,
    labels,
    knn_plots_and_log_directory,
):
    """
    Performs k-fold cross validation multiple times for multiple k's.
    It also plots and writes the results of every k-fold execution.

    Args:
        - k_fold_params (list): list of k fold parameters to test
        - array_possible_hyperparameter (list): list of hyperparameters (k of the knn model) to test
        - dataset (str): can be 'cartoon' or 'celeba' and represents the dataset we want to use
        - labels (str): can be 'gender' and 'smiling' for the cartoon dataset and 'eye_color' and
        'face_shape' for the celeba dataset
        - knn_plots_and_log_directory (str): path of the directory where to save the plots and files of the results

    Returns:
        - None
    """

    X_train, Y_train, X_test, Y_test = data_loading.load_X_Y_train_test(
        dataset, labels
    )

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

        # plot and write results of cross validation
        try:
            descriptive_string_for_logs_and_plots = f"{k_fold_param}_fold_{array_possible_hyperparameter[0]}_{array_possible_hyperparameter[-1]}_{array_possible_hyperparameter[1]-array_possible_hyperparameter[0]}"
        except IndexError as e:
            logging.error(
                "IndexError as occured because the array_possible_hyperparameter has length <=1"
            )

        path_plot = os.path.join(
            knn_plots_and_log_directory,
            "".join(["plot_", descriptive_string_for_logs_and_plots]),
        )
        path_file = os.path.join(
            knn_plots_and_log_directory,
            "".join(["log_", descriptive_string_for_logs_and_plots, ".txt"]),
        )
        plot_knn_k_fold_cross_validation(
            path_plot,
            k_fold_param,
            array_hyperparams_and_scores,
        )
        write_knn_k_fold_cross_validation(
            path_file,
            k_fold_param,
            array_hyperparams_and_scores,
            best_hyperparam,
            final_score_best_model,
        )


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
        #mlflow.log_metric(f"{k}-fold scores", scores)
        mlflow.log_metric(f"Mean score", mean_score)

    best_hyperparam = max(
        array_hyperparameter_and_scores, key=lambda x: x[1]
    )[0]
    best_model = KNeighborsClassifier(n_neighbors=best_hyperparam)
    best_model.fit(X_train, Y_train)
    mlflow.sklearn.log_model(best_model, "best model")
    final_score_best_hyperparam = best_model.score(X_test, Y_test)
    mlflow.log_metric(f"Score of best model", final_score_best_hyperparam)

    return (
        array_hyperparameter_and_scores,
        best_hyperparam,
        final_score_best_hyperparam,
    )


def plot_knn_k_fold_cross_validation(
    path_plot, k_fold_param, array_hyperparam_and_scores
):
    """
    Plots the scores of all the hyperparameter tested during k-fold cross validation and
    saves the plot locally

    Args:
        - path_plot (str): path where to save the plot
        - k_fold_param (int): k-fold parameter
        - array_hyperparam_and_scores (list): contains tuple of the form (hyperparam, score) that represent
        all the hyperparameters tested during k-fold cross validation and their results

    Returns:
        - None
    """
    hyperparams, scores = zip(*array_hyperparam_and_scores)
    plt.plot(hyperparams, scores)
    plt.title(f"{k_fold_param}-fold cross validation")
    plt.legend(loc="best")
    plt.savefig(path_plot)


def write_knn_k_fold_cross_validation(
    path_file,
    k_fold_param,
    array_hyperparam_and_scores,
    best_hyperparam,
    final_score_best_model,
):
    """
    Plots the scores of all the hyperparameter tested during k-fold cross validation locally

    Args:
        - path_file (str): path where to save the file
        - k_fold_param (int): k-fold parameter
        - array_hyperparam_and_scores (list): contains tuple of the form (hyperparam, score) that represent
        all the hyperparameters tested during k-fold cross validation and their results
        - best_hyperparam (int): best hyperparameter selected during k-fold cross validation
        - final_score_best_model (float): accuracy of the best model (knn with the best k) selected
        during k-fold cross validation on the original test dataset

    Returns:
        - None


    """
    with open(path_file, "w") as f:
        f.write(
            f"This file contains the scores calculated for a KNN model using {k_fold_param}-fold cross validation.\n"
        )
        f.write(
            f"The following lines show the score as follows: (hyperparameter, mean score)\n\n"
        )

        for (hyperparam, score) in array_hyperparam_and_scores:
            f.write(f"- k = {hyperparam},  mean_score = {score}\n")

        f.write(
            f"\nThe best model found was the one with k = {best_hyperparam}.\n"
        )
        f.write(
            f"\nThe best model tested on the entire test dataset has an accuracy of {final_score_best_model}.\n"
        )


def main():
    pass


if __name__ == "__main__":
    main()
