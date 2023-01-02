"""
This module contains all the functions used to run knn models as a baseline model
"""

from sklearn.svm import SVC
import os
import logging

from utilities import logging_utilities
from src import (
    data_loading,
    data_visualization,
    costants
)

from assignment_dataset import AssignmentDataset

def svm_for_every_dataset():
    """
    Train and tests SVM and writes the results for every combination of dataset
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
    logging_utilities.print_name_stage_project("SVM")
    
    scores = []
    path_directory = os.path.join(
        costants.PATH_PLOTS_LOGS_FOLDER, "svm")
    if not os.path.isdir(path_directory):
        os.mkdir(path_directory)
    path_directory_conf_matrix = os.path.join(
        path_directory, "conf_matrix")
    if not os.path.isdir(path_directory_conf_matrix):
        os.mkdir(path_directory_conf_matrix)
    path_results_file = os.path.join(path_directory, "svm_results")
    
    for key, value in AssignmentDataset.possible_combinations.items():
        dataset_name = key
        array_labels = value[0]
        for label_name in array_labels:
            logging.info(f"- DATASET: {dataset_name}")
            logging.info(f"- LABEL: {label_name}")
            conf_matrix_plot_file = f"{dataset_name}_{label_name}"
            conf_matrix_plot_path = os.path.join(
                path_directory_conf_matrix, conf_matrix_plot_file)
            
            dataset = AssignmentDataset(name=dataset_name, label=label_name)
            X_train, Y_train, X_test, Y_test = data_loading.load_X_Y_train_test(
                dataset_object=dataset
            )
            Y_pred, training_acc, testing_acc = svm(kernel='rbf', C=1.0,
                                                    X_train=X_train,
                                                    Y_train=Y_train,
                                                    X_test=X_test,
                                                    Y_test=Y_test)
            
            scores.append((dataset_name, label_name, training_acc,
                           testing_acc))
            labels_value = AssignmentDataset.labels_values[label_name]
            data_visualization.plot_confusion_matrix(
                    Y=Y_test, Y_pred=Y_pred, labels=labels_value, path_plot=conf_matrix_plot_path)
    write_svm_results(path_results_file, scores)

def svm(kernel, C, X_train, Y_train, X_test, Y_test):
    """
    Uses SVM on a dataset. It first fits the training data (X_train) and then returns the score
    on a test dataset (X_test).

    Args:
        - kernel (str): Specifies the kernel type to be used in the algorithm
        - C (float): Regularization parameter. The strength of the regularization is inversely proportional to C
        - X_train (np.ndarray): training dataset (the dataset that is divided in k subdatasets)
        - Y_train (np.ndarray): training labels
        - X_test (np.ndarray): testing dataset (for the final evaluation of the best hyperparameter selected)
        - Y_test (np.ndarray): testing labels (for the final evaluation of the best hyperparameter selected)
        
    Returns:
        - Y_pred (np.ndarray): predictions on the test data
        - training_acc (float): model mean accuracy on the train data (X_train)
        - testing_acc (float): model mean accuracy on the test data (X_test)
    """
    svm = SVC(C=C, kernel=kernel)
    svm.fit(X=X_train, y=Y_train)
    Y_pred = svm.predict(X=X_test)
    training_acc = svm.score(X=X_train, y=Y_train)
    testing_acc = svm.score(X=X_test, y=Y_test)
    
    return Y_pred, training_acc, testing_acc

def write_svm_results(file_path, scores):
    """
    Writes the results of SVM on every combination of dataset and target variable.

    Args:
        - file_path (str): path of the file we want to write
        - scores (List): contains tuples in the form (dataset, label, training_acc, testing_acc)
        
    Returns: None
    """
    logging.info(f"Writing results on {file_path}")
    with open(file_path, "w") as f:
        lines = [
            f"This file contains the scores of SVM model\n",
            "\n"
        ]
        for dataset, label, training_acc, testing_acc in scores:
            lines.append(f"\n- DATASET: {dataset}\n")
            lines.append(f"- LABEL: {label}\n")
            lines.append(f"- TRAINING ACCURACY: {training_acc}\n")
            lines.append(f"- TESTING ACCURACY: {testing_acc}\n")
        f.writelines(lines)
    