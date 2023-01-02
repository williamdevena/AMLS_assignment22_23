"""
This file contains all the functions used to visualize/plot data

"""

import utilities.logging_utilities as logging_utilities
import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

from src.costants import (
    SEPARATOR,
    PATH_CELEBA_TRAIN_LABELS,
    PATH_CARTOON_TRAIN_LABELS,
    PATH_CARTOON_TEST_LABELS,
    PATH_CELEBA_TEST_LABELS,
)

# sys.path.append("../")


def data_visualization_labels():
    """
    Visualizes the histograms of all the labels in the two datasets

    Args: None

    Returns:
        - None
    """
    logging_utilities.print_name_stage_project("DATA VISUALIZATION")
    visualize_hist_distribution_csv(PATH_CARTOON_TRAIN_LABELS)
    visualize_hist_distribution_csv(PATH_CELEBA_TRAIN_LABELS)
    visualize_hist_distribution_csv(PATH_CARTOON_TEST_LABELS)
    visualize_hist_distribution_csv(PATH_CELEBA_TEST_LABELS)


def histogram_df(df):
    """
    Plots a histogram for every column in a dataframe

    Args:
        - df: Dataframe

    Returns:
        - None

    """
    num_keys = len(df.keys())
    figure, axis = plt.subplots(num_keys)
    for (key, index) in zip(df.keys(), range(num_keys)):
        axis[index].hist(df[key], edgecolor="black")
        axis[index].set_title(key)
    plt.show()


def visualize_hist_distribution_csv(path_csv):
    """
    Reads a csv and plots a histogram from every variable in the csv

    Args:
        - path_csv (str): path of the csv file

    Returns:
        - None

    """
    df = pd.read_csv(path_csv, sep=SEPARATOR)
    # .iloc[:, 2:] because we pass the entire dataframe except for the index col and the first col that is the name of the image (img_name)
    histogram_df(df.iloc[:, 2:])


def visualize_clusters_dominant_pupil_colors(labels, centroids):
    """

    """
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')

    # c0 = np.array(labels == 0)
    # c1 = np.array(labels == 1)
    # c2 = np.array(labels == 2)
    # c3 = np.array(labels == 3)
    # c4 = np.array(labels == 4)
    # c5 = np.array(labels == 5)

    c_array = [np.array(labels == i) for i, _ in enumerate(centroids)]

    map(ax.scatter, )

    ax.scatter(X[c0][:, 0], X[c0][:, 1], X[c0][:, 2])
    ax.scatter(X[c1][:, 0], X[c1][:, 1], X[c1][:, 2])
    ax.scatter(X[c2][:, 0], X[c2][:, 1], X[c2][:, 2])
    ax.scatter(X[c3][:, 0], X[c3][:, 1], X[c3][:, 2])
    ax.scatter(X[c4][:, 0], X[c4][:, 1], X[c4][:, 2])
    ax.scatter(X[c5][:, 0], X[c5][:, 1], X[c5][:, 2])
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
               marker='x', s=169, linewidths=10,
               color='black', zorder=50)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    plt.show()


def plot_confusion_matrix(Y, Y_pred, labels, path_plot):
    """
    Generates and plots the confusion matrix

    Args:
        - Y (np.ndarray): True labels
        - Y_pred (np.ndarray): Predicted labels
        - labels (List): contains the names of the labels (example: ['smiling', 'non smiling'])
        - path_plot (str): path where to save the plot

    Returns: None
    """
    #print(Y, Y_pred)
    # print(np.unique(Y_pred, return_counts=True))
    cf_matrix = confusion_matrix(Y, Y_pred)
    # print(cf_matrix)
    sns.heatmap(cf_matrix, annot=True, fmt='g',
                xticklabels=labels, yticklabels=labels)
    plt.savefig(path_plot)
    plt.close()


def main():
    pass


if __name__ == "__main__":
    main()
