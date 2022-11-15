import logging

import src.data_preparation as data_preparation
import src.data_visualization as data_visualization
import baselines.knn as knn
import baselines.mlp as mlp

logging.basicConfig(format="%(message)s", level=logging.INFO)


def main():
    '''
    DATA PREPARATION
    '''
    # data_preparation.data_preparation()

    '''
    DATA VISUALIZATION
    '''
    # data_visualization.data_visualization()

    '''
    KNN CROSS VALIDATION
    '''
    # labels = "eye_color"
    # # k_fold_param = 5
    # k_fold_params = range(5, 11)
    # # k-fold cross validation
    # array_possible_hyperparameter = range(10, 101, 10)
    # k_fold_params = range(5, 7)
    # knn_plots_and_log_directory = "../plots_and_logs/test"
    # knn.knn_k_fold_cross_validation_with_multiple_k(
    #     k_fold_params,
    #     array_possible_hyperparameter,
    #     labels,
    #     knn_plots_and_log_directory,
    # )

    '''
    MLP
    '''
    mlp.main()


if __name__ == "__main__":
    main()
