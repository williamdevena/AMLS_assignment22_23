import logging
from sklearn.decomposition import PCA
from lazypredict.Supervised import LazyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt

from src import (
    data_loading,
    data_preparation,
    data_visualization,
    costants,
    image_manipulation
)

#import baselines.knn as knn
#import baselines.mlp as mlp
from baselines import knn

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
    DATA LOADING
    '''
    # dataset = "cropped_eyes_celeba"
    # label = "smiling"
    # width = 98
    # height = 38
    # image_dimensions = (width, height)
    # X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled = data_loading.load_X_Y_train_test(
    #     dataset, label, image_dimensions, scaling=True
    # )
    # X_train, Y_train, X_test, Y_test = data_loading.load_X_Y_train_test(
    #     dataset, label, image_dimensions, scaling=False
    # )
    # print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    # print(X_train_scaled.shape, Y_train_scaled.shape,
    #       X_test_scaled.shape, Y_test_scaled.shape)

    # print(X_train[0], X_train_scaled[0])

    """
    KNN
    """
    # knn.knn_for_every_dataset()

    '''
    KNN CROSS VALIDATION
    '''
    k_fold_params = range(5, 10)
    array_possible_hyperparameter = range(10, 101, 10)
    knn.knn_k_fold_cross_validation_for_every_dataset(
        k_fold_params, array_possible_hyperparameter
    )

    '''
    PCA
    '''
    # pca = PCA()

    # dataset = "celeba"
    # label = "smiling"
    # width = 178
    # height = 218
    # image_dimensions = (width, height)
    # X_train, Y_train, X_test, Y_test = data_loading.load_X_Y_train_test(
    #     dataset, label, image_dimensions
    # )
    # pca.fit(X_train)
    # print(pca.explained_variance_ratio_)
    # print(pca.singular_values_)

    '''
    CROPPING DATASET FOR 'SMILING' or for 'EYE_COLOR'
    '''
    # path_original_folder = costants.PATH_CARTOON_TEST_IMG
    # path_cropped_folder = '../Datasets/cartoon_set_test/img_eye'
    # crop_y = slice(248, 277)
    # crop_x = slice(283, 308)
    # extension_image = ".png"
    # image_manipulation.crop_images_dataset(
    #     path_original_folder, path_cropped_folder, extension_image, crop_y, crop_x)

    '''
    LAZY PREDICT
    '''
    # dataset = "cropped_eyes_celeba"
    # label = "smiling"
    # width = 98
    # height = 38
    # image_dimensions = (width, height)
    # X_train, Y_train, X_test, Y_test = data_loading.load_X_Y_train_test(
    #     dataset, label, image_dimensions
    # )
    # clf = LazyClassifier(verbose=1, ignore_warnings=True, custom_metric=None)

    # # X_train = X_train[:20]
    # # Y_train = Y_train[:20]
    # # X_test = X_test[:10]
    # # Y_test = Y_test[:10]
    # models, predictions = clf.fit(X_train, X_test, Y_train, Y_test)
    # print(models)

    """
    COUNT PUPIL COLORS
    """
    # data_preparation.count_pupil_colors(
    #     costants.PATH_CARTOON_TRAIN_IMG, pixel_y=260, pixel_x=294)

    # path = costants.PATH_CARTOON_TRAIN_CROPPED_EYE_IMG
    # dict_colors = data_preparation.count_pupil_colors_colorthief(path)
    # X = np.array([np.array(color) for color in dict_colors])
    # print(X.shape)
    # kmeans = KMeans(n_clusters=6)
    # kmeans.fit(X)

    # # Getting the cluster labels
    # labels = kmeans.predict(X)
    # centroids = kmeans.cluster_centers_
    # array_centroids = [(int(centroid[0]), int(centroid[1]), int(centroid[2]))
    #                    for centroid in centroids]
    # print(array_centroids)
    # labels = kmeans.predict(X)

    # fig = plt.figure(figsize=(20, 10))
    # ax = fig.add_subplot(111, projection='3d')

    # c0 = np.array(labels == 0)
    # c1 = np.array(labels == 1)
    # c2 = np.array(labels == 2)
    # c3 = np.array(labels == 3)
    # c4 = np.array(labels == 4)
    # c5 = np.array(labels == 5)

    # ax.scatter(X[c0][:, 0], X[c0][:, 1], X[c0][:, 2])
    # ax.scatter(X[c1][:, 0], X[c1][:, 1], X[c1][:, 2])
    # ax.scatter(X[c2][:, 0], X[c2][:, 1], X[c2][:, 2])
    # ax.scatter(X[c3][:, 0], X[c3][:, 1], X[c3][:, 2])
    # ax.scatter(X[c4][:, 0], X[c4][:, 1], X[c4][:, 2])
    # ax.scatter(X[c5][:, 0], X[c5][:, 1], X[c5][:, 2])
    # ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
    #            marker='x', s=169, linewidths=10,
    #            color='black', zorder=50)
    # ax.set_xlabel('R')
    # ax.set_ylabel('G')
    # ax.set_zlabel('B')
    # plt.show()


if __name__ == "__main__":
    main()
