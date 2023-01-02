

from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.data_loading import (
    load_X_Y_train_test,
    retrieve_img_and_labels_paths,
    # load_flatten_images_from_folder,
    load_images_from_folder,
    load_ds_labels_from_csv
)

from assignment_dataset import AssignmentDataset


class PytorchDataset(Dataset):
    def __init__(self, dataset_name, label_name, train_or_test):
        self.dataset_object = AssignmentDataset(
            name=dataset_name, label=label_name)
        self.image_dimensions = AssignmentDataset.possible_combinations[dataset_name][1]
        self.train_or_test = train_or_test

        path_train_img, path_train_labels, path_test_img, path_test_labels, use_dominant_color_dataset = retrieve_img_and_labels_paths(
            dataset_object=self.dataset_object)

        # if self.train_or_test == "train":
        #     self.X = load_flatten_images_from_folder(
        #         path_train_img, self.dataset_object.image_dimensions)
        #     self.Y = load_ds_labels_from_csv(path_train_labels)
        #     self.Y = self.Y[self.dataset_object.label]
        # elif self.train_or_test == "test":
        #     self.X = load_flatten_images_from_folder(
        #         path_test_img, self.dataset_object.image_dimensions)
        #     self.Y = load_ds_labels_from_csv(path_test_labels)
        #     self.Y = self.Y[self.dataset_object.label]

        if self.train_or_test == "train":
            self.X = load_images_from_folder(
                path_train_img, self.dataset_object.image_dimensions)
            self.Y = load_ds_labels_from_csv(path_train_labels)
            self.Y = self.Y[self.dataset_object.label]
        elif self.train_or_test == "test":
            self.X = load_images_from_folder(
                path_test_img, self.dataset_object.image_dimensions)
            self.Y = load_ds_labels_from_csv(path_test_labels)
            self.Y = self.Y[self.dataset_object.label]

        # SCALING
        self.X = self.X / 255

        # Changing the labels from -1,+1 to 0,+1
        # Note: in the case of multiclass classification
        # this line is not going to change anything
        # (because in the classes you don't have -1)
        self.Y = np.where(self.Y == -1, 0, self.Y)

        self.X = torch.Tensor(self.X)
        self.Y = torch.Tensor(self.Y).long()
        #self.Y = torch.unsqueeze(self.Y, dim=-1)

        # PERMUTING DIMENSIONS
        self.X = torch.permute(self.X, (0, 3, 1, 2))

        print(self.X.shape, self.Y.shape)
        # print(self.X[0][:1][:1])
        # print(self.Y[:30])

        self.X = self.X[:16]
        self.Y = self.Y[:16]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.Y[idx]

        return image, label
