import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torchvision import transforms

from assignment_dataset import AssignmentDataset
from src.data_loading import (  # load_flatten_images_from_folder,
    load_ds_labels_from_csv,
    load_images_from_folder,
    load_X_Y_train_test,
    retrieve_img_and_labels_paths,
)


class PytorchDataset(Dataset):
    def __init__(self, dataset_name, label_name, train_or_test, validation_split, transform=None):
        self.dataset_object = AssignmentDataset(
            name=dataset_name, label=label_name)
        self.image_dimensions = AssignmentDataset.possible_combinations[dataset_name][1]
        self.train_or_test = train_or_test
        self.transform = transform
        self.validation_split = validation_split

        path_train_img, path_train_labels, path_test_img, path_test_labels, use_dominant_color_dataset = retrieve_img_and_labels_paths(
            dataset_object=self.dataset_object
        )

        if self.train_or_test == "train":
            self.X = load_images_from_folder(
                path_train_img, self.dataset_object.image_dimensions)
            self.Y = load_ds_labels_from_csv(path_train_labels)
            self.Y = self.Y[self.dataset_object.label]
            self.X = self.X[:self.validation_split]
            self.Y = self.Y[:self.validation_split]
        elif self.train_or_test == "validation":
            self.X = load_images_from_folder(
                path_train_img, self.dataset_object.image_dimensions)
            self.Y = load_ds_labels_from_csv(path_train_labels)
            self.Y = self.Y[self.dataset_object.label]
            self.X = self.X[self.validation_split:]
            self.Y = self.Y[self.validation_split:]
        elif self.train_or_test == "test":
            self.X = load_images_from_folder(
                path_test_img, self.dataset_object.image_dimensions)
            self.Y = load_ds_labels_from_csv(path_test_labels)
            self.Y = self.Y[self.dataset_object.label]

        # SCALING
        #self.X = self.X / 255

        # Changing the labels from -1,+1 to 0,+1
        # Note: in the case of multiclass classification
        # this line is not going to change anything
        # (because in the classes you don't have -1)
        self.Y = np.where(self.Y == -1, 0, self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.Y[idx]

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
