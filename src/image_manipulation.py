import os
import cv2


def crop_images_dataset(path_original_folder, path_cropped_folder, extension_image, crop_y, crop_x):
    """
    It reads a folder with images and creates a new folder with inside the cropped version 
    of the images.

    Args:
        - path_original_folder (str): path of the original folder that contains the images
        that we want to crop
        - path_cropped_folder (str): path of the folder were we want to write the cropped images
        - extension_image (str): extension of the image (".jpg", "jpeg", ".png", ....)
        - crop_x (slice): represents the cropped area (the pixels we want to keep) on the x axis
        - crop_y (slice): represents the cropped area (the pixels we want to keep) on the y axis

    Returns: None

    """
    if not os.path.isdir(path_cropped_folder):
        os.mkdir(path_cropped_folder)
    for file in os.listdir(path_original_folder):
        if file.endswith(extension_image):
            img = cv2.imread(os.path.join(path_original_folder, file))
            cropped_image = img[crop_y, crop_x]
            cv2.imwrite(os.path.join(path_cropped_folder, file), cropped_image)
