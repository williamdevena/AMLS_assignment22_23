'''
This file contains all the functions used to prepare/clean data

'''

import sys, os
import pandas as pd
import numpy as np
import logging
from pprint import pprint
from PIL import Image
import cv2

sys.path.append('./src/')

from costants import *

def check_values_from_csv(path_csv):
    '''
    Reads a csv and returns the np.unique of every variable 
    (useful to check if there are NaN values, empty values or other types of values we might want to check)
    
    Args: 
        - path_csv (str): path of the csv file
        
    Returns:
        - df.dtypes (pandas.core.series.Series): contains the types of all the variables
        - dict_counts (dict): dictionary of the form {variable : ([all values in the distribution], [counts of the values in the distribution])}
    '''
    df = pd.read_csv(path_csv, sep=SEPARATOR)
    df_only_variables = df.iloc[:, 2:]
    dict_counts = {}
    for key in df_only_variables.keys():
        counts = np.unique(df_only_variables[key], return_counts=True)
        dict_counts[key] = counts
        
    return df.dtypes, dict_counts

def check_shape_images(path_folder_images):
    '''
    Reads a folder of images and returns every type of shape present
    (useful to check what shape the image have and to check if they have all the same shape)
    
    Args: 
        - path_folder_images (str): path of the csv file
        
    Returns:
         - returns a tuple of the form ([shapes present in the folder], [counts of the shapes in the folder])
    '''
    images = os.listdir(path_folder_images)
    array_shapes = [cv2.imread(os.path.join(path_folder_images, image)).shape for image in images]
    
    return np.unique(array_shapes, axis=0, return_counts=True)

def reformat_csv_cartoon(original_path, new_path):
    '''
    Function used to reformat the csv of the cartoon dataset to match it with the celeba datase:
    The file_name column becomes img_name and it is the first column ( not the last as originally)
    
    Args: 
        - original_path (str): path of the original csv file
        - old_path (str): path of the new csv file
        
    Returns:
         - None
    '''
    df = pd.read_csv(original_path, sep=SEPARATOR)
    df = df.rename(columns={"file_name": "img_name"})
    columns_titles = ["img_name","eye_color", "face_shape"]
    df=df.reindex(columns=columns_titles)
    df.to_csv(new_path, sep=SEPARATOR)
    print(df)
    


def main():
    pprint(check_values_from_csv(PATH_CARTOON_TRAIN_LABELS))
    pprint(check_values_from_csv(PATH_CELEBA_TRAIN_LABELS))
    pprint(check_values_from_csv(PATH_CARTOON_TEST_LABELS))
    pprint(check_values_from_csv(PATH_CELEBA_TEST_LABELS))
    print(check_shape_images(PATH_CELEBA_TRAIN_IMG))
    print(check_shape_images(PATH_CELEBA_TEST_IMG))
    print(check_shape_images(PATH_CARTOON_TRAIN_IMG))
    print(check_shape_images(PATH_CARTOON_TEST_IMG))
    
    

if __name__=="__main__":
    main()
        
    

