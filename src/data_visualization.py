'''
This file contains all the functions used to visualize/plot data

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys, os

sys.path.append('./src/')

from costants import *

def histogram_df(df):
    '''
    Plots a histogram for every column in a dataframe
    
    Args:
        - df: Dataframe
        
    Returns: 
        - None
    
    '''
    num_keys = len(df.keys())
    figure, axis = plt.subplots(num_keys)
    for (key, index) in zip(df.keys(), range(num_keys)):
        axis[index].hist(df[key], edgecolor="black")
        axis[index].set_title(key)
    plt.show()
    
def boxplot_df(data):
    pass

def visualize_hist_distribution_csv(path_csv):
    '''
    Reads a csv and plots a histogram from every variable in the csv
    
    Args:
        - path_csv (str): path of the csv file
        
    Returns: 
        - None
    
    '''
    df = pd.read_csv(path_csv, sep=SEPARATOR)
    # .iloc[:, 2:] because we pass the entire dataframe except for the index col and the first col that is the name of the image (img_name)
    histogram_df(df.iloc[:, 2:])     


def main():
    visualize_hist_distribution_csv(PATH_CARTOON_TRAIN_LABELS)
    visualize_hist_distribution_csv(PATH_CELEBA_TRAIN_LABELS)
    
if __name__=="__main__":
    main()
    

