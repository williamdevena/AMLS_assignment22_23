import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

'''

This file contains all the functions used to visualize/plot data

'''

def histogram_df(df):
    '''
    Plots a histogram for every column in the dataframe
    
    Args:
        - df: Dataframe
        
    Returns: None
    
    '''
    num_keys = len(df.keys())
    figure, axis = plt.subplots(num_keys)
    
    for (key, index) in zip(df.keys(), range(num_keys)):
        axis[index].hist(df[key], edgecolor="black")
        axis[index].set_title(key)
        
    plt.show()
    
        

def boxplot(data):
    pass


def main():
    d = {'col1': [1, 2, 3, 4, 9], 'col2': [5, 6, 7, 8, 9], 'col3': ['a', 'b', 'c', 'd', 'd'], 'col4': [5, 6, 7, 8, 7]}
    df = pd.DataFrame(data=d)
    
    histogram_df(df,1,1)
    
    
if __name__=="__main__":
    main()
    

