import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_variable_pairs(df):
    '''Takes a dataframe and produces a pairplot of all features.
    
    Recommended to use .sample() method for large dataframes.'''
    sns.pairplot(df, corner=True)

def plot_categorical_and_continuous_vars(df, cat, cont):
    '''Accepts dataframe, one categorical column, and one numerical column, and produces a boxplot, swarmplot, and barplot
    for those two variables.
    '''
    sns.boxplot(data=df, x=df[cat], y=df[cont])
    sns.swarmplot(data=df, x=df[cat], y=df[cont])
    sns.barplot(data=df, x=df[cat], y=df[cont])