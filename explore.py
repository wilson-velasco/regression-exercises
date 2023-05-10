import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_variable_pairs(df):
    '''Takes a dataframe and produces a pairplot of all features. Includes regression line.
    
    Recommended to use .sample() method for large dataframes.'''
    sns.pairplot(df, kind='reg', corner=True, plot_kws={'line_kws':{'color':'red'}})
    plt.show()

def plot_categorical_and_continuous_vars(df, cat, cont):
    '''Accepts dataframe, one categorical column, and one numerical column, and produces a boxplot, swarmplot, and barplot
    for those two variables.
    '''
    plt.figure(figsize=(13,4))
    plt.suptitle(f'Visualizations for {cat.capitalize()} vs. {cont.capitalize()}')
    plt.subplot(131)
    sns.boxplot(data=df, x=df[cat], y=df[cont]).axhline(df[cont].mean(), ls='--', c='cyan')
    plt.subplot(132)
    sns.violinplot(data=df, x=df[cat], y=df[cont]).axhline(df[cont].mean(), ls='--', c='cyan')
    plt.subplot(133)
    sns.barplot(data=df, x=df[cat], y=df[cont]).axhline(df[cont].mean(), ls='--', c='cyan')
    plt.show()
