import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_residuals(y, yhat):
    '''Plots the residuals for a given target variable against the target variable.'''

    sns.scatterplot(x=y, y=(yhat - y))
    plt.title('Residuals for Target Variable')
    plt.xlabel(f'Target Variable')
    plt.ylabel('Residual Values')
    plt.show()

def regression_errors(y, yhat):
    '''Takes in an actual value and a predicted value and outputs summary of regression errors.
    
    Regression errors included: SSE, ESS, TSS, MSE, RMSE.'''

    #SSE for model

    SSE_model = ((yhat - y) ** 2).sum()
    print(f'SSE for model = {SSE_model}')

    #ESS for model

    ESS_model = sum((yhat - y.mean())**2)
    print(f'ESS for model = {ESS_model}')

    #TSS for model

    TSS_model = ESS_model + SSE_model
    print(f'TSS for model = {TSS_model}')

    #MSE for model

    MSE_model = SSE_model / len(y)
    print(f'MSE for model = {MSE_model}')

    #RMSE for model

    RMSE_model = MSE_model ** 0.5
    print(f'RMSE for model = {RMSE_model}')

def baseline_mean_errors(y):
    '''Computes the SSE, MSE, and RMSE for the baseline model of a given target variable.'''

    #Set the baseline
    baseline = y.mean()

    #SSE for baseline

    SSE_baseline = ((np.full(len(y), baseline) - y) ** 2).sum()
    print(f'SSE for baseline = {SSE_baseline}')

    #MSE for baseline

    MSE_baseline = SSE_baseline / len(y)
    print(f'MSE for baseline = {MSE_baseline}')

    #RMSE for baseline

    RMSE_baseline = MSE_baseline ** 0.5
    print(f'RMSE for baseline = {RMSE_baseline}')

def better_than_baseline(y, yhat):
    '''Returns SSE values for actual vs. predicted, states whether the predictive model performed better, and returns True if so.'''

    #Set the baseline
    baseline = y.mean()

    SSE_model = ((yhat - y) ** 2).sum()
    SSE_baseline = ((np.full(len(y), baseline) - y) ** 2).sum()

    print(f'The SSE for the model is: {SSE_model}')
    print(f'The SSE for the baseline is: {SSE_baseline}')

    if SSE_model < SSE_baseline:
        print('The model outperforms the baseline.')
        return True
    else:
        print('The model did not perform better than the baseline.')
        return False