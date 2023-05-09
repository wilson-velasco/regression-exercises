import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from env import get_db_url

from sklearn.model_selection import train_test_split

import os

link = os.getcwd()+'/'

def wrangle_zillow():
    '''Retrieves the data from zillow database on CodeUp server.
    
    Returns bedroom, bathroom, squarefeet, tax, year, taxamount, and fips for single family residential houses.
    
    Cleans data to eliminate nulls and removes data outliers that are below 5th percentile and above 95th percentile.'''

    #Obtain filepath to connect to zillow db on CodeUp 
    z = get_db_url('zillow')

    # Checking to see if file exists in local directory. 

    if os.path.exists(link + 'zillow.csv'):
        zillow = pd.read_csv('zillow.csv')
    

    # Write to a local csv file if it doesn't exist. Includes query for requested data for Single Family Residential households.

    else:
        zillow = pd.read_sql('''SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips FROM properties_2017
	LEFT JOIN propertylandusetype USING (propertylandusetypeid)
    WHERE propertylandusedesc = 'Single Family Residential';''', z)
        zillow.to_csv('zillow.csv', index=False)
    

    #Rename columns
    zillow = zillow.rename(columns={'bedroomcnt': 'bedrooms'
                       ,'bathroomcnt': 'bathrooms'
                       ,'calculatedfinishedsquarefeet': 'sqft'
                       ,'taxvaluedollarcnt': 'value'
                       ,'fips': 'county'
                      })

    #Replace numerical values in county with their respective strings.

    zillow.county = zillow.county.replace([6037.0, 6059.0, 6111.0], ['LA', 'Orange', 'Ventura'])

    #Remove rows with NaNs from dataset
    zillow = zillow.dropna()

    #Handling outliers (we're keeping yearbuilt as is)
    zillow.bedrooms = zillow.bedrooms[zillow.bedrooms <= 10] #removing properties with more than 10 bedrooms
    zillow.bathrooms = zillow.bathrooms[zillow.bathrooms <=10] #removing properties with more than 10 bathrooms
    zillow.sqft = zillow.sqft[zillow.sqft < zillow.sqft.quantile(.99)] #removing top 1% of highest square footage
    zillow.value = zillow.value[zillow.value < zillow.value.quantile(.99)] #removing top 1% of highest value
    zillow.taxamount = zillow.taxamount[zillow.taxamount < zillow.taxamount.quantile(.99)] #removing top 1% of highest taxamount

    #Removing those newly-created nulls as well
    zillow = zillow.dropna()

    #Adjusting dtypes so dataframe is easier to read
    zillow[['bedrooms', 'sqft', 'yearbuilt', 'value', 'taxamount']] = zillow[['bedrooms', 'sqft', 'yearbuilt', 'value', 'taxamount']].astype(int)

    return zillow

def split_data(df):
    '''
    Takes in a DataFrame and returns train, validate, and test DataFrames; stratifies on target argument.
    
    Train, Validate, Test split is: 60%, 20%, 20% of input dataset, respectively.
    '''
    # First round of split (train+validate and test)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # Second round of split (train and validate)
    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       random_state=123)
    return train, validate, test