from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator

import pandas as pd


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def pre_process_data(df):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    x_dict = df[categorical + numerical].to_dict(orient='records')
    
    target = 'duration'
    y = df[target].values
    
    return x_dict, y


@transformer
def transform(
    df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    
    train_dict, y_train = pre_process_data(df)
    #print ('DOne')
    #dv = DictVectorizer()
    #x_train = dv.fit_transform(train_dict)
    #lr = LinearRegression()
    #lr.fit(x_train, y_train)
    
    #print (lr._intercept)
    return train_dict