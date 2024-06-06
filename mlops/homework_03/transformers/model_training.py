from typing import Tuple, Dict

import pandas as pd

from mlops.utils.data_preparation.cleaning import clean
from mlops.utils.data_preparation.feature_engineering import combine_features
from mlops.utils.data_preparation.feature_selector import select_features
from mlops.utils.data_preparation.splitters import split_on_value

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


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
) -> Tuple[Dict, pd.DataFrame]:
    
    train_dict, y = pre_process_data(df)

    return train_dict, y