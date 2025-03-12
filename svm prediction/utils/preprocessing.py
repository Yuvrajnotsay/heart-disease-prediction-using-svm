import pandas as pd 
from typing import Union
columns = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'cp_0', 'cp_1', 'cp_2', 'cp_3', 'restecg_0',
    'restecg_1', 'restecg_2', 'thal_0', 'thal_1', 'thal_2', 'thal_3']
def preprocess_data(data: Union[pd.DataFrame, dict]):
    """
    Place data cleaning and feature engineering here.

    Args:
        data (dict): A dictionary representing a single patient's data. 

    Returns:
        numpy.ndarray: Processed data as a NumPy array. 
    """
    if isinstance(data, dict):
        data = pd.DataFrame(data, index=[0])
    categorical_features = ['cp', 'restecg', 'thal']
    df = pd.get_dummies(data, columns=categorical_features)
    print(df.keys())
    for column in columns:
        if column not in df.columns:
            df[column] = 0
    df = df[columns]
    return df

    