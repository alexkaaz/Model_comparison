import pandas as pd
from typing import Tuple
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

def imbalance_level(dataset, target_col: str='') -> None: 
    '''
    Show ratio between between two classes
    
    :param dataset: Dataset ypu work with
    :param target_col: Your target column
    '''
    total_data = len(dataset)
    if isinstance(dataset ,pd.DataFrame):
        counted = dataset[target_col].value_counts(sort=True)
    else:
        counted = dataset.value_counts(sort=True)
    print(counted)
    if len(counted) > 2:
        raise ValueError('Cant handle non binary clases') 
    # calculate percent of each class
    fclass = int(round(counted.iloc[0]/total_data, 2) * 100)
    sclass = int(round(counted.iloc[1]/total_data, 2) * 100)
    print(f'class ratio {fclass}/{sclass}')
    

def undersampling(X: pd.DataFrame, y: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Delete majority class element 
    
    :param dataset: Dataset you work with
    :param target_col: Your target column
    '''
    counted = y.value_counts()

    # ValueError if target column is multiclass
    if len(counted) > 2:
        raise ValueError('Cant handle non binary clases')

    # Standardizing features
    print(f'before StandartScaler: {X.shape}')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f'after StandartScaler: {X_scaled.shape}')

    # undersampoling with using xentroids method
    claster_sampler = RandomUnderSampler(random_state=69)
    X_resampled, y_resampled = claster_sampler.fit_resample(X_scaled, y)

    # print new ratio
    imbalance_level(y_resampled, target_col)

    return X_resampled, y_resampled

def oversampling(X: pd.DataFrame, y: pd.DataFrame, k: int=5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Creates new element to achieve 50/50 ratio
    
    :param dataset: Dataset you work with
    :param target_col: Your target column
    :param k: k_neighbors for SMOTE
    '''
    # ValueError if target column is multiclass
    counted = y.value_counts(sort=True)
    if len(counted) > 2:
        raise ValueError('Cant handle non binary clases') 

    # Standardizing features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # initializes SMOTE with sampling_strategy='auto',
    # wich mean that classes will be balanced to 50/50 ratio
    smote = SMOTE(
        sampling_strategy='auto',
        k_neighbors=k,
        random_state=69
    )
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # print new ration
    imbalance_level(y_resampled)

    return X_resampled, y_resampled