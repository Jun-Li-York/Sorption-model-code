import os

import numpy as np
import pandas as pd
from knnimpute import knn_impute_few_observed
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm


def df_to_ndarray(input_df, with_soil, target_column):
    target_data = input_df[target_column]
    structure_data = input_df[input_df.columns[input_df.columns.str.startswith("structure")]]
    column_names = list(structure_data.columns)
    x = structure_data.values
    y = target_data.values
    if with_soil:
        soil_data = input_df[input_df.columns[input_df.columns.str.startswith("soil")]]
        column_names.extend(list(soil_data.columns))
        additonal_features = soil_data.values
        x = np.concatenate([x, additonal_features], axis=1)
    x = knn_impute_few_observed(x, np.isnan(x), k=3)
    return x, y, column_names


def k_fold_cross_validation(model_cls, params, x, y, k_fold):
    # The k-fold cv method to split
    split_method = KFold(n_splits=k_fold, shuffle=True, random_state=np.random.RandomState(1))
    split_indices = list(split_method.split(y))
    mse_list = []
    r2_list = []
    for train_indices, test_indices in tqdm(split_indices):
        train_x = x[train_indices]
        train_y = y[train_indices]
        test_x = x[test_indices]
        test_y = y[test_indices]
        model = model_cls(**params)
        model.fit(train_x, train_y)
        mse = mean_squared_error(test_y, model.predict(test_x))
        r2 = r2_score(test_y, model.predict(test_x))
        mse_list.append(mse)
        r2_list.append(r2)
    mean_mse = np.mean(mse_list)
    mean_r2 = np.mean(r2_list)
    return mean_mse, mean_r2


def load_df_from_file_path(data_folder, data_file_name):
    data_file_path = os.path.join(data_folder, data_file_name)
    data = pd.read_csv(data_file_path)
    return data
