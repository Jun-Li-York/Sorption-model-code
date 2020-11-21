import os
import random

import mxnet as mx
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from analyser.tpe_optimisation import TPEOptimiser
from analyser.utils import k_fold_cross_validation, df_to_ndarray, load_df_from_file_path


def start_analysis(
        data_folder,
        data_file_name,
        test_data_without_soil_file_name,
        test_data_with_soil_file_name,
        target_column,
        with_soil,
        num_rounds,
        tuning_k_fold,
        final_k_fold,
):
    data = load_df_from_file_path(data_folder, data_file_name)

    # Initialise the TPE optimiser
    print("Start tuning parameters with {}-fold cross validation...".format(tuning_k_fold))
    tpe_optimiser = TPEOptimiser(target_column, with_soil=with_soil)
    train_x, train_y, column_names = df_to_ndarray(data, with_soil, target_column)
    best_model_cls, best_params = tpe_optimiser.optimise(train_x, train_y, num_rounds=num_rounds, k_fold=tuning_k_fold)

    print("Start doing final {}-fold cross validation...".format(final_k_fold))
    mean_mse, mean_r2 = k_fold_cross_validation(best_model_cls, best_params, train_x, train_y, final_k_fold)
    print("The mean mse of final {}-fold cross validation is: {}".format(final_k_fold, mean_mse))
    print("The mean R2 of final {}-fold cross validation is: {}".format(final_k_fold, mean_r2))

    if test_data_without_soil_file_name is None and test_data_with_soil_file_name is None:
        return

    # This is only to use when test data file is given
    print("Start predictions with testing data set...")
    train_x, train_y, column_names = df_to_ndarray(data, with_soil, target_column)
    if with_soil and test_data_with_soil_file_name is not None:
        test_data = load_df_from_file_path(data_folder, test_data_with_soil_file_name)
    elif not with_soil and test_data_without_soil_file_name:
        test_data = load_df_from_file_path(data_folder, test_data_without_soil_file_name)
    else:
        raise ValueError(
            "With soil is {} while test_data_with_soil_file_name is {}, test_data_without_soil_file_name is {}".format(
                with_soil, test_data_with_soil_file_name, test_data_without_soil_file_name
            ))
    test_x, test_y, _ = df_to_ndarray(test_data, with_soil, target_column)
    best_model = best_model_cls(**best_params)
    best_model.fit(train_x, train_y)

    try:
        feature_importances = np.array([tree.feature_importances_ for tree in best_model.estimators_])
        feature_importance = np.mean(feature_importances, axis=0)
        feature_importance_sem = np.std(feature_importances, axis=0)/np.sqrt(np.shape(feature_importances)[1])
        feature_importance_df = pd.DataFrame.from_dict(
            {
                'feature_name': column_names,
                'feature_importance': feature_importance,
                'feature_sem':feature_importance_sem
            }
        )
        feature_importance_df = feature_importance_df.sort_values('feature_importance', ascending=False)
        pd.set_option('display.max_columns', None)
        print("The feature importance is {}".format(feature_importance_df))
    except AttributeError:
        pass

    # Results
    train_r2 = r2_score(train_y, best_model.predict(train_x))
    train_mse = mean_squared_error(train_y, best_model.predict(train_x))
    print("R2 on training dataset is {}, MSE on training dataset is {}".format(train_r2, train_mse))
    predictions = best_model.predict(test_x)
    test_r2 = r2_score(test_y, predictions)
    test_mse = mean_squared_error(test_y, predictions)
    print("R2 on testing dataset is {}, MSE on testing dataset is {}".format(test_r2, test_mse))

    # Write the prediction results
    result_df = pd.DataFrame.from_dict(
        {
            "True values": test_y,
            "Predictions": np.squeeze(predictions)
        }
    )
    print(result_df)
    result_df.to_csv(os.path.join(data_folder, './predictions.csv'))


if __name__ == '__main__':
    np.random.seed(1)
    mx.random.seed(1)
    random.seed(1)

    data_folder = "./data"
    data_file_name = "processed_data_training_ANN.csv"
    # Put the external file_name here to enable final testing on external data
    test_data_without_soil_file_name = "External_no_soil_ANN.csv"
    test_data_with_soil_file_name = "External_with_soil_ANN_final.csv"
    # target_column = "target_Kd"
    target_column = "target_LogKd"

    start_analysis(
        data_folder,
        data_file_name,
        test_data_without_soil_file_name,
        test_data_with_soil_file_name,
        target_column,
        with_soil=True,
        num_rounds=10,
        tuning_k_fold=10,
        final_k_fold=10
    )
