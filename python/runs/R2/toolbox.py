import sys
sys.path.append('../../')
import os
import log
import pandas as pd
import numpy as np
from constants import data_folder_name, model_folder_name, results_folder_name

logger = log.Log("DEBUG")

class Quantile:
    def __init__(self, quantile, sum_true, sum_predicted):
        self.quantile = quantile
        self.sum_true = sum_true
        self.sum_predicted = sum_predicted

def mkdir(folder_name):
    """Makes sure a folder exists, otherwise creates it"""

    if not os.path.exists(folder_name):
        logger.INFO(f"Folder did not exist, creating {folder_name}...")
        os.makedirs(folder_name)

def get_model_filepath(run_id):
    model_filename = f"model_{run_id}.h5"
    model_filepath = f"{model_folder_name}/{model_filename}"
    return model_filepath

def get_results_filepath(run_id):
    results_filename = f"results_{run_id}"
    results_filepath = f"{results_folder_name}/{results_filename}"
    return results_filepath
    

def load_data(collector, period, interval, start, stop, data_length, dataset):
    assert dataset in ["training", "validation", "testing"]
    logger.INFO(f"Loading {dataset} dataset...")

    npz_file = f"{collector}_period{period}_interval{interval}_start{start}_end{stop}_datalength{data_length}.npz"
    npz_filepath = f"{data_folder_name}/{npz_file}"
    with np.load(npz_filepath) as npz:
        data = npz[f'{dataset}_data']
        labels = npz[f'{dataset}_labels']

        if dataset == 'testing':
            testing_tickers = npz['testing_tickers']
            testing_last_data_values = npz['testing_last_data_values']

    # Remove any nans
    amount_of_data = len(labels)
    nan_idx = [idx[0] for idx in np.argwhere(np.isnan(data))]
    data = np.delete(data, nan_idx, axis=0)
    labels = np.delete(labels, nan_idx)

    logger.INFO(f"Removed {len(nan_idx)} nan values of of {amount_of_data} in total")

    if dataset == 'testing':
        testing_tickers = np.delete(testing_tickers, nan_idx)
        testing_last_data_values = np.delete(testing_last_data_values, nan_idx)
        
        return data, labels, testing_tickers, testing_last_data_values
    else:
        return data, labels

def load_quantile_data(run_id):
    npz_filepath = f'{get_results_filepath(run_id)}.npz'

    with np.load(npz_filepath, allow_pickle=True) as npz:
        quantile_sums = npz['quantile_sums']

    return quantile_sums
        

def load_training_data(collector, period, interval, start, stop, data_length):
    return load_data(collector, period, interval, start, stop, data_length, "training")

def load_validation_data(collector, period, interval, start, stop, data_length):
    return load_data(collector, period, interval, start, stop, data_length, "validation")

def load_testing_data(collector, period, interval, start, stop, data_length):
    return load_data(collector, period, interval, start, stop, data_length, "testing")