import sys
sys.path.append('../../')
import os
import log
import pandas as pd
import numpy as np
from constants import data_folder_name

logger = log.Log("DEBUG")

def mkdir(folder_name):
    """Makes sure a folder exists, otherwise creates it"""

    if not os.path.exists(folder_name):
        logger.INFO(f"Folder did not exist, creating {folder_name}...")
        os.makedirs(folder_name)

def load_data(collector, period, interval, start, stop, data_length, dataset):
    assert dataset in ["training", "validation", "testing"]
    logger.INFO(f"Loading {dataset} dataset...")

    npz_file = f"{collector}_period{period}_inteval{interval}_start{start}_end{stop}_datalength{data_length}.npz"
    npz_filepath = f"{data_folder_name}/{npz_file}"
    with np.load(npz_filepath) as npz:
        data = npz[f'{dataset}_data']
        labels = npz[f'{dataset}_labels']

    # Remove any nans
    nan_idx = [idx[0] for idx in np.argwhere(np.isnan(data))]
    data = np.delete(data, nan_idx, axis=0)
    labels = np.delete(labels, nan_idx)
    logger.INFO(f"Removed {len(nan_idx)} nan values of of {labels.shape} in total")

    return data, labels

def load_training_data(collector, period, interval, start, stop, data_length):
    return load_data(collector, period, interval, start, stop, data_length, "training")

def load_validation_data(collector, period, interval, start, stop, data_length):
    return load_data(collector, period, interval, start, stop, data_length, "validation")

def load_testing_data(collector, period, interval, start, stop, data_length):
    return load_data(collector, period, interval, start, stop, data_length, "testing")