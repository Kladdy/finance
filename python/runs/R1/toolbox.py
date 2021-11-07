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
    assert dataset in ["training", "validation"]
    logger.INFO(f"Loading {dataset} dataset...")

    npz_file = f"{collector}_period{period}_interval{interval}_start{start}_end{stop}_datalength{data_length}.npz"
    npz_filepath = f"{data_folder_name}/{npz_file}"
    with np.load(npz_filepath) as npz:
        data = npz[f'{dataset}_data']
        labels = npz[f'{dataset}_labels']

    return data, labels

def load_training_data(collector, period, interval, start, stop, data_length):
    return load_data(collector, period, interval, start, stop, data_length, "training")

def load_validation_data(collector, period, interval, start, stop, data_length):
    return load_data(collector, period, interval, start, stop, data_length, "validation")