import argparse
from os import close
from toolbox import logger, mkdir
from constants import run_name, collector_folder_path, training_split, validation_split, testing_split, data_folder_name
import csv
import numpy as np
import pandas as pd
import hashlib
from math import floor

# Parse arguments
parser = argparse.ArgumentParser(description='Prepare data for R1.')
parser.add_argument('collector', type=str, help='the collector, ie C1, C2...')
parser.add_argument('period', type=str, help='the period, ie 1mo, 1y...')
parser.add_argument('interval', type=str, help='the interval, ie 1m, 30m...')
parser.add_argument('start', type=str, help='the start date, ie 2021.04.29...')
parser.add_argument('stop', type=str, help='the stop date, ie 2021.05.28...')
parser.add_argument('data_length', type=int, help='the amount of samples to have in the trace, ie 20')

args = parser.parse_args()
collector = args.collector
period = args.period
interval = args.interval
start = args.start
stop = args.stop
data_length = args.data_length

logger.INFO(f"Preparing data for {run_name}...")

mkdir(data_folder_name)

# Define constants
collected_data_file = f"{collector}_period{period}_inteval{interval}_start{start}_end{stop}.pkl"
collected_data_filepath = f"{collector_folder_path}/{collector}/saved_data/{collected_data_file}"

# Load collected .pkl data
df = pd.read_pickle(collected_data_filepath)

# Get the tickers
column_amount = df.columns.get_level_values(1).values.size
ticker_amount = column_amount // 6 # 6 columns are created from yfinance

ticker_symbols = df.columns.get_level_values(1).values[:ticker_amount]

# Get the sample amount
sample_amount = df.index.get_level_values(0).values.size
logger.INFO(f"Sample amount: {sample_amount}")

# Calculate the amount of data that will be produced
# As sample_length samples are needed, the first few
# days will not have any historical data.
# Also, for R2, we subtract one additional due to needing to use one more to get absolute value
data_amount_per_ticker = sample_amount - data_length - 1

# Create numpy arrays for data and labels
closing_values = np.zeros((ticker_amount, sample_amount))

for i, ticker in enumerate(ticker_symbols):
    closing_values_tmp = df["Close"][ticker].values
    closing_values[i,:] = closing_values_tmp

# Extract data with data_length lenghts and labels
data_amount = data_amount_per_ticker * ticker_amount
data = np.zeros((data_amount, data_length))
labels = np.zeros((data_amount))

# For R2, we need to start at data_length + 1 as we need one extra to take relative difference
data_indicies = np.arange(data_length + 1, sample_amount)

idx = 0
for j, index in enumerate(data_indicies):
    for i, ticker in enumerate(ticker_symbols):
        # For R2, we need to load one more data. This is used to calculate relative difference (hence data_length + 1)
        data_tmp = closing_values[i, (index - (data_length + 1)):index]
        label_tmp = closing_values[i, index] # Get the j:th value, which is the closing value of interest

        
        # For R2, we take the relative difference between the values
        label_tmp_rel = (label_tmp - data_tmp[-1]) / data_tmp[-1]
        data_tmp_rel = np.array([(data_tmp[i] - data_tmp[i-1]) / data_tmp[i-1] for i in range(1, data_length+1)])

        data[idx, :] = data_tmp_rel
        labels[idx] = label_tmp_rel

        idx += 1 
        
# Create a shuffled index array based on the collected data file name
shuffeled_indicies = np.arange(data_amount)
hash_string = int(hashlib.sha256(collected_data_file.encode('utf-8')).hexdigest(), 16) % 10**9 # Compute hash
np.random.seed(hash_string) # Set the random seed to the .pkl file name
np.random.shuffle(shuffeled_indicies) # Shuffle the indicies

# Split into training, validation and testing data
training_indicies_start = 0
training_indicies_end = floor(training_split * data_amount)
validation_indicies_start = training_indicies_end
validation_indicies_end = validation_indicies_start + floor(validation_split * data_amount)
testing_indicies_start = validation_indicies_end
testing_indicies_end = data_amount

training_indicies = shuffeled_indicies[training_indicies_start:training_indicies_end]
validation_indicies = shuffeled_indicies[validation_indicies_start:validation_indicies_end]
testing_indicies = shuffeled_indicies[testing_indicies_start:testing_indicies_end]

# Make sure all data is utilized
assert training_indicies.size + validation_indicies.size + testing_indicies.size == data_amount

training_data = data[training_indicies, :]
training_labels = labels[training_indicies]
validation_data = data[validation_indicies, :]
validation_labels = labels[validation_indicies]
testing_data = data[testing_indicies, :]
testing_labels = labels[testing_indicies]

# # Save data as npz
npz_file = f"{collector}_period{period}_inteval{interval}_start{start}_end{stop}_datalength{data_length}.npz"
npz_filepath = f"{data_folder_name}/{npz_file}"
np.savez(npz_filepath, training_data=training_data, training_labels=training_labels, 
            validation_data=validation_data, validation_labels=validation_labels,
            testing_data=testing_data, testing_labels=testing_labels)