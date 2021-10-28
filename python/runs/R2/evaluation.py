from toolbox import logger, load_testing_data, mkdir
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
from constants import run_name, results_folder_name, model_folder_name

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluate model')
parser.add_argument('collector', type=str, help='the collector, ie C1, C2...')
parser.add_argument('period', type=str, help='the period, ie 1mo, 1y...')
parser.add_argument('interval', type=str, help='the interval, ie 1m, 30m...')
parser.add_argument('start', type=str, help='the start date, ie 2021.04.29...')
parser.add_argument('stop', type=str, help='the stop date, ie 2021.05.28...')
parser.add_argument('data_length', type=int, help='the amount of samples to have in the trace, ie 20')
parser.add_argument('batch_size', type=int, help='the batch size, ie 64')

args = parser.parse_args()
collector = args.collector
period = args.period
interval = args.interval
start = args.start
stop = args.stop
data_length = args.data_length
batch_size = args.batch_size

logger.INFO(f"Evaluating {run_name}...")

mkdir(results_folder_name)

# Load testing data and convert to dataset tensor
testing_data, testing_labels = load_testing_data(collector, period, interval, start, stop, data_length)
testing_dataset = tf.data.Dataset.from_tensor_slices((testing_data, testing_labels))

# Construct model file path
model_filename = f"model_{run_name}_{collector}_period{period}_inteval{interval}_start{start}_end{stop}_datalength{data_length}_batchsize{batch_size}.h5"
model_filepath = f"{model_folder_name}/{model_filename}"

# Load the model
model = load_model(model_filepath)

# Make new predictions
predictions = model.predict(testing_data)
predictions = np.squeeze(predictions) # Removes the axis of length 1

print("testing labels: ", testing_labels)
print("predictions: ", predictions)
print("Difference: ", testing_labels - predictions)