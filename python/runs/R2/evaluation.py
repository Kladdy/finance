from numpy.core.fromnumeric import size
from toolbox import logger, load_testing_data, mkdir
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
from constants import run_name, results_folder_name, model_folder_name
import matplotlib.pyplot as plt

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
testing_data, testing_labels, testing_tickers, testing_last_data_values = load_testing_data(collector, period, interval, start, stop, data_length)
# testing_data = np.expand_dims(testing_data, 2) if Conv1d
testing_dataset = tf.data.Dataset.from_tensor_slices((testing_data, testing_labels))

# Construct model file path
model_filename = f"model_{run_name}_{collector}_period{period}_inteval{interval}_start{start}_end{stop}_datalength{data_length}_batchsize{batch_size}.h5"
model_filepath = f"{model_folder_name}/{model_filename}"

# Construct results file path
results_filename = f"results_{run_name}_{collector}_period{period}_inteval{interval}_start{start}_end{stop}_datalength{data_length}_batchsize{batch_size}.png"
results_filepath = f"{results_folder_name}/{results_filename}"

# Load the model
model = load_model(model_filepath)

# Make new predictions
predictions = model.predict(testing_data)
predictions = np.squeeze(predictions) # Removes the axis of length 1

# Get true and predicted values
N = len(testing_labels)
predicted_values = np.zeros(N)
true_values = np.zeros(N)

# List for profit/loss if sample_prediction > 0
cost = []
profit_loss = []

for i in range(N):
    sample_data = testing_data[i, :]
    sample_label = testing_labels[i]
    sample_prediction = predictions[i]
    sample_ticker = testing_tickers[i]
    sample_last_data_value = testing_last_data_values[i]
    #logger.INFO(f'The ticker {sample_ticker.decode("utf-8")} had value {sample_last_data_value} as last data value')

    def get_data_absolute_value(rel_from_prev, prev):
        return rel_from_prev * prev + prev

    sample_true_absolute = get_data_absolute_value(sample_label, sample_last_data_value)
    sample_predicted_absolute = get_data_absolute_value(sample_prediction, sample_last_data_value)
    sample_diff_absolute = sample_predicted_absolute - sample_true_absolute

    #logger.INFO(f'predicted: {sample_predicted_absolute}, true: {sample_true_absolute}, diff: {sample_diff_absolute}')

    predicted_values[i] = sample_predicted_absolute
    true_values[i] = sample_true_absolute

    if sample_prediction > 0.001:
        cost.append(sample_last_data_value)
        profit_loss.append(sample_true_absolute - sample_last_data_value)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(predictions, testing_labels, '.')
axs[0].set_xlabel('predicted relative value')
axs[0].set_ylabel('true relative value')

coef = np.polyfit(predictions,testing_labels,1)
poly1d_fn = np.poly1d(coef) 
axs[0].plot(predictions, poly1d_fn(predictions), '--k') 
axs[0].legend([None, poly1d_fn])

axs[1].plot(predicted_values, true_values, '.')
axs[1].set_xlabel('predicted absolute value')
axs[1].set_xlabel('true absolute value')

fig.suptitle(f'Profit/loss: {sum(profit_loss)} at a cost of {sum(cost)}')
#print(profit_loss)

fig.savefig(results_filepath)

# print("testing labels: ", testing_labels)
# print("predictions: ", predictions)
# print("Difference: ", testing_labels - predictions)
# print("Difference relative: ", np.divide((testing_labels - predictions), predictions))