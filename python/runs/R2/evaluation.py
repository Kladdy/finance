import enum
from numpy.core.fromnumeric import size
from tensorflow.python.keras.backend import dtype
from toolbox import logger, load_testing_data, mkdir
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
from constants import run_name, results_folder_name, model_folder_name
import matplotlib.pyplot as plt

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluate model')
parser.add_argument('--collector', type=str, help='the collector, ie C1, C2...')
parser.add_argument('--period', type=str, help='the period, ie 1mo, 1y...')
parser.add_argument('--interval', type=str, help='the interval, ie 1m, 30m...')
parser.add_argument('--start', type=str, help='the start date, ie 2021.04.29...')
parser.add_argument('--stop', type=str, help='the stop date, ie 2021.05.28...')
parser.add_argument('--data_length', type=int, help='the amount of samples to have in the trace, ie 20')
parser.add_argument('--batch_size', type=int, help='the batch size, ie 64')
parser.add_argument('--conv_start', dest='conv_start', action='store_true', help='whether or not the model starts with convolutional layers')
parser.set_defaults(conv_start=False)

args = parser.parse_args()
collector = args.collector
period = args.period
interval = args.interval
start = args.start
stop = args.stop
data_length = args.data_length
batch_size = args.batch_size
conv_start = args.conv_start

logger.INFO(f"Evaluating {run_name}...")

mkdir(results_folder_name)

# Load testing data and convert to dataset tensor
testing_data, testing_labels, testing_tickers, testing_last_data_values = load_testing_data(collector, period, interval, start, stop, data_length)
if conv_start:  # If convolutional layers at the start, we need to reshape the data
  testing_data = np.expand_dims(testing_data, 2)
testing_dataset = tf.data.Dataset.from_tensor_slices((testing_data, testing_labels))

# Construct model file path
model_filename = f"model_{run_name}_{collector}_period{period}_interval{interval}_start{start}_end{stop}_datalength{data_length}_batchsize{batch_size}.h5"
model_filepath = f"{model_folder_name}/{model_filename}"

# Construct results file path
results_filename = f"results_{run_name}_{collector}_period{period}_interval{interval}_start{start}_end{stop}_datalength{data_length}_batchsize{batch_size}"
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

cost_absolute = np.zeros(N)
true_profit_loss_absolute = np.zeros(N)
true_margin_relative = np.zeros(N)

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

    cost_absolute[i] = sample_last_data_value
    true_profit_loss_absolute[i] = sample_true_absolute - sample_last_data_value
    true_margin_relative[i] = (sample_true_absolute - sample_last_data_value) / sample_last_data_value


# Get the percentage quantiles for predictions, and sum them up
def get_percentage_quantile_sums(percentage):
    idx_that_predicts_profit = predictions > 0
    amount_that_predicts_profit = sum(idx_that_predicts_profit)

    idx_and_predictions = np.array([np.array([idx, prediction]) for idx, prediction in enumerate(predictions)])
    sorted_idx_and_predictions = idx_and_predictions[idx_and_predictions[:, 1].argsort()] # sort by predictions, keeping index beside it
    idx_for_predictions_above_quantile = -round(percentage * amount_that_predicts_profit)

    if (idx_for_predictions_above_quantile < -N):
        raise ValueError(f"get_percentage_quantile_sums({percentage}): Index ({-idx_for_predictions_above_quantile}) exceeds the bound ({-N})")

    predictions_above_quantile = sorted_idx_and_predictions[(idx_for_predictions_above_quantile):, :]
    true_values_above_quantile = testing_labels[predictions_above_quantile[:, 0].astype(int)] # Extract what the true values will be

    # Return the sum of the predictions above the given quantile
    return sum(true_values_above_quantile), sum(predictions_above_quantile)

class Quantile:
    def __init__(self, quantile, sum_true, sum_predicted):
        self.quantile = quantile
        self.sum_true = sum_true
        self.sum_predicted = sum_predicted

quantiles = [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
quantile_sums = np.array([], dtype=Quantile)
for quantile in quantiles:
    try:
        sum_true, sum_predicted = get_percentage_quantile_sums(quantile)
        quantile_sums = np.append(quantile_sums, Quantile(quantile, sum_true, sum_predicted))
    except ValueError as e:
        logger.DEBUG(e)
    except Exception as e:
        logger.ERR(e)

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
axs[1].set_ylabel('true absolute value')

# fig.suptitle(f'Quantiles: {", ".join([f"({q[0]}: {q[1]:.3f})" for q in quantile_sums])}')
fig.tight_layout()

fig.savefig(f'{results_filepath}.png')

# Save quantile data
np.savez(f'{results_filepath}.npz', quantile_sums=quantile_sums)