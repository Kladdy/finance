import os
import sys
from tensorflow import python
from tensorflow.python.keras import activations, callbacks
from tensorflow.python.ops.gen_batch_ops import batch
from toolbox import logger, load_training_data, load_validation_data, mkdir, get_model_filepath, load_evaluation_data, get_results_filepath, check_if_data_is_prepared, Quantile
import wandb
from wandb.keras import WandbCallback
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model
import argparse
from constants import run_name, model_folder_name
from PIL import Image

# Parse arguments
parser = argparse.ArgumentParser(description='Run training')
parser.add_argument('--collector', type=str, help='the collector, ie C1, C2...')
parser.add_argument('--period', type=str, help='the period, ie 1mo, 1y...')
parser.add_argument('--interval', type=str, help='the interval, ie 1m, 30m...')
parser.add_argument('--start', type=str, help='the start date, ie 2021.04.29...')
parser.add_argument('--stop', type=str, help='the stop date, ie 2021.05.28...')
parser.add_argument('--data_length', type=int, help='the amount of samples to have in the trace, ie 20')
parser.add_argument('--batch_size', type=int, help='the batch size, ie 64')
parser.add_argument('--learning_rate', type=float, help='the learning rate, ie 0.000001')
parser.add_argument('--epochs', type=int, help='the amount of epochs, ie 50')
parser.add_argument('--optimizer', type=str, help='the optimizer, ie adam')
parser.add_argument('--activation_function', type=str, help='the activation function, ie relu')
parser.add_argument('--loss_function', type=str, help='the loss function, ie mae')
parser.add_argument('--conv_layers', type=int, help='the amount of conv layers')
parser.add_argument('--conv_filters', type=int, help='the amonunt of conv filters per layer')
parser.add_argument('--conv_kernel_size', type=int, help='the conv kernel size')
parser.add_argument('--conv_padding', type=str, help='the conv padding, ie same')
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
learning_rate = args.learning_rate
epochs = args.epochs
optimizer = args.optimizer
activation_function = args.activation_function
loss_function = args.loss_function
conv_layers = args.conv_layers
conv_filters = args.conv_filters
conv_kernel_size = args.conv_kernel_size
conv_padding = args.conv_padding
conv_start = args.conv_start

# Make sure that the kernel size is smaller or equal to the data length
assert conv_kernel_size <= data_length

logger.INFO(f"Training {run_name}...")

mkdir(model_folder_name)

# Define constants
es_patience = 3
es_min_delta = 0.0001

# Make sure data exists
if not check_if_data_is_prepared(collector, period, interval, start, stop, data_length):
  logger.INFO("Data has not been prepared. Generating new data...")
  os.system(f"python prepare_data.py --collector={collector} --period={period} --interval={interval} --start={start} --stop={stop} --data_length={data_length}")
else:
  logger.INFO("Data exists! Proceeding...")

# Load training and validation data and convert to dataset tensors
training_data, training_labels = load_training_data(collector, period, interval, start, stop, data_length)
validation_data, validation_labels = load_validation_data(collector, period, interval, start, stop, data_length)

# If convolutional layers at the start, we need to reshape the data and input shape
if conv_start:
  input_shape = (data_length, 1)
  training_data = np.expand_dims(training_data, 2)
  validation_data = np.expand_dims(validation_data, 2)
else:
  input_shape = (data_length, )

training_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_labels))
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_data, validation_labels))

# Put into batches
training_dataset = training_dataset.shuffle(100).batch(batch_size)
validation_dataset = validation_dataset.batch(batch_size)

# Initialize wandb
run = wandb.init(project="finance", entity="sigfid", group=run_name, config=args)

# Get the run id
run_id = wandb.run.name

# File paths
model_filepath = get_model_filepath(run_id)

# Create the optimizer
if optimizer == "adam":
  optimizer = tf.keras.optimizers.Adam(learning_rate)
elif optimizer == "sgd":
  optimizer =tf.keras.optimizers.SGD(learning_rate)
else:
  logger.ERR(f"Optimizer {optimizer} not supported")
  raise ValueError(f"Optimizer {optimizer} not supported")

# Create the loss function
if loss_function == "mean_absolute_error":
  loss = tf.keras.losses.MeanAbsoluteError()
elif loss_function == "mean_squared_error":
  loss = tf.keras.losses.MeanSquaredError()
elif loss_function == "mean_absolute_percentage_error":
  loss = tf.keras.losses.MeanAbsolutePercentageError()
elif loss_function == "mean_squared_logarithmic_error":
  loss = tf.keras.losses.MeanSquaredLogarithmicError()
elif loss_function == "cosine_similarity":
  loss = tf.keras.losses.CosineSimilarity()
elif loss_function == "huber":
  loss = tf.keras.losses.Huber()
elif loss_function == "log_cosh":
  loss = tf.keras.losses.LogCosh()
else:
  logger.ERR(f"Loss function {loss_function} not supported")
  raise ValueError(f"Loss function {loss_function} not supported")

# Create the activation function
if activation_function == "relu":
  activation = activations.relu
elif activation_function == "sigmoid":
  activation = activations.sigmoid
elif activation_function == "softmax":
  activation = activations.softmax
elif activation_function == "softplus":
  activation = activations.softplus
elif activation_function == "softsign":
  activation = activations.softsign
elif activation_function == "tanh":
  activation = activations.tanh
elif activation_function == "selu":
  activation = activations.selu
elif activation_function == "elu":
  activation = activations.elu
elif activation_function == "exponential":
  activation = activations.exponential
else:
  logger.ERR(f"Activation function {loss_function} not supported")
  raise ValueError(f"Activation function {loss_function} not supported")

# ---------------------------
#        Create model
# ---------------------------
model = tf.keras.Sequential()

# Add first conv layer
model.add(tf.keras.layers.Conv1D(conv_filters, conv_kernel_size, padding=conv_padding, activation=activation, input_shape=input_shape))

# Add the rest
for _ in range(conv_layers - 1):
  model.add(tf.keras.layers.Conv1D(conv_filters, conv_kernel_size, padding=conv_padding, activation=activation))

model.add(tf.keras.layers.Flatten())

  # If not using conv layers, flatten like this (first layer, is it really needed?)
  # model.add(tf.keras.layers.Flatten(input_shape=input_shape))

model.add(tf.keras.layers.Dense(128, activation=activation))
model.add(tf.keras.layers.Dense(10, activation=activation))

# Output layer
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer=optimizer, loss=loss)
# ---------------------------

# Save model as image
plot_model(model, to_file=f'{model_filepath}.png', show_shapes=True)

# Callbacks
es = EarlyStopping(monitor="val_loss", patience=es_patience, min_delta=es_min_delta, verbose=1),
mc = ModelCheckpoint(model_filepath, save_best_only=True, monitor='val_loss', 
                                    verbose=0, mode='auto', save_weights_only=False)
wb = WandbCallback(save_model=False)
callback_list = [es, mc, wb]

# Perform the fit
model.fit(x=training_dataset, validation_data=validation_dataset,
            callbacks=callback_list, epochs=epochs)

# Evaluate the model
os.system(f"python evaluation.py --collector={collector} --period={period} --interval={interval} --start={start} --stop={stop} --data_length={data_length}{' --conv_start' if conv_start else ''} --run_id={run_id}")

try:
  # Get evaluator from evaluator and send to wandb
  quantile_sums, linear_coef, cov_off_diagonal = load_evaluation_data(run_id)
  quantile_dictionary_true = {f"q_{quantile.quantile}_true": quantile.sum_true for quantile in quantile_sums}
  quantile_dictionary_pred = {f"q_{quantile.quantile}_pred": quantile.sum_predicted for quantile in quantile_sums}
  wandb.log(quantile_dictionary_true)
  wandb.log(quantile_dictionary_pred)
  wandb.log({
    'linear_k': linear_coef[0],
    'linear_m': linear_coef[1],
    'cov_off_diagonal': cov_off_diagonal
  })

  # Get evaluation plot and upload to wandb
  results_filepath = get_results_filepath(run_id)
  run_results_image = Image.open(f"{results_filepath}.png")
  wandb.log({"run_results_image": [wandb.Image(run_results_image, caption=f"results for {run_id}")]})
except Exception as e:
  logger.ERR(e)

run.finish()