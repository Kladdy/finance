import os
import sys
from tensorflow import python
from tensorflow.python.keras import activations, callbacks
from tensorflow.python.ops.gen_batch_ops import batch
from toolbox import logger, load_training_data, load_validation_data, mkdir, get_model_filepath, load_evaluation_data, get_results_filepath, Quantile
import wandb
from wandb.keras import WandbCallback
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
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

# Get argument string (for calling the evaluator)
arg_string = ' '.join(sys.argv[1:])

logger.INFO(f"Training {run_name}...")

mkdir(model_folder_name)

# Define constants
learning_rate=0.000001
epochs = 3

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
run = wandb.init(project="finance", entity="sigfid", group=run_name)
wandb.config = {
  learning_rate: learning_rate,
  epochs: epochs,
  collector: collector,
  period: period,
  interval: interval,
  start: start,
  stop: stop,
  data_length: data_length,
  batch_size: batch_size,
  conv_start: conv_start
}

# Get the run id
run_id = wandb.run.name

# File paths
model_filepath = get_model_filepath(run_id)

# Create model
model = tf.keras.Sequential()

# model.add(tf.keras.layers.Conv1D(128, 3, padding='same', input_shape=input_shape))
# model.add(tf.keras.layers.Conv1D(128, 3, padding='same'))
# model.add(tf.keras.layers.Conv1D(128, 3, padding='same'))
# model.add(tf.keras.layers.Conv1D(128, 3, padding='same'))
# model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Flatten(input_shape=input_shape))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss=tf.keras.losses.MeanAbsoluteError())

# Callbacks
modelCheckpoint = ModelCheckpoint(model_filepath, save_best_only=True, monitor='val_loss', 
                                    verbose=0, mode='auto', save_weights_only=False)
wandbCallback = WandbCallback()
callback_list = [modelCheckpoint, wandbCallback]

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