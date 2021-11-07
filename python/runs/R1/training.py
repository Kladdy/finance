from tensorflow.python.keras import activations, callbacks
from toolbox import logger, load_training_data, load_validation_data, mkdir
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
from constants import run_name, data_folder_name, model_folder_name

# Parse arguments
parser = argparse.ArgumentParser(description='Prepare data for R1.')
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

logger.INFO(f"Training {run_name}...")

mkdir(model_folder_name)

# Define constants
model_filename = f"model_{run_name}_{collector}_period{period}_interval{interval}_start{start}_end{stop}_datalength{data_length}_batchsize{batch_size}"
model_filepath = f"{model_folder_name}/{model_filename}"

# Load training and validation data and convert to dataset tensors
training_data, training_labels = load_training_data(collector, period, interval, start, stop, data_length)
validation_data, validation_labels = load_validation_data(collector, period, interval, start, stop, data_length)

training_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_labels))
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_data, validation_labels))

# Put into batches
training_dataset = training_dataset.shuffle(100).batch(batch_size)
validation_dataset = validation_dataset.batch(batch_size)

# Create model
model = tf.keras.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(data_length,)))
model.add(tf.keras.layers.Dense(1280, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanAbsoluteError())

# Checkpoints
modelCheckpoint = ModelCheckpoint(model_filepath, save_best_only=True)
modelCallbacks = [modelCheckpoint]

# Perform the fit
model.fit(x=training_dataset, validation_data=validation_dataset,
            callbacks=modelCallbacks, epochs=10)