from toolbox import logger, load_training_data, load_validation_data
import numpy as np
import tensorflow as tf
import argparse
from constants import run_name

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

# Load training and validation data and convert to dataset tensors
training_data, training_labels = load_training_data(collector, period, interval, start, stop, data_length)
validation_data, validation_labels = load_validation_data(collector, period, interval, start, stop, data_length)

training_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_labels))
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_data, validation_labels))

# Put into batches
training_dataset = training_dataset.batch(batch_size)
validation_dataset = validation_dataset.batch(batch_size)

#
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(10,)),
    tf.keras.layers.Dense(1280, activation='relu'),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanAbsoluteError())

model.fit(training_dataset, epochs=10)

model.evaluate(validation_dataset)