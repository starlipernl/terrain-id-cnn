# ECE 542
# Project 3: CNN
# October 2018
# Description: This file is the script to run the cross-validation of hyper-parameters.

# dataset: MNIST
# training set size: 60k
# test set size: 10k
# 28x28x1
# 10 class labesl (digits 0-9)

# Notes:
#   - validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data

################################################################################
# IMPORT
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import time
import cnn

# loading dataset
(train_images, train_labels), (test_images, test_labels) = cnn.load_mnist_data()

# creating validation set from training set
# validation set size:
valid_set_size = 8000
split = len(train_images) - valid_set_size
valid_images = train_images[split:]  # last
valid_labels = train_labels[split:]  # last
train_images = train_images[:split]  # first
train_labels = train_labels[:split]  # first

# HYPERPARAMETERS AND DESIGN CHOICES
# Change the hyper parameter you wish to tune to an array with the range of values
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_NEURONS_IN_DENSE_1 = [64, 128, 200, 256]
DROP_PROB = 0.6
ACTIV_FN = "relu"
activation_fn = cnn.get_activ_fn(ACTIV_FN)

# input image dimensions
img_rows, img_cols = 28, 28
num_channels = 1
input_shape = (img_rows, img_cols, num_channels)

# output dimensions
num_classes = 10

# callbacks for Save weights, Tensorboard
# creating a new directory for each run using timestamp
folder = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"), str(ACTIV_FN))
history_file = folder + "\cnn_" + str(ACTIV_FN) + ".h5"
save_callback = ModelCheckpoint(filepath=history_file, verbose=1)
tb_callback = TensorBoard(log_dir=folder)


# Specify parameter to be validated here
param_val = NUM_NEURONS_IN_DENSE_1
parm = 'Neurons_in_Dense_Layer'
val_train_loss = np.zeros(len(param_val))
val_train_acc = np.zeros(len(param_val))
val_valid_acc = np.zeros(len(param_val))
val_valid_loss = np.zeros(len(param_val))
train_time = np.zeros(len(param_val))

# Build and train the model and record validation and loss statistics for each value of the hyperparameter
# Make sure to add the [val] index after the hyper parameter that will be tuned.
for val, param in enumerate(param_val):
    model = cnn.build_model(input_shape, activation_fn, LEARNING_RATE, DROP_PROB, NUM_NEURONS_IN_DENSE_1[val], num_classes)
    t0 = time.time()
    train_accuracy, train_loss, valid_accuracy, valid_loss = cnn.train_model(model, train_images, train_labels,
                                                                             BATCH_SIZE, NUM_EPOCHS, valid_images,
                                                                             valid_labels, save_callback, tb_callback)
    t1 = time.time()
    train_time[val] = t1-t0
    val_train_loss[val] = train_loss[-1]
    val_train_acc[val] = train_accuracy[-1]
    val_valid_loss[val] = valid_loss[-1]
    val_valid_acc[val] = valid_accuracy[-1]


# Visualization and Output
# Loss curves
plt.figure(1)
plt.plot(param_val, val_train_loss, "b", label="Training Loss")
plt.plot(param_val, val_valid_loss, "r", label="Validation Loss")
plt.title("Loss Curves " + parm)
plt.xlabel("Parameter Values")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Figures/' + parm + '_loss.png')
#plt.show()

# Accuracy curves
plt.figure(2)
plt.plot(param_val, val_train_acc, "b", label="Training Accuracy")
plt.plot(param_val, val_valid_acc, "r", label="Validation Accuracy")
plt.title("Accuracy Curves " + parm)
plt.xlabel("Parameter Values")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('Figures/' + parm + '_acc.png')
#plt.show()

# Time curves
plt.figure(3)
plt.plot(param_val, train_time, "b", label="Training Accuracy")
plt.title("Computational Time " + parm)
plt.xlabel("Parameter Values")
plt.ylabel("Time(s)")
plt.savefig('Figures/' + parm + '_time.png')
#plt.show()
