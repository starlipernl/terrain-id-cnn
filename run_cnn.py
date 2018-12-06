# ECE 542
# Project 3: CNN
# October 2018
# Description: This script trains and tests the final CNN after hyper-parameter optimization.

# Labels
# 0 - sidewalk, 1 - carpet, 2 - brick, 3 - asphalt, 4 - grass, 5 - tile

################################################################################
# IMPORTs
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.backend import set_session
import os
import numpy as np
from datetime import datetime
import pandas as pd
import cnn
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


################################################################################
print("TF version: {}".format(tf.__version__))
print("GPU available: {}".format(tf.test.is_gpu_available()))

# Limit memory usage when running on titan
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.25
set_session(tf.Session(config=config))

################################################################################

# loading dataset
features, label = cnn.load_data(color=1)
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2)
y_test = y_test.reshape((len(y_test),))
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# x_train = x_train.reshape((len(x_train), x_train.shape[1], x_train.shape[2]), 1)
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
valid_images = x_train[2800:]
valid_labels = y_train[2800:]
x_train = x_train[0:2800]
y_train = y_train[0:2800]


################################################################################
# HYPERPARAMETERS AND DESIGN CHOICES
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.00017444
NUM_NEURONS_IN_DENSE_1 = 64
DROP_PROB = 0.3897
ACTIV_FN = "relu"
activation_fn = cnn.get_activ_fn(ACTIV_FN)

################################################################################
# input image dimensions

# input_shape = (img_rows, img_cols, num_channels)
input_shape = x_train[1].shape
'''
train_images = train_images.reshape(-1, img_rows, img_cols, num_channels)
valid_images = valid_images.reshape(-1, img_rows, img_cols, num_channels)
test_images = test_images.reshape(-1, img_rows, img_cols, num_channels)
'''

# output dimensions
num_classes = 6

################################################################################
# callbacks for Save weights, Tensorboard
# creating a new directory for each run using timestamp
folder = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"), str(ACTIV_FN))
history_file = folder + "\cnn_" + str(ACTIV_FN) + ".h5"
save_callback = ModelCheckpoint(filepath=history_file, verbose=1)
tb_callback = TensorBoard(log_dir=folder)

# Build, train, and test model
model = cnn.build_model(input_shape=input_shape, learn_rate=LEARNING_RATE, drop_prob=DROP_PROB, num_neurons=NUM_NEURONS_IN_DENSE_1)
train_accuracy, train_loss, valid_accuracy, valid_loss = cnn.train_model(model, x_train, y_train, BATCH_SIZE,
                                                                     NUM_EPOCHS, valid_images, valid_labels,
                                                                     save_callback, tb_callback)
test_accuracy, test_loss, predictions = cnn.test_model(model, x_test, y_test)

# save test set results to csv
predictions = np.round(predictions)
predictions = predictions.astype(int)
df = pd.DataFrame(predictions)
df.to_csv("predictions.csv", header=None, index=None)

################################################################################
# Visualization and Output
num_epochs_plot = range(1, len(train_accuracy) + 1)

# Loss curves
plt.figure()
plt.plot(num_epochs_plot, train_loss, "b", label="Training Loss")
plt.plot(num_epochs_plot, valid_loss, "r", label="Validation Loss")
plt.title("Loss Curves_" + ACTIV_FN)
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Figures/' + ACTIV_FN + '_loss.png')
plt.close()
#plt.show()

# Accuracy curves
plt.figure(2)
plt.plot(num_epochs_plot, train_accuracy, "b", label="Training Accuracy")
plt.plot(num_epochs_plot, valid_accuracy, "r", label="Validation Accuracy")
plt.title("Accuracy Curves_" + ACTIV_FN)
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('Figures/' + ACTIV_FN + '_acc.png')
plt.close()
#plt.show()

# Test loss and accuracy
print("\n##########")
print("Test Loss: {:.4f}".format(test_loss))
print("Test Accuracy: {:.4f}".format(test_accuracy))
print("##########")