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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from datetime import datetime
import pandas as pd
import cnn
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle


################################################################################
print("TF version: {}".format(tf.__version__))
print("GPU available: {}".format(tf.test.is_gpu_available()))

# Limit memory usage when running on titan
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

################################################################################

# Data augmentation
# loading dataset
with open('features.pkl', 'rb') as file:
    features = pickle.load(file)
with open('labels.pkl', 'rb') as file:
    labels = pickle.load(file)
feature_list = np.asarray(features)
label_list = np.asarray(labels).reshape((-1, 1))
x_train, x_test, y_train, y_test = train_test_split(feature_list, label_list, test_size=0.2)
y_test = y_test.reshape((len(y_test),))
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
val_split = round(x_train.shape[0] * 0.8)
x_valid = x_train[val_split:]
y_valid = y_train[val_split:]
x_train = x_train[0:val_split]
y_train = y_train[0:val_split]
datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, rotation_range=20,
                             horizontal_flip=True, width_shift_range=0.15, height_shift_range=0.15)
val_datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False)
datagen.fit(x_train)
val_datagen.fit(x_valid)
# # Center mean and standardize testing data
# x_test -= np.mean(x_test, axis=0)
# x_test /= np.std(x_test, axis=0)



################################################################################
# HYPERPARAMETERS AND DESIGN CHOICES
NUM_EPOCHS = 1000
BATCH_SIZE = 16
LEARNING_RATE = 0.0075633
NUM_NEURONS_IN_DENSE_1 = 256
DROP_PROB = 0.02076
#ACTIV_FN = "relu"


################################################################################
# input image dimensions
# input_shape = (img_rows, img_cols, num_channels)
input_shape = x_train[1].shape

# output dimensions
num_classes = 6

################################################################################
# callbacks for Save weights, Tensorboard
# creating a new directory for each run using timestamp
folder = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
history_file = folder + "\cnn_" + ".h5"
save_callback = ModelCheckpoint(filepath=history_file, verbose=1)
tb_callback = TensorBoard(log_dir=folder)

# Build, train, and test model
model = cnn.build_model(input_shape=input_shape, learn_rate=LEARNING_RATE, drop_prob=DROP_PROB, num_neurons=NUM_NEURONS_IN_DENSE_1)
train_accuracy, train_loss, valid_accuracy, valid_loss = cnn.train_model_datagen(model, datagen, val_datagen, x_train,
                                                         y_train, BATCH_SIZE, NUM_EPOCHS, x_valid, y_valid,
                                                         save_callback, tb_callback)
test_accuracy, test_loss, predictions = cnn.test_model(model, x_test, y_test)

# save test set results to csv
predictions = np.round(predictions)
predictions = predictions.astype(int)
df = pd.DataFrame(predictions)
df.to_csv("predictions_datagen.csv", header=None, index=None)

################################################################################
# Visualization and Output
num_epochs_plot = range(1, len(train_accuracy) + 1)

# Loss curves
plt.figure()
plt.plot(num_epochs_plot, train_loss, "b", label="Training Loss")
plt.plot(num_epochs_plot, valid_loss, "r", label="Validation Loss")
plt.title("Loss Curves")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Figures/loss_datagen.png')
plt.close()
#plt.show()

# Accuracy curves
plt.figure(2)
plt.plot(num_epochs_plot, train_accuracy, "b", label="Training Accuracy")
plt.plot(num_epochs_plot, valid_accuracy, "r", label="Validation Accuracy")
plt.title("Accuracy Curves")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('Figures/acc_datagen.png')
plt.close()
#plt.show()

# Test loss and accuracy
print("\n##########")
print("Test Loss: {:.4f}".format(test_loss))
print("Test Accuracy: {:.4f}".format(test_accuracy))
print("##########")