# ECE 542
# Project 3: CNN
# October 2018
# Description: This script trains and tests the final CNN after hyper-parameter optimization.

# Labels
# 0 - sidewalk, 1 - carpet, 2 - brick, 3 - asphalt, 4 - grass, 5 - tile

################################################################################
# IMPORTs
import tensorflow as tf
from tensorflow.keras.backend import set_session
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.activations import relu, softmax, tanh, sigmoid
import numpy as np
import cnn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.use('Agg')

################################################################################
# print("TF version: {}".format(tf.__version__))
# print("GPU available: {}".format(tf.test.is_gpu_available()))
#
# # Limit memory usage when running on titan
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# set_session(tf.Session(config=config))

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
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=20,
                             vertical_flip=True, width_shift_range=0.15, height_shift_range=0.15)
val_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# fit parameters from data
datagen.fit(x_train)
val_datagen.fit(valid_images)
# Center mean and standardize testing data
x_test -= np.mean(x_test, axis=0)
x_test /= np.std(x_test, axis=0)



################################################################################
# HYPERPARAMETERS AND DESIGN CHOICES
NUM_EPOCHS = 50
# BATCH_SIZE = 20
# LEARNING_RATE = 0.0001
# NUM_NEURONS_IN_DENSE_1 = 128
# DROP_PROB = 0.5
# ACTIV_FN = "relu"
# activation_fn = cnn.get_activ_fn(ACTIV_FN)

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
# folder = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"), str(ACTIV_FN))
# history_file = folder + "\cnn_" + str(ACTIV_FN) + ".h5"
# save_callback = ModelCheckpoint(filepath=history_file, verbose=1)
# tb_callback = TensorBoard(log_dir=folder)

# Build, train, and test model
model = KerasClassifier(build_fn=cnn.build_model(), epochs=NUM_EPOCHS, verbose=2)
activation = [relu, tanh, sigmoid]
learn_rate = [0.00001, 0.0001, 0.0005, 0.001, 0.01]
drop_prob = [0.1, 0.3, 0.5, 0.6]
num_neurons = [64, 128, 256, 512, 1024]
batch_size = [10, 20, 40]
param_grid = dict(learn_rate=learn_rate, drop_prob=drop_prob, num_neurons=num_neurons, batch_size=batch_size)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# train_accuracy, train_loss, valid_accuracy, valid_loss = cnn.train_model_datagen(model, datagen, val_datagen, x_train,
#                                                          y_train, BATCH_SIZE, NUM_EPOCHS, valid_images, valid_labels,
#                                                          save_callback, tb_callback)


