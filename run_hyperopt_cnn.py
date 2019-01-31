"""
Main script for Bayesian hyperparameter optimization. This script is completely independent of the cnn.py script.
This script contains many of the same functions as cnn.py however, because of the nature of the hyper optimization
library, we had to recreate the scripts within this self contained script.
"""

# IMPORTs
import tensorflow as tf
from tensorflow.keras.backend import set_session
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.activations import relu, softmax, tanh, elu
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, lognormal
import csv
import pickle


# data function for loading data, load from pkl files, split 80/20 training testing, further split training into
# 80/20 training validation. Set up data generators for data augmentation
def data():
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
    return x_valid, y_valid, datagen, val_datagen, x_train, y_train,  x_test, y_test


# function to build CNN sequential model. The search space is first constructed, then the sequential model is built
def create_model(x_valid, y_valid, datagen, val_datagen, x_train, y_train):
    # construct the search space
    conv1_numfilters = {{choice([32, 64, 128, 256])}}
    conv1_size = {{choice([(2, 2), (3, 3), (5, 5)])}}
    conv1_activation = {{choice([relu, tanh, elu])}}
    conv2_numfilters = {{choice([32, 64, 128, 256])}}
    conv2_size = {{choice([(2, 2), (3, 3), (5, 5)])}}
    conv2_activation = {{choice([relu, tanh, elu])}}
    conv3_numfilters = {{choice([32, 64, 128, 256])}}
    conv3_size = {{choice([(2, 2), (3, 3), (5, 5)])}}
    conv3_activation = {{choice([relu, tanh, elu])}}
    conv4_numfilters = {{choice([32, 64, 128, 256])}}
    conv4_size = {{choice([(2, 2), (3, 3), (5, 5)])}}
    conv4_activation = {{choice([relu, tanh, elu])}}
    maxpool1_size = {{choice([2, 4, 8])}}
    maxpool2_size = {{choice([2, 4, 8])}}
    dense_activation = {{choice([relu, tanh, elu])}}
    num_neurons = {{choice([64, 128, 256, 512])}}
    dropout = {{uniform(0, 1)}}
    learn_rate = {{uniform(0.00001, 0.01)}}
    batch_size = {{choice([16, 32, 64])}}
    # build model
    model = Sequential()
    model.add(Conv2D(conv1_numfilters, conv1_size, input_shape=(64, 64, 1), padding="same", activation=conv1_activation))
    model.add(Conv2D(conv2_numfilters, conv2_size, padding="same", activation=conv2_activation))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=maxpool1_size))
    model.add(Conv2D(conv3_numfilters, conv3_size, padding="same", activation=conv3_activation))
    model.add(Conv2D(conv4_numfilters, conv4_size, padding="same", activation=conv4_activation))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=maxpool2_size))
    model.add(Flatten())
    model.add(Dense(units=num_neurons, activation=dense_activation))
    model.add(Dropout(dropout))
    model.add(Dense(units=6, activation=softmax))
    model.compile(loss=sparse_categorical_crossentropy,  optimizer=Adam(lr=learn_rate, decay=0.001), metrics=["accuracy"])
    model.summary()
    train_steps = x_train.shape[0]//batch_size
    valid_steps = x_valid.shape[0]//batch_size
    result = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                 epochs=80,
                                 verbose=2,
                                 validation_data=val_datagen.flow(x_valid, y_valid, batch_size=batch_size),
                                 steps_per_epoch=train_steps,
                                 validation_steps=valid_steps)
    # take max validation accuracy as metric
    validation_acc = np.amax(result.history['val_acc'])
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model,
              'history.val_loss': result.history['val_loss'], 'history.val_acc': result.history['val_acc']}


# main script runs the hyper optimization and saves results
if __name__ == '__main__':
    # Limit memory usage of gpu
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # set_session(tf.Session(config=config))
    # loading dataset
    # setup trials objects that saves and contains all information from optimization steps
    trials = Trials()
    best_run, best_model, space = optim.minimize(model=create_model,
                                                 data=data,
                                                 algo=tpe.suggest,
                                                 max_evals=200,
                                                 trials=trials,
                                                 return_space=True)
    x_valid, y_valid, datagen, val_datagen, x_train, y_train, x_test, y_test = data()
    best_loss, best_acc = best_model.evaluate(x_test, y_test)  # evaluate model returned by optimization
    best_result = [trials.best_trial['tid'], best_loss, best_acc, best_run]
    best_model.save('best_model.h5')
    losses = [ii['history.val_loss'] for ii in trials.results]
    acc = [ii['history.val_acc'] for ii in trials.results]
    cv_params = trials.vals
    cv_results = {'best_run': best_result, 'params': cv_params, 'losses': losses, 'acc': acc}
    # save results to csv and pickle
    with open('hyperparams.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, val in best_run.items():
            writer.writerow([key, val])
    with open('cv_results.pkl', 'wb') as file:
        pickle.dump(cv_results, file)
    print("Evalutation of best performing model:")
    print(best_loss, best_acc)
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
