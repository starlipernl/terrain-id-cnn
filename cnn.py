"""
Main script containing all functions pertaining to the CNN model. This script is referenced by the other scripts for
loading data, building, training, and testing model.

"""


# IMPORTs
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.activations import relu, softmax, tanh, sigmoid, elu
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


# import data from pickle files. setup train test split (80/20) and normalize data to [0,1]
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
    return x_train, y_train, x_test, y_test


# build model, update hyperparameters here after running "run_hyperopt_cnn.py" with optimal params
def build_model(input_shape=(64, 64, 1), learn_rate=0.00075633, drop_prob=0.02076, num_neurons=256, num_classes=6):
    model = Sequential()
    model.add(Conv2D(64, 5, input_shape=input_shape, padding="same", activation=elu))
    model.add(Conv2D(256, 3, padding="same", activation=relu))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Conv2D(64, 3, padding="same", activation=relu))
    model.add(Conv2D(32, 3, padding="same", activation=relu))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=8, strides=8))
    model.add(Flatten())
    model.add(Dense(units=num_neurons, activation=elu))
    model.add(Dropout(drop_prob))
    model.add(Dense(units=num_classes, activation=softmax))

    # configure model for training
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(lr=learn_rate, decay=0.001), metrics=["accuracy"])
    model.summary()
    return model


# train model
def train_model(model, train_images, train_labels, BATCH_SIZE, NUM_EPOCHS,
                valid_images, valid_labels, save_callback, tb_callback):
    history = model.fit(
        x=train_images,
        y=train_labels,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=(valid_images, valid_labels),
        shuffle=True,
        callbacks=[save_callback, tb_callback],
        verbose=1
    )
    history_dict = history.history
    train_accuracy = history_dict["acc"]
    train_loss = history_dict["loss"]
    valid_accuracy = history_dict["val_acc"]
    valid_loss = history_dict["val_loss"]
    return train_accuracy, train_loss, valid_accuracy, valid_loss


# train model with data generator
# data generator requires manual input of number of steps
def train_model_datagen(model, data_generator, valid_generator, x_train, y_train, batch_size, num_epochs,
                        x_valid, y_valid, save_callback, tb_callback):
    train_steps = x_train.shape[0]//batch_size
    valid_steps = x_valid.shape[0]//batch_size
    history = model.fit_generator(data_generator.flow(x_train, y_train, batch_size=batch_size),
                                  epochs=num_epochs,
                                  verbose=1,
                                  validation_data=valid_generator.flow(x_valid, y_valid, batch_size=batch_size),
                                  steps_per_epoch=train_steps,
                                  validation_steps=valid_steps,
                                  callbacks=[save_callback, tb_callback])
    history_dict = history.history
    train_accuracy = history_dict["acc"]
    train_loss = history_dict["loss"]
    valid_accuracy = history_dict["val_acc"]
    valid_loss = history_dict["val_loss"]
    return train_accuracy, train_loss, valid_accuracy, valid_loss


# evaluation on test set
def test_model(model, test_images, test_labels):
    test_loss, test_accuracy = model.evaluate(
        x=test_images,
        y=test_labels,
        verbose=0
    )
    # predictions with test set
    predictions = model.predict_proba(
        x=test_images,
        batch_size=None,
        verbose=0
    )
    return test_accuracy, test_loss, predictions





