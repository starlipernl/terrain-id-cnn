
# ECE 542
# Final Project
# December 2018
# Description: This script runs bayesian hyper-parameter optimization

# Labels
# 0 - sidewalk, 1 - carpet, 2 - brick, 3 - asphalt, 4 - grass, 5 - tile

################################################################################
# IMPORTs

import numpy as np
from keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from tensorflow.keras import Model, Sequential
from tensorflow.keras.activations import relu, tanh, elu
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.losses import mse, binary_crossentropy

from sklearn.model_selection import train_test_split
from hpsklearn import HyperoptEstimator
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, lognormal
import csv
import pickle
import matplotlib as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
import time

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

# ---------------------
# Run Random Forest Classifier
# ---------------------
def run_rf(train_img, y_train, test_img, y_test, msg):
    clf = RandomForestClassifier(50)
    clf.fit(train_img, np.squeeze(y_train))

    print(msg + 'Score')
    print(clf.score(test_img, np.squeeze(y_test)))

    return clf.score(test_img, np.squeeze(y_test))

def data():
    with open('features.pkl', 'rb') as file:
        features = pickle.load(file)
    with open('labels.pkl', 'rb') as file:
        labels = pickle.load(file)
    feature_list = np.asarray(features)
    label_list = np.asarray(labels).reshape((-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(feature_list, label_list, test_size=0.2)
    y_test = y_test.reshape((len(y_test),))

    val_split = round(x_train.shape[0] * 0.8)
    x_valid = x_train[val_split:]
    y_valid = y_train[val_split:]
    x_train = x_train[0:val_split]
    y_train = y_train[0:val_split]
    #datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, rotation_range=20,
    #                             vertical_flip=True, width_shift_range=0.15, height_shift_range=0.15)
    #val_datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False)

    #datagen.fit(x_train)
    #val_datagen.fit(x_valid)

    # Normalizing images
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

    x_valid = x_valid.astype('float32') / 255.
    x_valid = x_valid.reshape((len(x_valid), np.prod(x_valid.shape[1:])))

    x_test = x_test.astype('float32') / 255.0
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    return x_valid, y_valid, x_train, y_train,  x_test, y_test


################################################################################
def create_model(x_valid, y_valid, datagen, val_datagen, x_train, y_train):

    global gb_acc
    global gb_bch_sz
    global gb_inter_dim
    global gb_lat_dim
    global gb_act_fn
    global gb_lrn_rate
    global gb_lr_dcy

    # ---------------------
    # Run Random Forest Classifier
    # ---------------------
    def run_rf(train_img, y_train, test_img, y_test, msg):
        clf = RandomForestClassifier(50)
        clf.fit(train_img, np.squeeze(y_train))

        print(msg + 'Score')
        print(clf.score(test_img, np.squeeze(y_test)))

        return clf.score(test_img, np.squeeze(y_test))

    # construct the search space
    def sampling(args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    original_dim = len(x_train[1])
    input_shape = (original_dim,)
    batch_size = {{choice([16, 32, 64, 128, 256])}}
    intermediate_dim = {{choice([32, 64, 128, 256, 512])}}
    latent_dim = {{choice([2, 4, 16, 32, 64, 256])}}
    act_fncs = {{choice([relu, tanh, elu])}}
    learn_rate = {{uniform(0.00001, 0.005)}}
    lr_decay = {{uniform(0.0001, 0.005)}}
    epochs = 400

    # save for later
    print('---------------------------------------')
    print('Batch size:  {}'.format(batch_size))
    print('Latent dim:  {}'.format(latent_dim))
    print('Act Fncs:    {}'.format(act_fncs))
    print('Learn Rate:   {}'.format(learn_rate))
    print('Learn Decay:  {}'.format(lr_decay))

    # network parameters
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation=act_fncs)(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    #encoder.summary()
    # plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation=act_fncs)(latent_inputs)
    outputs = Dense(original_dim, activation=act_fncs)(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    #decoder.summary()
    # plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=Adam(lr=learn_rate, decay=lr_decay), metrics=['accuracy'])
    vae.summary()
    vae.save_weights('VAE_weights.h5')

    train_steps = x_train.shape[0]//batch_size
    valid_steps = x_valid.shape[0]//batch_size

    result = vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            verbose=0,
            batch_size=batch_size,
            validation_data=(x_valid, None))

    encoder = Model(inputs, z_mean)

    z_valid = encoder.predict(x_valid, batch_size=batch_size)
    train_img = encoder.predict(x_train, batch_size=batch_size)
    # take max validation accuracy as metric

    # Run Random Forest Classifier and plot result if n < 4
    validation_acc = run_rf(train_img, y_train, z_valid, y_valid, "VAE-RF ")

    # save for later
    print('Val. Acc:    {}'.format(validation_acc))

    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': encoder, 'history.val_loss': result.history['val_loss']}

################################################################################


if __name__ == '__main__':

    t0 = time.time()
    # Limit memory usage of gpu
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # set_session(tf.Session(config=config))
    # loading dataset
    trials = Trials()
    evals = 400
    best_run, best_model, space = optim.minimize(model=create_model,
                                                 data=data,
                                                 algo=tpe.suggest,
                                                 max_evals=evals,
                                                 trials=trials,
                                                 return_space=True)
    x_valid, y_valid, x_train, y_train, x_test, y_test = data()
    best_z = best_model.predict(x_test,)
    train_img = best_model.predict(x_train)
    # take max validation accuracy as metric

    # Run Random Forest Classifier and plot result if n < 4
    best_acc = run_rf(train_img, y_train, best_z, y_test, "VAE-RF-Best ")
    best_loss = 1 - best_acc

    best_result = [trials.best_trial['tid'], best_loss, best_acc, best_run]
    best_model.save('best_model.h5')
    losses = [ii['history.val_loss'] for ii in trials.results]
    #acc = [ii['history.val_acc'] for ii in trials.results]
    cv_params = trials.vals
    cv_results = {'best_run': best_result, 'params': cv_params, 'losses': losses}#, 'acc': acc}
    with open('hyperparams.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, val in best_run.items():
            writer.writerow([key, val])
    with open('cv_results.pkl', 'wb') as file:
        pickle.dump(cv_results, file)
    print("Evaluation of best performing model:")
    print(best_loss,)
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print('Time taken:{}'.format((time.time()-t0)/60/60))
