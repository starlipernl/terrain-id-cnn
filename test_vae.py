import numpy as np
from keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from tensorflow.keras import Model, Sequential
from tensorflow.keras.activations import relu, tanh, elu
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.losses import mse, binary_crossentropy
from keras.datasets import mnist
import pickle
from hyperopt import Trials, STATUS_OK, tpe
from sklearn.ensemble import RandomForestClassifier
from hyperas.distributions import choice, uniform, lognormal
import scipy.io as sio

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

def create_model(x_train, x_test, y_test, encoding_dim, intermediate_dim, epochs):
    # ---------------------
    # Run Random Forest Classifier
    # ---------------------
    def run_rf(train_img, y_train, test_img, y_test, msg):
        clf = RandomForestClassifier(50)
        clf.fit(train_img, np.squeeze(y_train))

        print(msg + 'Score')
        print(clf.score(test_img, np.squeeze(y_test)))

        return clf.score(test_img, np.squeeze(y_test))


    def sampling(args):
        # construct the search space
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
    batch_size = 32
    latent_dim = encoding_dim
    act_fncs = elu
    learn_rate = 0.0026607621768993824
    lr_decay = 0.0021721614264192577

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
    encoder.summary()
    # plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation=act_fncs)(latent_inputs)
    outputs = Dense(original_dim, activation=act_fncs)(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
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
    valid_steps = x_test.shape[0]//batch_size

    result = vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            verbose=1,
            batch_size=batch_size,
            validation_data=(x_test, None))

    encoder = Model(inputs, z_mean)
    z_test = encoder.predict(x_test, batch_size=batch_size)
    train_img = encoder.predict(x_train, batch_size=batch_size)
    # take max validation accuracy as metric

    # Run Random Forest Classifier and plot result if n < 4
    ok = run_rf(train_img, y_train, z_test, y_test, "VAE-RF ")


    return ok, z_test, train_img, result.history['loss'], result.history['val_loss']


if __name__ == '__main__':

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

    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    ensemble_score = []
    ensemble_val_loss = []
    # Normalizing images
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

    x_valid = x_valid.astype('float32') / 255.
    x_valid = x_valid.reshape((len(x_valid), np.prod(x_valid.shape[1:])))

    x_test = x_test.astype('float32') / 255.0
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    encoding_dim = 64
    intermediate_dim = 128
    epch = 250
    NumberOfExperiments = 50
    for k in range(0, NumberOfExperiments):
        print('++++++++++++++++++++++++++++++++++++++++++')
        print('Run: {}'.format(k))
        ok, z_test, train_img, loss, val_loss = create_model(x_train, x_test, y_test, encoding_dim, intermediate_dim, epch)
        ensemble_score.append(ok)
        ensemble_val_loss.append(val_loss)
    arr_ensemble_score = np.asarray(ensemble_score)
    arr_val_loss = np.asarray(ensemble_val_loss)
    sio.savemat(str(epch)+'_exp_'+str(NumberOfExperiments)+'scores_vae_epch_'+'.mat', {'arr_ensemble_score': arr_ensemble_score, 'arr_val_loss': arr_val_loss})
