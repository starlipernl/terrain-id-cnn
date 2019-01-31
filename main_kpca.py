
# ECE 542
# Final Project
# December 2018
# Description: This script runs bayesian hyper-parameter optimization

# Labels
# 0 - sidewalk, 1 - carpet, 2 - brick, 3 - asphalt, 4 - grass, 5 - tile

################################################################################
# IMPORTs

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp
from sklearn.decomposition import PCA, KernelPCA
from hyperopt import Trials, STATUS_OK, tpe
import scipy.io as sio
import csv
import pickle

import time


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

x_valid, y_valid, x_train, y_train, x_test, y_test = data()

ensemble_kernel = []
ensemble_comp = []
ensemble_acc = []

################################################################################
def create_model(params):


    kpca = KernelPCA(kernel=params['kernel']['ktype'], n_components=params['n_components'])
    print('---------------------------------------')
    print('Kernel:  {}'.format(params['kernel']['ktype']))
    ensemble_kernel.append(params['kernel']['ktype'])
    print('N comp:   {}'.format(params['n_components']))
    ensemble_comp.append(params['n_components'])
    kpca.fit(x_train)
    train_img = kpca.transform(x_train)
    X_kpca = kpca.transform(x_valid)

    # Run Random Forest Classifier and plot result if n < 4
    validation_acc = run_rf(train_img, y_train, X_kpca, y_valid, "KPCA-RF ")
    ensemble_acc.append(validation_acc)
    # save for later
    print('Val. Acc:    {}'.format(validation_acc))

    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': kpca}

################################################################################


if __name__ == '__main__':

    t0 = time.time()
    # Limit memory usage of gpu
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # set_session(tf.Session(config=config))
    # loading dataset
    trials = Trials()

    space = {
        'n_components': 1 + hp.randint('n_components', 128),
        'kernel': hp.choice('kpca_kernel', [
            {'ktype': 'linear'},
            {'ktype': 'poly', 'gamma': hp.uniform('poly_gamma', 0.001, x_train.shape[1]),
             'degree': hp.randint('poly_degree', 10), 'coef0': hp.uniform('poly_coef0', 0, 1)},
            {'ktype': 'rbf', 'gamma': hp.uniform('rbf_gamma', 0.001, x_train.shape[1])},
            {'ktype': 'sigmoid', 'gamma': hp.uniform('k_gamma', 0.001, x_train.shape[1]),
             'coef0': hp.uniform('sigmoid_coef0', 0, 1)},
            {'ktype': 'cosine'},
        ])
    }

    evals = 400
    best = fmin(fn=create_model, space=space, algo=tpe.suggest, max_evals=evals, trials=trials)

    arr_ensemble_acc = np.asarray(ensemble_acc)
    arr_ensemble_comp = np.asarray(ensemble_comp)

    print(best)
    print('++++++++++++++++++++++++++++++++++++++')
    print('Double checking...')
    print('Maximum accuracy:    {}'.format(max(ensemble_acc)))
    print('Best evaluation number:  {}'.format(ensemble_acc.index(max(ensemble_acc))))
    print('Respective Kernel:    {}'.format(ensemble_kernel[ensemble_acc.index(max(ensemble_acc))]))
    print('Respective No. comp.:    {}'.format(ensemble_comp[ensemble_acc.index(max(ensemble_acc))]))

    sio.savemat(str(evals) + '_exp_' + 'scores_kpca_evals' + '.mat',
                {'arr_ensemble_acc': arr_ensemble_acc, 'arr_ensemble_comp': arr_ensemble_comp})

    print('Time taken:{}'.format((time.time()-t0)/60/60))




