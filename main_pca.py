import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
#from matplotlib import pyplot as plt
import scipy.io as sio
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

    x_train = x_train[0:val_split]
    y_train = y_train[0:val_split]

    # Normalizing images
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

    pca = PCA().fit(x_train)
    #plt.plot(np.cumsum(pca.explained_variance_ratio_))
    #plt.xlabel('number of components')
    #plt.ylabel('cumulative explained variance')
    #plt.savefig('pca_variance_explained.png')

    n_comp = []
    val_acc = []
    NumberOfExperiments = 400
    for k in range(0, NumberOfExperiments):

        x_train, x_test, y_train, y_test = train_test_split(feature_list, label_list, test_size=0.2)
        y_test = y_test.reshape((len(y_test),))

        val_split = round(x_train.shape[0] * 0.8)
        x_valid = x_train[val_split:]
        y_valid = y_train[val_split:]
        x_train = x_train[0:val_split]
        y_train = y_train[0:val_split]

        # Normalizing images
        x_train = x_train.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

        x_valid = x_valid.astype('float32') / 255.
        x_valid = x_valid.reshape((len(x_valid), np.prod(x_valid.shape[1:])))

        x_test = x_test.astype('float32') / 255.0
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        pca = PCA(0.9).fit(x_train)
        train_img = pca.transform(x_train)
        X_kpca = pca.transform(x_valid)
        n_comp.append(pca.n_components_)
        # Run Random Forest Classifier and plot result if n < 4
        validation_acc = run_rf(train_img, y_train, X_kpca, y_valid, "PCA-RF ")
        val_acc.append(validation_acc)
        # save for later
    arr_n_comp = np.asarray(n_comp)
    arr_val_acc = np.asarray(val_acc)
    sio.savemat('exp_'+str(NumberOfExperiments)+'scores_pca'+'.mat', {'arr_n_comp': arr_n_comp, 'arr_val_acc': arr_val_acc})
    print('Best Val. Acc:    {}'.format(max(val_acc)))
    print('Respective No. components:   {}'.format(n_comp[val_acc.index(max(val_acc))]))


