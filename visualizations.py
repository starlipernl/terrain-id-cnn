"""
Code for creating plots and visualization of confusion matrices and bayesian visualizations used in reports
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cnf
import cnn


# CONFUSION MATRICES
# loading data
with open('features.pkl', 'rb') as file:
    features = pickle.load(file)
with open('labels.pkl', 'rb') as file:
    labels = pickle.load(file)
feature_list = np.asarray(features)
label_list = np.asarray(labels).reshape((-1, 1))
x_train, x_test, y_train, y_test = train_test_split(feature_list, label_list, test_size=0.2, stratify=label_list)
y_test = y_test.reshape((len(y_test),))
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
val_split = round(x_train.shape[0] * 0.8)
x_valid = x_train[val_split:]
y_valid = y_train[val_split:]
x_train = x_train[0:val_split]
y_train = y_train[0:val_split]

# model params
input_shape = x_train[1].shape
num_classes = 6
NUM_EPOCHS = 1000
BATCH_SIZE = 16
LEARNING_RATE = 0.0075633
NUM_NEURONS_IN_DENSE_1 = 256
DROP_PROB = 0.02076

# build model
model = cnn.build_model(input_shape=input_shape, learn_rate=LEARNING_RATE, drop_prob=DROP_PROB,
                        num_neurons=NUM_NEURONS_IN_DENSE_1)
model.load_weights('Saved_Results/best_model.h5')
test_accuracy, test_loss, predictions = cnn.test_model(model, x_test, y_test)
predictions = np.round(predictions)
predictions = predictions.astype(int)
# Labels
# 0 - sidewalk, 1 - carpet, 2 - brick, 3 - asphalt, 4 - grass, 5 - tile
labels = ['Sidewalk', 'Carpet', 'Brick', 'Asphalt', 'Grass', 'Tile']
predictions_cat = np.argmax(predictions, axis=1)
# compute confusion matrix and normalized confusion matrix
cm = cnf(y_test, predictions_cat)
cm_norm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
# plot confusion matrix
plt.figure()
plt.imshow(cm, cmap=plt.get_cmap('Blues'))
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

# plot normalized confusion matrix
plt.figure()
plt.imshow(cm_norm, cmap=plt.get_cmap('Blues'))
plt.title('Normalized Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm_norm[i, j], '.3f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

################################################################################################################
# BAYESIAN OPTIMIZATION VISUALS
# open results from run_hyperopt_cnn.py
with open('cv_results.pkl', 'rb') as file:
    cv_results = pickle.load(file)
# load results from dict as numpy arrays
eval_idx = range(len(cv_results['losses']))
losses = np.asarray([cv_results['losses'][ii][-1] for ii in eval_idx])
acc = np.asarray([cv_results['acc'][ii][-1] for ii in eval_idx])
learn_rate = np.asarray(cv_results['params']['learn_rate'])
dropout = np.asarray(cv_results['params']['dropout'])
dense_neurons = np.asarray(cv_results['params']['num_neurons'])
num_filters_conv1 = np.asarray(cv_results['params']['conv1_numfilters'])
best_run = cv_results['best_run']
neurons = [64, 128, 256, 512, 1024]
best_neuron = neurons[best_run[3]['num_neurons']]
# convert indices into actual num neurons value
for (idx, num) in enumerate(neurons):
    dense_neurons[dense_neurons == idx] = num
num_filters = [32, 64, 128, 256]
num_filters_best = num_filters[best_run[3]['conv1_numfilters']]
# convert indices into num_filters value
for (idx, num) in enumerate(num_filters):
    num_filters_conv1[num_filters_conv1 == idx] = num

# plot learning rate vs optimization step (time)
plt.figure()
plt.plot(eval_idx, learn_rate, 'bo', best_run[0], best_run[3]['learn_rate'], 'ro',
         eval_idx, np.ones((len(eval_idx), 1)) * best_run[3]['learn_rate'], 'r--')
plt.xlabel('Optimization Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Over Time')
plt.show()

# plot number of dense neurons vs optimization step (time)
plt.figure()
plt.plot(eval_idx, dense_neurons, 'bo', best_run[0], best_neuron, 'ro',
         eval_idx, np.ones((len(eval_idx), 1)) * best_neuron, 'r--')
plt.xlabel('Optimization Step')
plt.ylabel('Number of Neurons')
plt.title('Number of Neurons in Dense Layer Over Time')
plt.show()

# plot dropout probability vs optimization step (time)
plt.figure()
plt.plot(eval_idx, dropout, 'bo', best_run[0], best_run[3]['dropout'], 'ro',
         eval_idx, np.ones((len(eval_idx), 1)) * best_run[3]['dropout'], 'r--')
plt.xlabel('Optimization Step')
plt.ylabel('Dropout Probability')
plt.title('Dropout Probability Over Time')
plt.show()

# plot number of filters in first convolutional layer
plt.figure()
plt.plot(eval_idx, num_filters_conv1, 'bo', best_run[0], num_filters_best, 'ro',
         eval_idx, np.ones((len(eval_idx), 1)) * num_filters_best, 'r--')
plt.xlabel('Optimization Step')
plt.ylabel('Number of Filters')
plt.title('Number of Filters in Convolutional Layer 1 Over Time')
plt.show()


