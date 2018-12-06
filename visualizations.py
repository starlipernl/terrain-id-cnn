import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('cv_results.pkl', 'rb') as file:
    cv_results = pickle.load(file)

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
for (idx, num) in enumerate(neurons):
    dense_neurons[dense_neurons == idx] = num
num_filters = [32, 64, 128, 256]
num_filters_best = num_filters[best_run[3]['conv1_numfilters']]
for (idx, num) in enumerate(num_filters):
    num_filters_conv1[num_filters_conv1 == idx] = num
# plt.figure(1)
# plt.plot(eval_idx, losses)
# plt.xlabel('Optimization Search Step')
# plt.ylabel('Validation Loss')
# plt.title('Validation Loss During Search')
# plt.show()
#
# plt.figure(2)
# plt.plot(eval_idx, acc)
# plt.xlabel('Optimization Search Step')
# plt.ylabel('Validation Accuracy')
# plt.title('Validation Accuracy During Search')
# plt.show()
#
# plt.figure(3)
# plt.plot(dropout, acc, 'bo', best_run[3]['dropout'], best_run[2], 'ro', label='best')
# plt.xlabel('Dropout Probability')
# plt.ylabel('Validation Acc')
# plt.title('Dropout Probability Selection')
# plt.show()
#
# plt.figure(4)
# plt.plot(learn_rate, acc, 'bo', best_run[3]['learn_rate'], best_run[2], 'ro')
# plt.xlabel('Learning Rate')
# plt.ylabel('Validation Acc')
# plt.title('Learning Rate Selection')
# plt.show()
#
# plt.figure(5)
# plt.plot(dense_neurons, acc, 'bo', best_run[3]['num_neurons'], best_run[2], 'ro', label='best')
# plt.xlabel('Number of Neurons in Dense Layer')
# plt.ylabel('Validation Acc')
# plt.title('Number of Dense Layer Neurons Selection')
# plt.show()

plt.figure(6)
plt.plot(eval_idx, learn_rate, 'bo', best_run[0], best_run[3]['learn_rate'], 'ro',
         eval_idx, np.ones((len(eval_idx), 1)) * best_run[3]['learn_rate'], 'r--')
plt.xlabel('Optimization Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Over Time')
plt.show()

plt.figure(7)
plt.plot(eval_idx, dense_neurons, 'bo', best_run[0], best_neuron, 'ro',
         eval_idx, np.ones((len(eval_idx), 1)) * best_neuron, 'r--')
plt.xlabel('Optimization Step')
plt.ylabel('Number of Neurons')
plt.title('Number of Neurons in Dense Layer Over Time')
plt.show()

plt.figure(8)
plt.plot(eval_idx, dropout, 'bo', best_run[0], best_run[3]['dropout'], 'ro',
         eval_idx, np.ones((len(eval_idx), 1)) * best_run[3]['dropout'], 'r--')
plt.xlabel('Optimization Step')
plt.ylabel('Dropout Probability')
plt.title('Dropout Probability Over Time')
plt.show()

plt.figure(9)
plt.plot(eval_idx, num_filters_conv1, 'bo', best_run[0], num_filters_best, 'ro',
         eval_idx, np.ones((len(eval_idx), 1)) * num_filters_best, 'r--')
plt.xlabel('Optimization Step')
plt.ylabel('Number of Filters')
plt.title('Number of Filters in Convolutional Layer 1 Over Time')
plt.show()