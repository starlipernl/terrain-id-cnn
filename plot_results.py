import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
import time

'''
#VAE best parameters (manually input)
batch_size = 32
act_fncs = 'elu'
learn_rate = 0.0026607621768993824
lr_decay = 0.0021721614264192577
encoding_dim = 64
intermediate_dim = 128

vae_table = sio.loadmat('vaedata.mat')
vaedata = vae_table['vaedata']
tit_text = ['Latent_Vector_Size', 'Batch_Size', 'Learning_Decay', 'Learning_Rate', 'Accuracy']

print('Printing VAE hyperparms search results..')
for k in range(0, vaedata.shape[1]):
    plt.figure()
    plt.plot(np.reshape(vaedata[:, k], (vaedata.shape[0], 1)), 'bo')

    if k == 0:
        plt.plot(encoding_dim * np.ones((vaedata.shape[0], 1)), '--r', label='Best value')
        plt.plot(258, encoding_dim, 'or')
        #found manually
        plt.ylabel('Latent Vector Size')
    elif k == 1:
        plt.plot(batch_size * np.ones((vaedata.shape[0], 1)), '--r', label='Best value')
        #found manually
        plt.plot(258, batch_size, 'or')
        plt.ylabel('Batch Size')
    elif k == 2:
        plt.plot(lr_decay * np.ones((vaedata.shape[0], 1)), '--r', label='Best value')
        #found manually
        plt.plot(258, lr_decay, 'or')
        plt.ylabel('Learning Decay')
    elif k == 3:
        plt.plot(learn_rate * np.ones((vaedata.shape[0], 1)), '--r', label='Best value')
        #found manually
        plt.plot(258, learn_rate, 'or')
        plt.ylabel('Learning Rate')
    elif k == 4:
        max_score = np.amax((vaedata[:, k]))
        plt.plot(max_score * np.ones((vaedata.shape[0], 1)), '--r', label='Best value')
        #found manually
        plt.plot(258, max_score, 'or')        
        plt.ylabel('Accuracy')

    plt.legend()
    plt.title(tit_text[k] + ' Over Time')
    plt.xlabel('Optimization Step')
    plt.savefig(tit_text[k] + '.png')
    plt.close()

print('Printing VAE 5k run results..')
datafile = ['epch_5000_exp_1_scores_vae_1', 'epch_5000_exp_1_scores_vae_2', 'epch_5000_exp_1_scores_vae_3',
            'epch_5000_exp_1_scores_vae_4']

for k in range(0, len(datafile)):
    arr_ensemble_score = sio.loadmat(datafile[k]+'.mat')['arr_ensemble_score']
    arr_val_loss = sio.loadmat(datafile[k]+'.mat')['arr_val_loss']
    mean_score = np.mean(arr_ensemble_score)
    max_score = np.amax(arr_ensemble_score)
    min_val_loss = np.amin(arr_val_loss)

    plt.figure()
    plt.plot(np.reshape(np.arange(0, arr_val_loss.shape[1], 1), (1, arr_val_loss.shape[1])), arr_val_loss, 'b-o')
    plt.title('Loss Evaluation Over Time, Min: ' + str(min_val_loss))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('loss_epoch'+str(k)+'.png')
    plt.close()

'''
print('Printing VAE 50 experiments of 400 epochs each..')
datafile = '250_exp_50scores_vae_epch_'

arr_ensemble_score = sio.loadmat(datafile+'.mat')['arr_ensemble_score']
arr_val_loss = sio.loadmat(datafile+'.mat')['arr_val_loss']
mean_score = np.mean(arr_ensemble_score)
max_score = np.amax(arr_ensemble_score)
min_val_loss = np.amin(arr_val_loss)

plt.figure()

plt.plot(np.reshape(np.arange(0, arr_ensemble_score.shape[1]), (1, arr_ensemble_score.shape[1])), arr_ensemble_score, 'b-o')
plt.plot(max_score * np.ones((arr_ensemble_score.shape[1], 1)), '--r', label='Best value')
plt.plot(mean_score * np.ones((arr_ensemble_score.shape[1], 1)), '--r', label='Average value')
#found manually
plt.plot(21, max_score, 'or')
plt.title('Score VAE, Max Acc = {}, Average = {}'.format(max_score.round(decimals=3), mean_score.round(decimals=3)))
plt.legend()
plt.xlabel('Architecture #')
plt.ylabel('Accuracy')
plt.savefig('arch_acc.png')
time.sleep(5)
plt.close()

plt.figure()
plt.plot(arr_val_loss)
plt.title('Minimum loss ' + str(min_val_loss))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('loss_epoch.png')
plt.close()

#KPCA
#KPCA best parameters (manually input)

coef0 = 0.39287172354078076
gamma = 2034.9926453230612
degree = 3
n_components = 15
kpca_kernel = 'poly'
best_acc = 0.8081123244929798
print('Printing KPCA hyperameter results..')
arr_ensemble_acc = sio.loadmat('400_exp_scores_kpca_evals.mat')['arr_ensemble_acc']
arr_ensemble_comp = sio.loadmat('400_exp_scores_kpca_evals.mat')['arr_ensemble_comp']

plt.figure()
plt.plot(np.reshape(np.arange(0, arr_ensemble_acc.shape[1], 1), (1, arr_ensemble_acc.shape[1])), arr_ensemble_acc,
         'b-o')
plt.plot(best_acc*np.ones((arr_ensemble_comp.shape[1], 1)), 'r--', label='Best value')
#found manually
plt.plot(340, best_acc, 'or')
plt.legend()
plt.xlabel('Optimization Step')
plt.ylabel('Accuracy')
plt.savefig('kpca_eval_acc.png')
plt.close()

plt.figure()
plt.plot(np.reshape(np.arange(0, arr_ensemble_comp.shape[1], 1), (1, arr_ensemble_comp.shape[1])), arr_ensemble_comp,
         'b-o')
plt.plot(n_components*np.ones((arr_ensemble_comp.shape[1], 1)), 'r--', label='Best value')
plt.plot(340, n_components, 'or')
plt.legend()
plt.xlabel('Optimization Step')
plt.ylabel('Kernel PCA No. of Principal Components]')
plt.savefig('kpca_ncomp_acc.png')
plt.close()

#PCA
#PCA best parameters (manually input)

best_acc = 0.8174726989079563
n_components = 79

print('Printing PCA hyperameter results..')
arr_val_acc = sio.loadmat('exp_400scores_pca.mat')['arr_val_acc']
arr_n_comp = sio.loadmat('exp_400scores_pca.mat')['arr_n_comp']

plt.figure()
plt.plot(np.reshape(np.arange(0, arr_val_acc.shape[1], 1), (1, arr_val_acc.shape[1])), arr_val_acc,
         'b-o')
plt.plot(best_acc*np.ones((arr_val_acc.shape[1], 1)), 'r--', label='Best value')
plt.plot(119, best_acc, 'or')

plt.legend()
plt.xlabel('Evaluation #')
plt.ylabel('Accuracy [%]')
plt.savefig('PCA_eval_acc.png')
plt.close()

plt.figure()
plt.plot(np.reshape(np.arange(0, arr_n_comp.shape[1], 1), (1, arr_n_comp.shape[1])), arr_n_comp,
         'b-o')
plt.plot(n_components*np.ones((arr_n_comp.shape[1], 1)), 'r--', label='Best value')
plt.plot(119, n_components, 'or')
plt.legend()
plt.xlabel('Evaluation #')
plt.ylabel('PCA no. of components]')
plt.savefig('PCA_ncomp_acc.png')
plt.close()

