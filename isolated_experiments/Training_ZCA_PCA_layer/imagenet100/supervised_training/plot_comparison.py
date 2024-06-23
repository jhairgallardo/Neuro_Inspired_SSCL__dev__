import os
import numpy as np
import matplotlib.pyplot as plt

plots_path = 'plots_results/'
os.makedirs(plots_path, exist_ok=True)


experiments_path = 'output/'
# Load saved stats standard
standard_expt = 'resnet18'
standard_trainloss = np.load(f'{experiments_path}/{standard_expt}/train_loss_all.npy')
standard_trainacc = np.load(f'{experiments_path}/{standard_expt}/train_accuracy_all.npy')
standard_valloss = np.load(f'{experiments_path}/{standard_expt}/val_loss_all.npy')
standard_valacc = np.load(f'{experiments_path}/{standard_expt}/val_accuracy_all.npy')
# Load saved stats zca
zca_expt = 'resnet18_zca_eps0.0005'
zca_trainloss = np.load(f'{experiments_path}/{zca_expt}/train_loss_all.npy')
zca_trainacc = np.load(f'{experiments_path}/{zca_expt}/train_accuracy_all.npy')
zca_valloss = np.load(f'{experiments_path}/{zca_expt}/val_loss_all.npy')
zca_valacc = np.load(f'{experiments_path}/{zca_expt}/val_accuracy_all.npy')
# Load saved stats pca
pca_expt = 'resnet18_pca_eps0.0005'
pca_trainloss = np.load(f'{experiments_path}/{pca_expt}/train_loss_all.npy')
pca_trainacc = np.load(f'{experiments_path}/{pca_expt}/train_accuracy_all.npy')
pca_valloss = np.load(f'{experiments_path}/{pca_expt}/val_loss_all.npy')
pca_valacc = np.load(f'{experiments_path}/{pca_expt}/val_accuracy_all.npy')

# Plot Losses
plt.figure()
plt.plot(standard_valloss, label='Standard', color='red', linewidth=2)
plt.plot(zca_valloss, label='ZCA (10 filters)', color='blue', linewidth=2)
plt.plot(pca_valloss, label='PCA (54 filters)', color='green', linewidth=2)
plt.plot(standard_trainloss, alpha=0.4, color='red')
plt.plot(zca_trainloss, alpha = 0.4, color='blue')
plt.plot(pca_trainloss, alpha = 0.4, color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Losses')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f'{plots_path}/losses.png', bbox_inches='tight')
plt.close()

# Plot Accuracies
plt.figure()
plt.plot(standard_valacc, label='Standard', color='red', linewidth=2)
plt.plot(zca_valacc, label='ZCA (10 filters)', color='blue', linewidth=2)
plt.plot(pca_valacc, label='PCA (54 filters)', color='green', linewidth=2)
plt.plot(standard_trainacc, alpha=0.4, color='red')
plt.plot(zca_trainacc, alpha = 0.4, color='blue')
plt.plot(pca_trainacc, alpha = 0.4, color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracies')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f'{plots_path}/accuracies.png', bbox_inches='tight')
plt.close()

