import numpy as np
import matplotlib.pyplot as plt

###
# CLUSTERING ACCURACY PLOTS
###

# Read data 
ACC_2views = np.load('./../output/fromScratch_50epochs_2views_0.25lr_128bs/ACC.npy')
ACC_4views = np.load('./../output/fromScratch_50epochs_4views_0.25lr_128bs/ACC.npy')
ACC_8views = np.load('./../output/fromScratch_50epochs_8views_0.25lr_128bs/ACC.npy')
ACC_12views = np.load('./../output/fromScratch_50epochs_12views_0.25lr_128bs/ACC.npy')

# add value for epoch 0, which is 0.0184
ACC_2views = np.insert(ACC_2views, 0, 0.0184)
ACC_4views = np.insert(ACC_4views, 0, 0.0184)
ACC_8views = np.insert(ACC_8views, 0, 0.0184)
ACC_12views = np.insert(ACC_12views, 0, 0.0184)

# Plot curves
plt.plot(ACC_2views, label='2 views', linewidth=2.5)
plt.plot(ACC_4views, label='4 views', linewidth=2.5)
plt.plot(ACC_8views, label='8 views', linewidth=2.5)
plt.plot(ACC_12views, label='12 views', linewidth=2.5)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Clustering Accuracy', fontsize=14)
plt.xticks(np.arange(0, 11, 1), fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0,0.25)
plt.legend(fontsize=12)
plt.grid()
plt.savefig('ACC_curves_views.png', bbox_inches='tight')
plt.close()