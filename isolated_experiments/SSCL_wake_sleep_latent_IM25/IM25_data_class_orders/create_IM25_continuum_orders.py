import numpy as np
import random

seed = 0
random.seed(seed)
np.random.seed(seed)

num_classes = 25
data_class_order = np.random.permutation(num_classes) + 1
np.savetxt(f'IM25_data_class_order{seed}.txt', data_class_order, fmt='%d')

# num_classes = 25
# data_class_order = np.arange(25) + 1
# np.savetxt(f'IM25_data_class_orderoriginal.txt', data_class_order, fmt='%d')
