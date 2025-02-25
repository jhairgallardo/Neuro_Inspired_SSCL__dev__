import numpy as np
import random

seed = 0
random.seed(seed)
np.random.seed(seed)

num_classes = 10
data_class_order = np.random.permutation(num_classes) + 1
np.savetxt(f'IM10_data_class_order{seed}.txt', data_class_order, fmt='%d')

# num_classes = 10
# data_class_order = np.arange(10) + 1
# np.savetxt(f'IM10_data_class_orderoriginal.txt', data_class_order, fmt='%d')
