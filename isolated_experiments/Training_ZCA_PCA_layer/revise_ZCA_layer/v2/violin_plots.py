import numpy as np
import matplotlib.pyplot as plt

### Read zca data for each augmentation case
aug_types = ['none', 'colorjitter', 'grayscale', 'gaussianblur', 'solarization']
data = []

for aug in aug_types:
    aug_data = np.load(f'output/{aug}aug_initbias_3kernerlsize_10channels_0.0005eps/zca_out_raw_values.npy')
    aug_data = aug_data.flatten()
    data.append(aug_data)

### Plot violin plots
fig, ax = plt.subplots()
ax.violinplot(data, showmeans=False, showmedians=True)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(aug_types)
plt.savefig('output/violin_plots.png')
plt.close()


### Plot mean, std, min, and max

all_mean = []
all_std = []
all_min = []
all_max = []
for aug in aug_types:
    aug_data = np.load(f'output/{aug}aug_initbias_3kernerlsize_10channels_0.0005eps/zca_out_raw_values.npy')
    mean = np.mean(aug_data)
    std = np.std(aug_data)
    min_val = np.min(aug_data)
    max_val = np.max(aug_data)
    all_mean.append(mean)
    all_std.append(std)
    all_min.append(min_val)
    all_max.append(max_val)

plt.figure()
plt.errorbar(aug_types, all_mean, yerr=all_std, fmt='o')
plt.title('Mean +- std')
plt.savefig('output/mean_std.png')
plt.close()

plt.figure()
plt.plot(aug_types, all_min, 'o')
plt.title('Max Values')
plt.savefig('output/min.png')
plt.close()

plt.figure()
plt.plot(aug_types, all_max, 'o')
plt.title('Min Values')
plt.savefig('output/max.png')
plt.close()
    