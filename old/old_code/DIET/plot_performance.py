import matplotlib.pyplot as plt

epochs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
iid_acc = [40.11, 44.21, 45.95, 47.81, 48.55, 49.65, 49.67, 50.14, 50.68, 50.60]
non_iid_acc = [25.07, 32.84, 35.71, 37.32, 38.88, 40.58, 40.75, 41.45, 41.31, 40.17]

# Plot performance with nice seaborn style
plt.style.use('seaborn')
plt.plot(epochs, iid_acc, '-o', label='IID')
plt.plot(epochs, non_iid_acc, '-o', label='Non-IID')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('DIET CIFAR-100 Linear Evaluation')
plt.legend()
plt.savefig('diet_cifar100_linear_eval.png', dpi=300, bbox_inches='tight')