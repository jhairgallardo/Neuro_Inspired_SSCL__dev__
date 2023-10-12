import os
import torch
from torchvision.models import *
import torchvision.transforms as Transforms
import torchvision.datasets as Datasets

import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
sns.set_style("whitegrid")

### Seed everything
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
sklearn.utils.check_random_state(seed)

datapath = '/data/datasets/CIFAR100/'
arch = 'resnet18'
num_classes = 100
batch_szie = 256
num_workers = 4
gpu = 0
pretrained_folder = './experiments/non_iid/230913_233206' # './experiments/iid/230913_232941'
pretrained_network = 'resnet18_encoder.pth'

### Load val data
val_transform = Transforms.Compose([
            Transforms.ToTensor(),
            Transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
        ])
val_dataset = Datasets.CIFAR100(root=datapath, train=False, download=True, transform=val_transform)
val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_szie, shuffle=False,
        num_workers=num_workers, pin_memory=True)

### Load Model
device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
model = eval(arch)(num_classes=num_classes)
# Edit conv1 for CIFAR100
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
model.maxpool = torch.nn.Identity()
state_dict = torch.load(os.path.join(pretrained_folder, pretrained_network))
missing_keys, unexpected_keys = model.load_state_dict(state_dict['state_dict'], strict=False)
assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
model.fc = torch.nn.Identity() # erase classifier so I can get feature representations
model = model.to(device)

### Get embeddings
model.eval()
embeddings = []
val_labels = []
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        e = model(data)
        embeddings.append(e.cpu().numpy())
        val_labels.append(target.cpu().numpy())

embeddings = np.concatenate(embeddings, axis=0)
val_labels = np.concatenate(val_labels, axis=0)
print(embeddings.shape)
print(val_labels.shape)

### Apply PCA to embeddings
pca = PCA(n_components=100) #20
pca.fit(embeddings)
pca_embeddings = pca.transform(embeddings)
print(pca_embeddings.shape)

### Apply t-SNE to embeddings
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=2)
tsne_embeddings = tsne.fit_transform(pca_embeddings)
print(tsne_embeddings.shape)

colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(val_labels))))

plt.figure(figsize=(6, 6))
for i, category in enumerate(np.unique(val_labels)):
    plt.scatter(tsne_embeddings[val_labels==category, 0], 
                tsne_embeddings[val_labels==category, 1], 
                # label=class_number2name[str(category)], 
                s=10,
                color=colors[i])
# legend outside plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, markerscale=3)
# save tight layout
plt.savefig(os.path.join(pretrained_folder,f'tsne_{pretrained_network}.png'), bbox_inches='tight', dpi=300)


# ### Plot embeddings
# # map labels numbers to name
# test_labels_numpy = alldata['test_labels'].squeeze()
# test_labels_names = np.array([class_number2name[str(i)] for i in test_labels_numpy])
# # plot
# plt.figure(figsize=(10, 10))
# plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=test_labels_names, cmap='tab10')
# plt.colorbar()
# plt.savefig('tsne.png')


