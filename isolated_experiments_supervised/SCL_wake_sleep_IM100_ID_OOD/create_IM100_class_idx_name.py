# Create a dictionary for ImageNet100 with class index as key as a list [class_folder_name,class_human_readable_name] as value.
# I can read the mapping from class_folder_name to class_human_readable_name from /data/datasets/ImageNet2012/imagenet_class_index.json 
# which is a json file with the same format I mentioned above but for the full ImageNet dataset.

import json

import torch
from torchvision import datasets

# Step 1: Let's create a mapping from class_folder_name to class_human_readable_name from the complete ImageNet dataset.
# for this. we can use the /data/datasets/ImageNet2012/imagenet_class_index.json.

with open('/data/datasets/ImageNet2012/imagenet_class_index.json', 'r') as f:
    imagenet_class_index = json.load(f)

classfolder2humanreadable = {}
for class_idx, class_info in imagenet_class_index.items():
    class_folder_name = class_info[0]
    class_human_readable_name = class_info[1]
    classfolder2humanreadable[class_folder_name] = class_human_readable_name

# Step 2: Let's create a mapping from class_index to class_folder_name and class_human_readable_name for the ImageNet100 dataset
# Let's get class_idx and folder name from the ImageNet100 dataset with Dataset class

ImageNet100_path = '/data/datasets/ImageNet-100/train'
ImageNet100_dataset = datasets.ImageFolder(ImageNet100_path)
classfolder2idx = ImageNet100_dataset.class_to_idx

# Step 3: Finally, let's create imagenet100_class_index.json file with the following format:
# {class_idx: [class_folder_name, class_human_readable_name]}

imagenet100_class_index = {}
for class_folder_name, class_idx in classfolder2idx.items():
    imagenet100_class_index[class_idx] = [class_folder_name, classfolder2humanreadable[class_folder_name]]

with open('imagenet100_class_index.json', 'w') as f:
    json.dump(imagenet100_class_index, f)

print('end')