import torch
import torchvision
import torchvision.transforms as transforms
from scipy.ndimage import uniform_filter, sobel
import matplotlib.pyplot as plt
import numpy as np

# Seed everything
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

# 1, 25

# Load ImageNet data
transform = transforms.Compose([transforms.Resize((256, 256)), 
                                transforms.CenterCrop(224),
                                transforms.ToTensor()
                                ])
trainset = torchvision.datasets.ImageFolder(root='/data/datasets/ImageNet-10/train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# Fetch one image
dataiter = iter(trainloader)
images, labels = next(dataiter)
image = images[26]  # Get the first image in the batch

# Convert to grayscale
image_gray = image.mean(0)  # Average across the color channels

# Convert tensor to numpy array for processing with SciPy
image_np = image_gray.numpy()

# Apply Sobel filter
sobel_horizontal = sobel(image_np, axis=0)
sobel_vertical = sobel(image_np, axis=1)
sobel_image = np.hypot(sobel_horizontal, sobel_vertical)

# Apply a mean filter
window_size = 36
smoothed_image = uniform_filter(sobel_image, size=window_size)

# Normalize to get probability heatmap
probability_map = smoothed_image / np.sum(smoothed_image)

# Plotting
fig, ax = plt.subplots(1, 4, figsize=(24, 6))
ax[0].imshow(image.permute(1, 2, 0))  # Convert C, H, W to H, W, C for plotting
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(image_np, cmap='gray')
ax[1].set_title('Grayscale Image')
ax[1].axis('off')

ax[2].imshow(sobel_image, cmap='gray')
ax[2].set_title('Sobel Edge Image')
ax[2].axis('off')

ax[3].imshow(probability_map, cmap='hot')
ax[3].set_title('Probability Heatmap')
ax[3].axis('off')

plt.savefig("saliency_map.png", bbox_inches='tight')
