import torch
import einops
import numpy as np

def calculate_ZCA_weights(imgs, kernel_size=3, zca_epsilon=1e-6):
    # extract Patches 
    patches = extract_patches(imgs, kernel_size, step=kernel_size)
    # get weight with ZCA
    weight = get_filters(patches, zca_epsilon=zca_epsilon)
    return weight

def extract_patches(images,window_size,step):
    n_channels = images.shape[1]
    aux = images.unfold(2, window_size, step)
    aux = aux.unfold(3, window_size, step)
    patches = aux.permute(0, 2, 3, 1, 4, 5).reshape(-1, n_channels, window_size, window_size)
    return patches

def get_filters(patches_data, zca_epsilon=1e-6):
    _, n_channels, filt_size, _ = patches_data.shape
    data = einops.rearrange(patches_data, 'n c h w -> (c h w) n') # data is a d x N
    data = data.double()

    # Compute ZCA
    W = ZCA(data, epsilon = zca_epsilon).to(data.device) # shape: k (c h w)

    # Reshape filters
    W = einops.rearrange(W, 'k (c h w) -> k c h w', c=n_channels, h=filt_size, w=filt_size)

    # Pick centered filters only (filter is in the middle of the patch)
    W_center = torch.zeros((n_channels, n_channels, filt_size, filt_size), dtype=W.dtype)
    for i in range(n_channels):
        index = int( (filt_size*filt_size)*(i) + (filt_size*filt_size-1)/2 ) # index of the filter that is in the center of the patch
        W_center[i, :] = W[index, :, :, :]

    # back to single precision
    W_center = W_center.float()

    return W_center

def ZCA(data, epsilon=1e-6):
    # data is a d x N matrix, where d is the dimensionality and N is the number of samples
    C = (data @ data.T) / data.size(1)
    D, V = torch.linalg.eigh(C)
    sorted_indices = torch.argsort(D, descending=True)
    D = D[sorted_indices]
    V = V[:, sorted_indices]
    Di = (D + epsilon)**(-0.5)
    zca_matrix = V @ torch.diag(Di) @ V.T
    return zca_matrix

def scaled_filters(zca_layer, imgs, prob_threshold=0.99999):
    # Maxscale the filters
    weight = zca_layer.weight
    weight = weight / torch.amax(weight, keepdims=True)
    zca_layer.weight = torch.nn.Parameter(weight)
    zca_layer.weight.requires_grad = False

    # Get the output of the zca layer
    zca_layer.eval()
    zca_layer_output = zca_layer(imgs)

    # Get CDF on the absolute values of the output
    zca_layer_output = zca_layer_output.abs().flatten().numpy()
    count, bins_count = np.histogram(zca_layer_output, bins=100) 
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)

    # Find threshold, multiplier, and scale the filters
    threshold = bins_count[1:][np.argmax(cdf>prob_threshold)]
    multiplier = np.arctanh(0.999)/threshold # find value where after a tanh function, everythinng after the threshold will be near 1 (0.999)
    print(f'threshold: {threshold}')
    print(f'multiplier: {multiplier}')
    weight = weight * multiplier

    return weight