import torch
import numpy as np
import random
import os
from scipy.spatial import distance_matrix

def seed_everything(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
    return None

class Datasetwithindex(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y, task_id = self.data[index]
        return x, y, index, task_id
    
def intra_cluster_distance(embeddings, assignments):
    """
    Calculates the maximum intra-cluster distance. A lower intra-cluster
    distance is better.
    """

    ### Calculate centroids (mean of each cluster)
    centroids = {}
    for cluster_id in np.unique(assignments):
        cluster_mean = np.mean(embeddings[assignments == cluster_id], axis=0)
        centroids[cluster_id] = cluster_mean

    dists = []
    for cluster_id in np.unique(assignments):
        centroid = centroids[cluster_id]
        datapoints = embeddings[assignments == cluster_id]
        distances = distance_matrix(datapoints, np.expand_dims(centroid, axis=0))
        dists.append(np.sum(distances)/len(datapoints))

    return np.nanmax(dists)

def inter_cluster_distance(embeddings, assignments):
    """
    Calculates the minimum inter-cluster distance. A higher inter-cluster
    distance is better.
    """

    ### Calculate centroids (mean of each cluster)
    centroids = []
    for cluster_id in np.unique(assignments):
        cluster_mean = np.mean(embeddings[assignments == cluster_id], axis=0)
        centroids.append(cluster_mean)
    centroids = np.array(centroids)

    # if there is only 1 centroid, return 0
    if len(centroids) == 1:
        return 0

    ### Calculate distance matrix
    dists = distance_matrix(centroids, centroids)
    remove_self = dists[dists > 0]

    return np.min(remove_self)


def encode_label(labels, classes_list, num_pseudoclasses):
    targets = torch.zeros(len(labels), num_pseudoclasses)
    for i in range(len(labels)):
      label = labels[i]
      idx = classes_list.index(label.item()+1)
      targets[i, idx] = 1
    return targets.to(labels.device)


        

    
