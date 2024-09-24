import numpy as np
from scipy.spatial import distance_matrix

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

    ### Calculate distance matrix
    dists = distance_matrix(centroids, centroids)
    remove_self = dists[dists > 0]

    return np.min(remove_self)


        

    
