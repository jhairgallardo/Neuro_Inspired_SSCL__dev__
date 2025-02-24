# coding: utf-8
# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random
from sklearn import metrics
import torch
from scipy.optimize import linear_sum_assignment

def evaluate(label, pred, n_clusters, calc_acc=False, total_probs=None, ):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ami = metrics.adjusted_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    purity = purity_score(label, pred, n_clusters)
    hmg = metrics.homogeneity_score(label, pred)
    cm = metrics.completeness_score(label, pred)
    cfi = class_fragmentation_index(label, pred)
    if not calc_acc:
        return nmi, ami, ari, f, -1, -1, -1, -1, purity, hmg, cm, cfi
    if total_probs is not None:
        acc, match, reordered_preds, top5 = hungarian_evaluate(torch.Tensor(label).cuda(), torch.Tensor(pred).cuda(), torch.Tensor(total_probs).cuda())
        return nmi, ami, ari, f, acc, match, reordered_preds.cpu().detach().numpy(), top5, purity, hmg, cm, cfi
    else:
        acc, match, reordered_preds = hungarian_evaluate(torch.Tensor(label).cuda(), torch.Tensor(pred).cuda(), total_probs)
        return nmi, ami, ari, f, acc, match, reordered_preds.cpu().detach().numpy(), -1, purity, hmg, cm, cfi


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels

# evaluate function modified from SCAN
@torch.no_grad()
def hungarian_evaluate(targets, predictions, total_probs, class_names=None, compute_purity=True, compute_confusion_matrix=False, confusion_matrix_file='confusion.pdf', percent=[1.0]):
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)
    
    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    # np.save('imagenet_match.npy', np.array(match))
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)

    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    # print("Using {} Samples to Estimate Pseudo2Real Label Mapping, Acc:{:.4f}".format(int(num_elems), acc))

    if total_probs is not None:
        _, preds_top5 = total_probs.topk(5, 1, largest=True)
        reordered_preds_top5 = torch.zeros_like(preds_top5)
        for pred_i, target_i in match:
            reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
        correct_top5_binary = reordered_preds_top5.eq(targets.view(-1,1).expand_as(reordered_preds_top5))
        top5 = float(correct_top5_binary.sum()) / float(num_elems)
        # print("Using {} Samples to Estimate Pseudo2Real Label Mapping, Acc Top-5 :{:.4f}".format(int(num_elems), top5))

    ## Compute confusion matrix
    if compute_confusion_matrix:
        confusion_matrix(reordered_preds.cpu().numpy(), targets.cpu().numpy(), class_names, confusion_matrix_file)
    if total_probs is not None:
        return acc, match, reordered_preds, top5
    else:
        return acc, match, reordered_preds


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res

def confusion_matrix(predictions, gt, class_names, output_file='confusion.pdf'):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)

    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def purity_score(y_true, y_pred, n_clusters):
    labels = np.unique(y_true)
    confusion = np.zeros((len(labels), n_clusters), dtype=int)
    # Map each label to a row index
    label_to_row = {label: idx for idx, label in enumerate(labels)}
    # Fill the confusion matrix
    for t, p in zip(y_true, y_pred):
        row_idx = label_to_row[t]
        confusion[row_idx, p] += 1
    max_in_each_cluster = confusion.max(axis=0)
    return np.sum(max_in_each_cluster) / len(y_true)

def class_fragmentation_index(y_true, y_pred):
    unique_labels = np.unique(y_true)
    # For each label, gather which clusters it occupies
    counts = []
    for label in unique_labels:
        # clusters that contain at least one sample of `label`
        clusters_for_label = np.unique(y_pred[y_true == label])
        counts.append(len(clusters_for_label))
    # Average number of clusters per label
    return np.mean(counts)
