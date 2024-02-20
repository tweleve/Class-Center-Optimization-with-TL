#! /usr/bin/env python3
import sys
import logging
import faiss
import torch
import numpy as np

# modified from https://github.com/facebookresearch/deepcluster
def get_knn(    reference_embeddings, test_embeddings, k, embeddings_come_from_same_source=False):
    """
    Finds the k elements in reference_embeddings that are closest to each
    element of test_embeddings.
    Args:
        reference_embeddings: numpy array of size (num_samples, dimensionality).
        test_embeddings: numpy array of size (num_samples2, dimensionality).
        k: int, number of nearest neighbors to find
        embeddings_come_from_same_source: if True, then the nearest neighbor of
                                         each element (which is actually itself)
                                         will be ignored.

    查找reference_embeddings中最接近test_embeddings的每个元素的 k 个元素。
    参数：
        reference_embeddings：大小（num_samples，维度）的numpy数组。
        test_embeddings：大小（num_samples2，维度）的numpy数组。
        k：int，要查找的最近邻数
        embeddings_come_from_same_source：如果为 True，则每个元素的最近邻居（实际上是其自身）将被忽略。
    """
    d = reference_embeddings.shape[1]
    logging.info("running k-nn with k=%d"%k)
    logging.info("embedding dimensionality is %d"%d)
    index = faiss.IndexFlatL2(d)
    if faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(reference_embeddings)
    _, indices = index.search(test_embeddings, k + 1)
    if embeddings_come_from_same_source:
        return indices[:, 1:]
    return indices[:, :k]


# modified from https://raw.githubusercontent.com/facebookresearch/deepcluster/
def run_kmeans(x, nmb_clusters):
    """
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape
    logging.info("running k-means clustering with k=%d"%nmb_clusters)
    logging.info("embedding dimensionality is %d"%d)

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    index = faiss.IndexFlatL2(d)
    if faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(index)
    # perform the training
    clus.train(x, index)
    _, idxs = index.search(x, 1)

    return [int(n[0]) for n in idxs]


# modified from https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
def run_pca(x, output_dimensionality):
    mat = faiss.PCAMatrix(x.shape[1], output_dimensionality)
    mat.train(x)
    assert mat.is_trained
    return mat.apply_py(x)