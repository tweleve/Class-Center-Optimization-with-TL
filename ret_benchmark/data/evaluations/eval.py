#! /usr/bin/env python3

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from . import stat_utils


def get_relevance_mask(shape, gt_labels, embeddings_come_from_same_source=False, label_counts=None):
    """
    获取一个r_m用于评估聚类算法的性能。其是一个二维数组，行代表一个数据样本，列代表一个类别。
    gt_labels:真实样本的标签
    embeddings_come_from_same_source：嵌入是否来自同一个数据源，是则最后一列不需要预留位置用于噪声点，否则预留一个位置
    label_counts：字典，表示每个类别在数据集中出现的次数，不提供则自己算
    """
    # This assumes that k was set to at least the max number of relevant items
    if label_counts is None:
        label_counts = {k: v for k, v in zip(*np.unique(gt_labels, return_counts=True))}
    relevance_mask = np.zeros(shape=shape, dtype=np.int)
    for k, v in label_counts.items():
        matching_rows = np.where(gt_labels == k)[0]
        max_column = v - 1 if embeddings_come_from_same_source else v
        relevance_mask[matching_rows, :max_column] = 1
    return relevance_mask


def get_label_counts(reference_labels):
    """
    获取输入样本的标签计数。
    给定一组样本的标签(reference_labels)，它返回一个字典(label_counts)。
    字典中的键是样本标签，值是该标签在输入中出现的次数。
    此外，代码还计算了一个整数(num_k)，表示一个类别中最大允许的样本数(即该类别出现的最大次数)。
    最大值被限制在1023，因为faiss只允许最大k值为1024，而我们需要做k+1。
    """
    # 返回输入标签中出现的每个唯一标签及其出现次数
    unique_labels, label_counts = np.unique(reference_labels, return_counts=True)
    num_k = min(1023, int(np.max(label_counts)))  # faiss can only do a max of k=1024, and we have to do k+1
    return {k: v for k, v in zip(unique_labels, label_counts)}, num_k


class AccuracyCalculator:
    def __init__(self, include=(), exclude=()):
        self.function_keyword = "calculate_"
        function_names = [x for x in dir(self) if x.startswith(self.function_keyword)]
        metrics = [x.replace(self.function_keyword, "", 1) for x in function_names]
        self.original_function_dict = {x: getattr(self, y) for x, y in zip(metrics, function_names)}
        self.original_function_dict = self.get_function_dict(include, exclude)
        self.curr_function_dict = self.get_function_dict()

    def get_function_dict(self, include=(), exclude=()):
        """
        get_function_dict 函数根据 include 和 exclude 的设置，返回一个字典，其中包含需要计算的指标函数。
        """
        if len(include) == 0:
            include = list(self.original_function_dict.keys())
        included_metrics = [k for k in include if k not in exclude]
        return {k: v for k, v in self.original_function_dict.items() if k in included_metrics}

    def get_curr_metrics(self):
        """
        get_curr_metrics 函数返回当前包含的指标。
        """
        return [k for k in self.curr_function_dict.keys()]

    def requires_clustering(self):
        """requires_clustering 和 requires_knn 函数返回需要聚类和需要 kNN 的指标。"""
        return ["NMI", "AMI"]

    def requires_knn(self):
        """requires_clustering 和 requires_knn 函数返回需要聚类和需要 kNN 的指标。"""
        return ["precision_at_1", "mean_average_precision_at_r", "r_precision"]

    def get_cluster_labels(self, query, query_labels, **kwargs):
        """
        get_cluster_labels 函数对查询数据进行 kMeans 聚类，返回聚类标签。
        """
        num_clusters = len(set(query_labels.flatten()))
        return stat_utils.run_kmeans(query, num_clusters)

    def calculate_NMI(self, query_labels, cluster_labels, **kwargs):
        """
        本函数：衡量聚类的准确度
        calculate_NMI、calculate_AMI、
        calculate_precision_at_1、calculate_mean_average_precision_at_r、calculate_r_precision
        分别对应不同的指标计算函数。
        它们的输入是查询数据、参考数据、查询标签、参考标签、是否为同源嵌入（即查询和参考数据是否来自同一个数据集）等参数，
        其中 calculate_mean_average_precision_at_r 和 calculate_r_precision 还需要计算的标签数量 label_counts。
        这些函数的返回值是指标的值。
        """
        return normalized_mutual_info_score(query_labels, cluster_labels)

    def calculate_AMI(self, query_labels, cluster_labels, **kwargs):
        """
        归一化NMI
        """
        return adjusted_mutual_info_score(query_labels, cluster_labels)

    def calculate_precision_at_1(self, knn_labels, query_labels, **kwargs):
        def precision_at_k(knn_labels, gt_labels, k):
            """
            表示模型再前k个最近邻中预测正确的样本数占比，可以衡量模型再给定的k值下的精度。
            从 knn_labels 矩阵中取出每行前 k 个元素，即只取每个样本的前 k 个最相似的邻居的标签，存储到 curr_knn_labels 中。
            这个操作是在计算 Top-k Precision 指标时经常使用的。
            然后将每个样本的前K个最近邻中正确匹配的数量除以K，得到的是每个样本的精度，
            最后再对所有样本的精度取平均，得到的就是这个K个最近邻的平均精度。
            """
            curr_knn_labels = knn_labels[:, :k]
            precision = np.mean(np.sum(curr_knn_labels == gt_labels, axis=1) / k)
            return precision

        return precision_at_k(knn_labels, query_labels[:, None], 1)

    def calculate_mean_average_precision_at_r(self, knn_labels, query_labels, embeddings_come_from_same_source=False,
                                              label_counts=None, **kwargs):
        def mean_average_precision_at_r(knn_labels, gt_labels, embeddings_come_from_same_source=False,
                                        label_counts=None):
            """
            根据输入的knn_labels和gt_labels，生成1个用于指示每个样本是否为相关项的掩码。
            然后计算每个样本在前K个邻居中的精度，得到1个长度为k的精度向量precision_at_ks，
            cumulative_correct是一个与knn_lables相同形状的矩阵，表示前k个邻居中每个位置与相关项的交集数量的累计和。
            最后将每个样本的精度向量与r_m相乘，再对每行求和得到summed_precision_per_row,
            用summed_precision_per_row除以max_possible_matches_per_row得到每个样本的平均精度，再对所有样本求平均得到map@R
            """

            relevance_mask = get_relevance_mask(knn_labels.shape, gt_labels,
                                                embeddings_come_from_same_source,
                                                label_counts)
            num_samples, num_k = knn_labels.shape
            equality = (knn_labels == gt_labels) * relevance_mask.astype(bool)
            cumulative_correct = np.cumsum(equality, axis=1)
            k_idx = np.tile(np.arange(1, num_k + 1), (num_samples, 1))
            precision_at_ks = (cumulative_correct * equality) / k_idx
            summed_precision_per_row = np.sum(precision_at_ks * relevance_mask, axis=1)
            max_possible_matches_per_row = np.sum(relevance_mask, axis=1)
            return np.mean(summed_precision_per_row / max_possible_matches_per_row)

        return mean_average_precision_at_r(knn_labels, query_labels[:, None], embeddings_come_from_same_source,
                                           label_counts)

    def calculate_r_precision(self, knn_labels, query_labels, embeddings_come_from_same_source=False, label_counts=None,
                              **kwargs):
        def r_precision(knn_labels, gt_labels, embeddings_come_from_same_source=False, label_counts=None):
            """
            # 评价指标评估KNN算法的性能，在针对某个查询时，检索的前R个结果，与查询相关的正确结果占所有相关正确结果的比例。
            knn_lables:算法返回的k个近邻的标签
            gt_labels:真实样本的标签
            embeddings_come_from_same_source：嵌入是否来自同一个数据源，是则最后一列不需要预留位置用于噪声点，否则预留一个位置
            label_counts：字典，表示每个类别在数据集中出现的次数，不提供则自己算
            """

            relevance_mask = get_relevance_mask(knn_labels.shape, gt_labels,
                                                embeddings_come_from_same_source, label_counts)
            matches_per_row = np.sum((knn_labels == gt_labels) * relevance_mask.astype(bool), axis=1)
            max_possible_matches_per_row = np.sum(relevance_mask, axis=1)
            return np.mean(matches_per_row / max_possible_matches_per_row)

        return r_precision(knn_labels, query_labels[:, None], embeddings_come_from_same_source, label_counts)

    def get_accuracy(self, query, reference,
                     query_labels, reference_labels,
                     embeddings_come_from_same_source,
                     include=(), exclude=()):
        """
        get_accuracy 函数是计算准确度的核心函数。它接受查询数据、参考数据、查询标签、参考标签、是否为同源嵌入、包含和排除的指标等参数。

        如果需要聚类的指标，它会调用 get_cluster_labels 函数计算聚类标签。
        最后，它会通过 _get_accuracy 函数调用各个指标的计算函数，返回计算结果。

        给定查询集和参考集的嵌入向量及其对应的标签，以及一些其他参数，它将返回一个包含各种指标及其对应值的字典。
        该函数首先根据输入的参数构建一个 kwargs 字典，该字典包含了计算各个指标所需的所有参数。
        然后它会检查当前所需要计算的指标，看看是否需要计算k最近邻或者聚类等中间结果，如果需要，则调用相应的函数来计算这些中间结果。
        最后，它使用当前指标的函数字典来计算每个指标的值，并将其存储在一个结果字典中返回
        """
        embeddings_come_from_same_source = embeddings_come_from_same_source or (query is reference)

        self.curr_function_dict = self.get_function_dict(include, exclude)

        kwargs = {"query": query,
                  "reference": reference,
                  "query_labels": query_labels,
                  "reference_labels": reference_labels,
                  "embeddings_come_from_same_source": embeddings_come_from_same_source}

        if any(x in self.requires_knn() for x in self.get_curr_metrics()):
            # 如果需要计算kNN的指标，它会通过get_label_counts函数获取标签数量和k值，然后调用stat_utils.get_knn函数计算kNN。
            label_counts, num_k = get_label_counts(reference_labels)
            knn_indices = stat_utils.get_knn(reference, query, num_k, embeddings_come_from_same_source)
            knn_labels = reference_labels[knn_indices]
            kwargs["label_counts"] = label_counts
            kwargs["knn_labels"] = knn_labels

        if any(x in self.requires_clustering() for x in self.get_curr_metrics()):
            kwargs["cluster_labels"] = self.get_cluster_labels(**kwargs)

        return self._get_accuracy(self.curr_function_dict, **kwargs)

    def _get_accuracy(self, function_dict, **kwargs):
        return {k: v(**kwargs) for k, v in function_dict.items()}
