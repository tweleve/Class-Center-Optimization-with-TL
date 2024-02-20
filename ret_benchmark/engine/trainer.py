# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import datetime
import time

import numpy as np
import torch

from ret_benchmark.data.evaluations.eval import AccuracyCalculator
from ret_benchmark.utils.feat_extractor import feat_extractor
from ret_benchmark.utils.metric_logger import MetricLogger
from ret_benchmark.utils.log_info import log_info
from ret_benchmark.modeling.xbm import XBM
from center_loss import CenterLoss
from fvcore.nn import FlopCountAnalysis
from sklearn.manifold import TSNE
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch


def flush_log(writer, iteration):
    for k, v in log_info.items():
        if isinstance(v, np.ndarray):
            writer.add_histogram(k, v, iteration)
        else:
            writer.add_scalar(k, v, iteration)
    for k in list(log_info.keys()):
        del log_info[k]


def batch_former(cfg_add_bf, bf_feats, bf_targets):
    from ret_benchmark.modeling.batchformer import TransformerDecorator
    BF = TransformerDecorator(add_bf=cfg_add_bf, dim=512, eval_global=0)
    BF.to('cuda')
    feats_bf, targets_bf = BF.forward(bf_feats, bf_targets)
    return feats_bf, targets_bf


# 计算参数量
def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def do_train(cfg, model, train_loader, val_loader, scheduler,
             criterion, criterion_cent,
             optimizer, optimizer_centloss,
             checkpointer, writer, device, checkpoint_period, arguments, logger, ):
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    # meters1 = MetricLogger(delimiter="  ")
    max_iter = cfg.SOLVER.MAX_ITERS

    best_iteration = -1
    best_mapr = 0
    best_recall = 0
    best_recall2 = 0
    best_recall4 = 0
    best_recall8 = 0

    start_training_time = time.time()
    end = time.time()

    if cfg.XBM.ENABLE:
        logger.info(">>> use XBM")
        xbm = XBM(cfg)
    print("==BF:   ", cfg.BF.ADD_BF)
    print("==Data: ", cfg.DATA.TRAIN_IMG_SOURCE)
    iteration = 0

    _train_loader = iter(train_loader)
    while iteration <= max_iter:
        try:
            images, targets, indices = _train_loader.next()
        except StopIteration:
            _train_loader = iter(train_loader)
            images, targets, indices = _train_loader.next()

        """验证！！！"""
        if iteration == -1:
            # 假设 val_loader 是你的验证集数据加载器
            model.eval()  # 确保模型处于评估模式
            labels = val_loader[0].dataset.label_list
            labels = np.array([int(k) for k in labels])
            for data in val_loader:
                # data = data.to(device)  # 确保数据移动到与模型相同的设备
                features = feat_extractor(cfg.BF.ADD_BF, model, val_loader[0], logger=logger)  # (batch,dim)
                # features_np = features.detach().cpu().numpy()  # 转换为NumPy数组
                features_np = features

                # t-SNE降维
                tsne = TSNE(n_components=2, random_state=42)
                features_2d = tsne.fit_transform(features_np)

                # KMeans聚类
                kmeans = KMeans(n_clusters=10, random_state=42)
                labels = kmeans.fit_predict(features_2d)

                # 可视化
                # plt.figure(figsize=(10, 8))
                # scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
                # plt.title('Feature Clustering with t-SNE on Validation Set')
                # plt.xlabel('t-SNE feature 1')
                # plt.ylabel('t-SNE feature 2')
                # plt.legend(*scatter.legend_elements(), title="Clusters")
                # plt.show()

                unique_labels = np.unique(labels)
                n_labels = len(unique_labels)
                colors = [str(item) for item in np.linspace(0, 1, n_labels)]

                plt.figure(figsize=(10, 8))

                for label, color in zip(unique_labels, colors):
                    plt.scatter(features_2d[labels == label, 0], features_2d[labels == label, 1], c=color, label=label,
                                alpha=0.5)

                # plt.legend(title="Labels")
                # plt.title('Feature Clustering on Validation Set with t-SNE')
                # plt.xlabel('Dimension 1')
                # plt.ylabel('Dimension 2')
                plt.axis('off')
                plt.show()
        if iteration == -2:
            # 假设 val_loader 是你的验证集数据加载器
            model.eval()  # 确保模型处于评估模式
            labels = val_loader[0].dataset.label_list
            labels = np.array([int(k) for k in labels])
            for data in val_loader:
                # data = data.to(device)  # 确保数据移动到与模型相同的设备
                features = feat_extractor(cfg.BF.ADD_BF, model, val_loader[0], logger=logger)  # (batch,dim)
                # features_np = features.detach().cpu().numpy()  # 转换为NumPy数组
                features_np = features

                # t-SNE降维
                tsne = TSNE(n_components=2, random_state=42)
                features_2d = tsne.fit_transform(features_np)

                # KMeans聚类
                kmeans = KMeans(n_clusters=10, random_state=42)
                labels = kmeans.fit_predict(features_2d)

                # 可视化
                # plt.figure(figsize=(10, 8))
                # scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
                # plt.title('Feature Clustering with t-SNE on Validation Set')
                # plt.xlabel('t-SNE feature 1')
                # plt.ylabel('t-SNE feature 2')
                # plt.legend(*scatter.legend_elements(), title="Clusters")
                # plt.show()

                unique_labels = np.unique(labels)
                n_labels = len(unique_labels)
                colors = [str(item) for item in np.linspace(0, 1, n_labels)]

                plt.figure(figsize=(10, 8))

                for label, color in zip(unique_labels, colors):
                    plt.scatter(features_2d[labels == label, 0], features_2d[labels == label, 1], c=color, label=label,
                                alpha=0.5)

                # plt.legend(title="Labels")
                # plt.title('Feature Clustering on Validation Set with t-SNE')
                # plt.xlabel('Dimension 1')
                # plt.ylabel('Dimension 2')
                plt.axis('off')
                plt.show()
        if (iteration % cfg.VALIDATION.VERBOSE == 0 or iteration == max_iter) and iteration > 0:
            """验证！！！"""
            model.eval()
            # ====================================================================

            if iteration == 10000:
                # 假设 val_loader 是你的验证集数据加载器
                model.eval()  # 确保模型处于评估模式
                labels = val_loader[0].dataset.label_list
                labels = np.array([int(k) for k in labels])
                for data in val_loader:
                    # data = data.to(device)  # 确保数据移动到与模型相同的设备
                    features = feat_extractor(cfg.BF.ADD_BF, model, val_loader[0], logger=logger)  # (batch,dim)
                    # features_np = features.detach().cpu().numpy()  # 转换为NumPy数组
                    features_np = features

                    # t-SNE降维
                    tsne = TSNE(n_components=2, random_state=42)
                    features_2d = tsne.fit_transform(features_np)

                    # KMeans聚类
                    kmeans = KMeans(n_clusters=10, random_state=42)
                    labels = kmeans.fit_predict(features_2d)
                    unique_labels = np.unique(labels)
                    n_labels = len(unique_labels)
                    colors = [str(item) for item in np.linspace(0, 1, n_labels)]

                    plt.figure(figsize=(10, 8))

                    for label, color in zip(unique_labels, colors):
                        plt.scatter(features_2d[labels == label, 0], features_2d[labels == label, 1],
                                    c=color, label=label, alpha=0.5)
                    # 可视化
                    # plt.figure(figsize=(10, 8))
                    # scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
                    # plt.title('Feature Clustering with t-SNE on Validation Set')
                    # plt.xlabel('t-SNE feature 1')
                    # plt.ylabel('t-SNE feature 2')
                    # plt.legend(*scatter.legend_elements(), title="Clusters")
                    plt.axis('off')
                    plt.show()

                    # break  # 如果你只想处理验证集中的第一批数据，则使用break语句

            total_params = get_parameter_count(model)
            print(f"Total parameters: {total_params}")
            # 计算FLOPs
            # 假设你的输入尺寸为1x3x224x224，这里的尺寸应该与你模型的实际输入尺寸相匹配
            inputs = torch.randn(1, 3, 224, 224).to(device)
            flop_analysis = FlopCountAnalysis(model, inputs)
            total_flops = flop_analysis.total()
            print(f"Total FLOPs: {total_flops}")
            # ===================================================================
            logger.info("Validation")

            labels = val_loader[0].dataset.label_list
            labels = np.array([int(k) for k in labels])
            feats = feat_extractor(cfg.BF.ADD_BF, model, val_loader[0], logger=logger)  # (batch,dim)
            # ================================================================================
            # ================================================================================
            # feats, labels = batch_former(feats, labels)
            # ================================================================================
            # ================================================================================
            ret_metric = AccuracyCalculator(include=("precision_at_1",
                                                     "mean_average_precision_at_r",
                                                     "r_precision"),
                                            exclude=())
            ret_metric = ret_metric.get_accuracy(feats, feats, labels, labels, True)
            mapr_curr = ret_metric['mean_average_precision_at_r']
            for k, v in ret_metric.items():
                log_info[f"e_{k}"] = v

            scheduler.step(log_info['e_precision_at_1'])
            log_info["lr"] = optimizer.param_groups[0]["lr"]
            # print("\n\nSTART", end='*******************\n')
            if mapr_curr > best_mapr:
                best_mapr = mapr_curr
                best_iteration = iteration
                logger.info(f"Best iteration {iteration}: {ret_metric}")
            else:
                logger.info(f"Performance at iteration {iteration:05d}: {ret_metric}")
            flush_log(writer, iteration)
            # ====================================================================
            # ====================================================================
            # ====================================================================
            from ret_benchmark.data.evaluations import RetMetric
            ret_metric1 = RetMetric(feats=feats, labels=labels)
            recall_curr1 = ret_metric1.recall_k(1)

            # =====================================================================================================
            if recall_curr1 > best_recall:
                best_recall = recall_curr1
                best_iteration1 = iteration
                logger.info(f'Best iteration {iteration}: recall@1: {best_recall:.3f}')
                checkpointer.save(f"best_model1")
            else:
                logger.info(f'Recall@1 at iteration {iteration:06d}: {recall_curr1:.3f}')
            # print("", end='\n')
            # =====================================================================================================
            recall_curr2 = ret_metric1.recall_k(2)

            if recall_curr2 > best_recall2:
                best_recall2 = recall_curr2
                best_iteration2 = iteration
                logger.info(f'Best iteration {iteration}: recall@2: {best_recall2:.3f}')
                checkpointer.save(f"best_model2")
            else:
                logger.info(f'Recall@2 at iteration {iteration:04d}: {recall_curr2:.3f}')
            # print("", end='*******************\n')
            # ==============================================================
            recall_curr4 = ret_metric1.recall_k(4)

            if recall_curr4 > best_recall4:
                best_recall4 = recall_curr4
                best_iteration4 = iteration
                logger.info(f'Best iteration {iteration}: recall@4: {best_recall4:.3f}')
                checkpointer.save(f"best_model4")
            else:
                logger.info(f'Recall@4 at iteration {iteration:04d}: {recall_curr4:.3f}')
            # print("END", end='*******************\n')
            # ==============================================================
            recall_curr8 = ret_metric1.recall_k(8)
            if recall_curr8 > best_recall8:
                best_recall8 = recall_curr8
                best_iteration8 = iteration
                logger.info(f'Best iteration {iteration}: recall@8: {best_recall8:.3f}')
                checkpointer.save(f"best_model8")
            else:
                logger.info(f'Recall@8 at iteration {iteration:04d}: {recall_curr8:.3f}')
            print("", end='*******************\n')
            # ==============================================================

        """训练！！！"""

        model.train()

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = targets.to(device)
        feats = model(images)  # 得到预测值32.512
        # ==================================
        if iteration == 100000:
            # t-SNE降维
            tsne = TSNE(n_components=2, random_state=42)
            # 假设 features 是从模型中提取的特征张量，并且它们位于CUDA设备上
            features_on_cpu = feats.cpu()  # 将特征移动到CPU
            features_np = features_on_cpu.detach().cpu().numpy()  # 转换为NumPy数组

            tsne = TSNE(n_components=2, random_state=42)
            features_2d = tsne.fit_transform(features_np)

            # 假设 labels 也是一个张量，并且位于CUDA上
            labels_on_cpu = targets.cpu()  # 将标签移动到CPU
            labels_np = labels_on_cpu.numpy()  # 转换为NumPy数组

            plt.figure(figsize=(10, 8))
            unique_labels = np.unique(labels_np)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            for label, color in zip(unique_labels, colors):
                plt.scatter(features_2d[labels_np == label, 0], features_2d[labels_np == label, 1], color=color,
                            label=label, alpha=0.5)

            plt.legend()
            plt.title('Feature Clustering with t-SNE')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.show()

        # print("train")
        feats, targets = batch_former(cfg.BF.ADD_BF, feats, targets)
        # ============================================================
        # ============================================================

        if cfg.XBM.ENABLE and iteration > cfg.XBM.START_ITERATION:
            # print("train========")
            xbm.enqueue_dequeue(feats.detach(), targets.detach())

        x_loss = criterion(feats, targets, feats, targets)
        # loss = criterion(feats, targets, feats, targets)
        # print("train")
        if criterion_cent != 0:
            loss_center = criterion_cent(feats, targets)
            loss = x_loss + loss_center * 1  # 1是权重
            optimizer.zero_grad()
            optimizer_centloss.zero_grad()
        else:
            loss = x_loss
            optimizer.zero_grad()
            optimizer_centloss.zero_grad()
        log_info["batch_loss"] = loss.item()

        if cfg.XBM.ENABLE and iteration > cfg.XBM.START_ITERATION:
            # 使用xbm
            xbm_feats, xbm_targets = xbm.get()
            xbm_loss = criterion(feats, targets, xbm_feats, xbm_targets)
            log_info["xbm_loss"] = xbm_loss.item()
            loss = loss + cfg.XBM.WEIGHT * xbm_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for param in criterion_cent.parameters():
            # 这样做，权重不会影响中心损失学习率
            param.grad.data *= (1. / 1)  # 第二个1是weight-cent
        optimizer_centloss.step()  # 添加的代码

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time, loss=loss.item())
        # meters1.update(time=batch_time, data=data_time, loss=loss.item())
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        # eta_seconds1 = meters1.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        # eta_string1 = str(datetime.timedelta(seconds=int(eta_seconds1)))

        if iteration % 100 == 0 or iteration == max_iter:
            logger.info(meters.delimiter.join(
                ["eta: {eta}", "iter: {iter}", "{meters}", "lr: {lr:.6f}", "max mem: {memory:.1f} GB"]).format(
                eta=eta_string, iter=iteration, meters=str(meters), lr=optimizer.param_groups[0]["lr"],
                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0)
            )
            log_info["loss"] = loss.item()
            flush_log(writer, iteration)

        if iteration % checkpoint_period == 0 and cfg.SAVE:
            checkpointer.save("model_{:06d}".format(iteration))
            pass

        del feats
        del loss

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / (max_iter)))

    logger.info(f"Best iteration: {best_iteration :06d} | best MAP@R {best_mapr} ")
    logger.info(f"Best iteration: {best_iteration1 :04d} | best recall@1 {best_recall} ")
    logger.info(f"Best iteration: {best_iteration2 :04d} | best recall@2 {best_recall2} ")
    logger.info(f"Best iteration: {best_iteration4 :04d} | best recall@4 {best_recall4} ")
    logger.info(f"Best iteration: {best_iteration8 :04d} | best recall@8 {best_recall8} ")
    writer.close()
