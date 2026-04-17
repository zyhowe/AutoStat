"""评估指标模块 - 计算各种模型评估指标"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    confusion_matrix, classification_report
)


class MetricsCalculator:
    """评估指标计算器"""

    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                         y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        计算分类指标

        参数:
        - y_true: 真实标签
        - y_pred: 预测标签
        - y_proba: 预测概率（可选）

        返回:
        - 指标字典
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        }

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # 计算分类报告
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics["classification_report"] = report

        # 计算 AUC（仅二分类且提供了概率）
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            except:
                metrics["roc_auc"] = None

        return metrics

    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        计算回归指标

        参数:
        - y_true: 真实值
        - y_pred: 预测值

        返回:
        - 指标字典
        """
        metrics = {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "explained_variance": float(explained_variance_score(y_true, y_pred))
        }

        # 计算 MAPE（避免除零）
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            metrics["mape"] = float(mape)
        else:
            metrics["mape"] = None

        return metrics

    @staticmethod
    def calculate_clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        计算聚类指标

        参数:
        - X: 特征矩阵
        - labels: 聚类标签

        返回:
        - 指标字典
        """
        metrics = {}

        n_clusters = len(np.unique(labels))
        n_samples = len(labels)

        metrics["n_clusters"] = n_clusters
        metrics["n_samples"] = n_samples

        # 计算轮廓系数（需要至少2个簇）
        if n_clusters >= 2 and n_clusters < n_samples:
            try:
                metrics["silhouette_score"] = float(silhouette_score(X, labels))
            except:
                metrics["silhouette_score"] = None

            try:
                metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X, labels))
            except:
                metrics["calinski_harabasz_score"] = None

            try:
                metrics["davies_bouldin_score"] = float(davies_bouldin_score(X, labels))
            except:
                metrics["davies_bouldin_score"] = None
        else:
            metrics["silhouette_score"] = None
            metrics["calinski_harabasz_score"] = None
            metrics["davies_bouldin_score"] = None

        # 计算每个簇的样本数
        unique, counts = np.unique(labels, return_counts=True)
        metrics["cluster_sizes"] = {int(k): int(v) for k, v in zip(unique, counts)}

        return metrics

    @staticmethod
    def calculate_time_series_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        计算时间序列指标

        参数:
        - y_true: 真实值
        - y_pred: 预测值

        返回:
        - 指标字典
        """
        metrics = {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mape": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
        }

        # 计算 sMAPE
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
        metrics["smape"] = float(smape)

        return metrics

    @staticmethod
    def format_metrics(metrics: Dict[str, Any], task_type: str) -> str:
        """格式化指标为可读字符串"""
        if task_type == "classification":
            lines = [
                f"准确率 (Accuracy): {metrics.get('accuracy', 0):.4f}",
                f"精确率 (Precision): {metrics.get('precision', 0):.4f}",
                f"召回率 (Recall): {metrics.get('recall', 0):.4f}",
                f"F1分数: {metrics.get('f1_score', 0):.4f}"
            ]
            if metrics.get('roc_auc'):
                lines.append(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            return "\n".join(lines)

        elif task_type == "regression":
            return "\n".join([
                f"MSE: {metrics.get('mse', 0):.4f}",
                f"RMSE: {metrics.get('rmse', 0):.4f}",
                f"MAE: {metrics.get('mae', 0):.4f}",
                f"R²: {metrics.get('r2', 0):.4f}",
                f"MAPE: {metrics.get('mape', 0):.2f}%" if metrics.get('mape') else ""
            ])

        elif task_type == "clustering":
            lines = [f"聚类数量: {metrics.get('n_clusters', 0)}"]
            if metrics.get('silhouette_score'):
                lines.append(f"轮廓系数: {metrics['silhouette_score']:.4f}")
            if metrics.get('calinski_harabasz_score'):
                lines.append(f"CH指数: {metrics['calinski_harabasz_score']:.2f}")
            if metrics.get('davies_bouldin_score'):
                lines.append(f"DB指数: {metrics['davies_bouldin_score']:.4f}")
            return "\n".join(lines)

        elif task_type == "time_series":
            return "\n".join([
                f"MSE: {metrics.get('mse', 0):.4f}",
                f"RMSE: {metrics.get('rmse', 0):.4f}",
                f"MAE: {metrics.get('mae', 0):.4f}",
                f"MAPE: {metrics.get('mape', 0):.2f}%",
                f"sMAPE: {metrics.get('smape', 0):.2f}%"
            ])

        return str(metrics)