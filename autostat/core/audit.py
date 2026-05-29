# autostat/core/audit.py
"""勾稽关系发现模块 - 四条路径：相关聚类 + 共同非空聚类 + 非空数相近聚类 + 行相似度聚类"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AuditRuleDiscoverer:
    """勾稽规则发现器 - 四条路径"""

    def __init__(self,
                 precision: float = 1e-6,
                 min_confidence: float = 0.5,
                 corr_threshold: float = 0.9,
                 min_nonnull_count: int = 10,
                 min_nonnull_rate: float = 0.01,
                 cooccur_ratio: float = 0.9,
                 min_cooccurrence_rows: int = 10,
                 nonnull_diff_ratio: float = 0.2,
                 row_similarity_threshold: float = 0.99,
                 min_cluster_size: int = 3,
                 max_numeric_fields: int = 100,
                 ransac_iter: int = 50,
                 ransac_sample_size: int = 20,
                 inlier_ratio: float = 0.7,
                 debug: bool = False):
        """
        勾稽规则发现器初始化

        参数:
        - precision: 浮点数比较精度
        - min_confidence: 最小置信度阈值
        - corr_threshold: 相关系数阈值（路径一使用）
        - min_nonnull_count: 最小非空数（低于此值排除）
        - min_nonnull_rate: 最小非空率（低于此值排除）
        - cooccur_ratio: 共同非空比例阈值（路径二使用）
        - min_cooccurrence_rows: 最小共同非空行数
        - nonnull_diff_ratio: 非空数差异比例阈值（路径三使用）
        - row_similarity_threshold: 行相似度阈值（路径四使用）
        - min_cluster_size: 最小聚类大小（路径四使用）
        - max_numeric_fields: 最多处理的数值字段数
        - ransac_iter: RANSAC 迭代次数
        - ransac_sample_size: RANSAC 采样大小
        - inlier_ratio: 内点比例阈值
        - debug: 是否输出调试信息
        """
        self.precision = precision
        self.min_confidence = min_confidence
        self.corr_threshold = corr_threshold
        self.min_nonnull_count = min_nonnull_count
        self.min_nonnull_rate = min_nonnull_rate
        self.cooccur_ratio = cooccur_ratio
        self.min_cooccurrence_rows = min_cooccurrence_rows
        self.nonnull_diff_ratio = nonnull_diff_ratio
        self.row_similarity_threshold = row_similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.max_numeric_fields = max_numeric_fields
        self.ransac_iter = ransac_iter
        self.ransac_sample_size = ransac_sample_size
        self.inlier_ratio = inlier_ratio
        self.debug = debug

    def _log(self, msg: str):
        if self.debug:
            print(f"  [DEBUG] {msg}")

    def discover_all(self, data: pd.DataFrame, variable_types: Dict[str, str],
                     foreign_keys: List[Dict] = None) -> Dict[str, Any]:
        """发现所有类型的勾稽规则"""
        self._log("=" * 60)
        self._log("开始勾稽关系发现（四条路径）")
        self._log("=" * 60)

        rules = {
            "arithmetic_rules": [],
            "functional_dependencies": [],
            "temporal_rules": [],
            "foreign_keys": foreign_keys or []
        }

        total_rows = len(data)
        nonnull_counts = {col: data[col].notna().sum() for col in data.columns}
        nonnull_rates = {col: data[col].notna().mean() for col in data.columns}

        self._log(f"总行数: {total_rows}")
        self._log("=== 字段非空统计（前20） ===")
        for col in list(data.columns)[:20]:
            self._log(f"  {col}: 非空数={nonnull_counts[col]}, 非空率={nonnull_rates[col]:.2%}")

        # 过滤高缺失字段
        self._keep_cols = [
            col for col in data.columns
            if nonnull_counts[col] >= self.min_nonnull_count
               or nonnull_rates[col] >= self.min_nonnull_rate
        ]
        excluded = [col for col in data.columns if col not in self._keep_cols]
        if excluded:
            self._log(f"排除字段: {excluded}")
        self._log(f"保留字段数: {len(self._keep_cols)} / {len(data.columns)}")

        # 数值关系发现
        numeric_cols = [col for col, typ in variable_types.items()
                        if typ == 'continuous' and col in data.columns
                        and col in self._keep_cols]
        self._log(f"连续变量: {len(numeric_cols)} 个")

        if len(numeric_cols) >= 3:
            arithmetic_rules = self._discover_arithmetic_rules(data, numeric_cols)
            rules["arithmetic_rules"] = arithmetic_rules
            self._log(f"发现 {len(arithmetic_rules)} 条数值关系")
        else:
            self._log("连续变量不足3个，跳过数值关系发现")

        # 函数依赖发现
        fd_cols = [col for col, typ in variable_types.items()
                   if typ in ['categorical', 'categorical_numeric', 'ordinal']
                   and col in data.columns and col in self._keep_cols]
        self._log(f"分类变量: {len(fd_cols)} 个")

        if len(fd_cols) >= 2:
            fd_rules = self._discover_functional_dependencies(data, fd_cols)
            rules["functional_dependencies"] = fd_rules
            self._log(f"发现 {len(fd_rules)} 条函数依赖")

        # 时序关系发现
        date_cols = [col for col, typ in variable_types.items()
                     if typ == 'datetime' and col in data.columns
                     and col in self._keep_cols]
        self._log(f"日期变量: {len(date_cols)} 个")

        if len(date_cols) >= 2:
            temporal_rules = self._discover_temporal_rules(data, date_cols)
            rules["temporal_rules"] = temporal_rules
            self._log(f"发现 {len(temporal_rules)} 条时序关系")

        self._log("=" * 60)
        return rules

    # ==================== 函数依赖 ====================
    def _discover_functional_dependencies(self, df: pd.DataFrame,
                                          categorical_cols: List[str]) -> List[Dict]:
        """发现分类字段间的函数依赖"""
        self._log("  [函数依赖] 开始...")

        if len(categorical_cols) < 2:
            return []

        date_derived_suffixes = ['_year', '_month', '_quarter', '_week', '_weekday', '_day', '_is_weekend']
        filtered_cols = []

        for col in categorical_cols:
            is_derived = any(col.endswith(suffix) for suffix in date_derived_suffixes)
            if not is_derived and col in self._keep_cols:
                filtered_cols.append(col)

        self._log(f"    有效分类变量: {len(filtered_cols)}")

        if len(filtered_cols) < 2:
            return []

        rules = []
        unique_counts = {col: df[col].dropna().nunique() for col in filtered_cols}
        constant_cols = [col for col, cnt in unique_counts.items() if cnt <= 1]

        for i, col1 in enumerate(filtered_cols):
            for col2 in filtered_cols[i + 1:]:
                if col2 not in constant_cols and unique_counts[col1] >= unique_counts[col2]:
                    valid_df = self._get_valid_data(df, [col1, col2])
                    if len(valid_df) > 0:
                        grouped = valid_df.groupby(col1)[col2].nunique()
                        if len(grouped) > 0 and (grouped == 1).all():
                            rules.append({
                                "rule": f"{col1} → {col2}",
                                "confidence": 1.0,
                                "priority": "高",
                                "fields": [col1, col2],
                                "violation_count": 0,
                                "violation_samples": []
                            })
                            self._log(f"      发现: {col1} → {col2}")

                if col1 not in constant_cols and unique_counts[col2] >= unique_counts[col1]:
                    valid_df = self._get_valid_data(df, [col2, col1])
                    if len(valid_df) > 0:
                        grouped = valid_df.groupby(col2)[col1].nunique()
                        if len(grouped) > 0 and (grouped == 1).all():
                            rules.append({
                                "rule": f"{col2} → {col1}",
                                "confidence": 1.0,
                                "priority": "高",
                                "fields": [col2, col1],
                                "violation_count": 0,
                                "violation_samples": []
                            })
                            self._log(f"      发现: {col2} → {col1}")

        seen = set()
        unique_rules = []
        for rule in rules:
            if rule["rule"] not in seen:
                seen.add(rule["rule"])
                unique_rules.append(rule)

        return unique_rules

    # ==================== 时序关系 ====================
    def _discover_temporal_rules(self, df: pd.DataFrame,
                                 date_cols: List[str]) -> List[Dict]:
        """发现日期字段间的时序关系"""
        self._log("  [时序关系] 开始...")

        if len(date_cols) < 2:
            return []

        rules = []
        valid_cols = [c for c in date_cols
                      if c in self._keep_cols and df[c].notna().any()]

        for i, col1 in enumerate(valid_cols):
            for col2 in valid_cols[i + 1:]:
                valid_df = self._get_valid_data(df, [col1, col2])
                if len(valid_df) < 3:
                    continue

                try:
                    col1_data = pd.to_datetime(valid_df[col1], errors='coerce')
                    col2_data = pd.to_datetime(valid_df[col2], errors='coerce')
                except Exception:
                    continue

                valid_mask = col1_data.notna() & col2_data.notna()
                if valid_mask.sum() < 3:
                    continue

                c1_valid = col1_data[valid_mask]
                c2_valid = col2_data[valid_mask]

                if (c1_valid <= c2_valid).all():
                    rules.append({
                        "rule": f"{col1} ≤ {col2}",
                        "confidence": 1.0,
                        "priority": "高",
                        "fields": [col1, col2],
                        "violation_count": 0,
                        "violation_samples": []
                    })
                    self._log(f"      发现: {col1} ≤ {col2}")

                if (c1_valid == c2_valid).all():
                    rules.append({
                        "rule": f"{col1} = {col2}",
                        "confidence": 1.0,
                        "priority": "高",
                        "fields": [col1, col2],
                        "violation_count": 0,
                        "violation_samples": []
                    })
                    self._log(f"      发现: {col1} = {col2}")

                diff = (c2_valid - c1_valid).dt.days
                if diff.nunique() == 1 and len(diff) > 0:
                    days = diff.iloc[0]
                    if days != 0:
                        rules.append({
                            "rule": f"{col2} = {col1} + {days}天",
                            "confidence": 1.0,
                            "priority": "高",
                            "fields": [col1, col2],
                            "violation_count": 0,
                            "violation_samples": []
                        })
                        self._log(f"      发现: {col2} = {col1} + {days}天")

        return rules

    # ==================== 主流程 ====================
    def _discover_arithmetic_rules(self, df: pd.DataFrame,
                                   numeric_cols: List[str]) -> List[Dict]:
        """
        主流程：四条路径独立运行，合并去重
        """
        self._log("  [数值关系] 开始...")

        if len(numeric_cols) < 3:
            return []

        all_rules = []

        # # 路径四：行相似度聚类
        self._log("\n  【路径四：行相似度聚类】")
        row_clusters = self._cluster_rows_by_similarity(df, numeric_cols)

        # 调试：打印行聚类结果中是否包含目标字段组（需要检查每个行聚类子集的字段相关性聚类）
        if self.debug and len(row_clusters) > 0:
            self._log(f"路径四行聚类结果（共{len(row_clusters)}个行类）:")

        rules4 = []
        # 调用相关聚类再发现规则
        for i, cluster_df in enumerate(row_clusters):
            self._log(f"处理行聚类{i + 1}: {len(cluster_df)} 行")
            sub_rules = self._cluster_columns_and_rules(
                cluster_df, numeric_cols,
                min_nonnull_in_cluster=self.min_cluster_size,  # 全1字段才保留
                min_nonnull_ratio_in_cluster=0.01  # 或者100%非空
            )
            rules4.extend(sub_rules)

        #直接规则发现，不先相关的话，效果极差，基本得不到规则
        # for i, cluster_df in enumerate(row_clusters):
        #     weights = cluster_df.attrs.get('column_weights', None)
        #     all_numeric_cols = cluster_df.attrs.get('numeric_cols', numeric_cols)
        #     self._log(f"处理行聚类{i + 1}: {len(cluster_df)} 行")
        #
        #     # 关键修复：只保留在 all_numeric_cols 中的数值字段
        #     numeric_fields_in_cluster = [col for col in all_numeric_cols if col in cluster_df.columns]
        #
        #     if len(numeric_fields_in_cluster) < 3:
        #         self._log(f"    数值字段不足3个，跳过")
        #         continue
        #
        #     self._log(f"    数值字段数: {len(numeric_fields_in_cluster)}")
        #
        #     sub_rules = self._discover_rules_in_subcluster(
        #         cluster_df,  # 使用行聚类子集
        #         numeric_fields_in_cluster,  # 只传数值字段
        #         weights=weights,
        #         all_numeric_cols=all_numeric_cols
        #     )
        #     rules4.extend(sub_rules)

        all_rules.extend(rules4)
        self._log(f"    路径四发现规则数: {len(rules4)}")

        # 合并去重
        seen = set()
        unique_rules = []
        for rule in all_rules:
            key = frozenset(rule["fields"])
            if key not in seen:
                seen.add(key)
                unique_rules.append(rule)

        self._log(f"\n    合并后总计: {len(unique_rules)} 条数值关系")
        return unique_rules

    # ==================== 行相似度聚类 ====================
    def _cluster_rows_by_similarity(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[pd.DataFrame]:
        """
        基于0-1矩阵的行相似度在线聚类（加权余弦相似度，使用IDF权重）

        步骤：
        1. 将数值字段转换为0-1矩阵（非空=1，空=0）
        2. 计算每个字段的IDF权重（稀有字段权重大）
        3. 对0-1矩阵按行进行在线聚类（加权余弦相似度 > 阈值）
        4. 返回每个聚类的原始DataFrame
        """
        if len(df) < self.min_cluster_size:
            return []

        # 1. 转换为0-1矩阵（非空=1，空=0）
        X_binary = df[numeric_cols].notna().astype(int).values
        total_rows, n_fields = X_binary.shape

        # 2. 计算IDF权重
        field_freq = X_binary.sum(axis=0)
        idf = np.log((total_rows + 1) / (field_freq + 1)) + 1
        self._log(f"    IDF1: {idf}")

        # # 自适应映射
        # idf_min = idf.min()
        # idf_max = idf.max()
        # if idf_max == idf_min:
        #     weights = np.ones_like(idf)
        # else:
        #     n_fields_count = len(numeric_cols)
        #     max_multiplier = min(50, 10 * np.log2(max(n_fields_count, 2)))
        #     idf_norm = (idf - idf_min) / (idf_max - idf_min)
        #     k = 4.0
        #     exp_scale = (np.exp(k * idf_norm) - 1) / (np.exp(k) - 1)
        #     weights = 1 + exp_scale * (max_multiplier - 1)
        weights = idf ** 3

        self._log(f"    weights：{weights}")

        # 3. 在线聚类（加权余弦相似度）
        classes = []
        centers = []

        # 只聚到1类
        for i, row in enumerate(X_binary):
            weighted_row = row * weights
            best_class = -1
            best_sim = -1

            for j, center in enumerate(centers):
                both = (weighted_row * center).sum()
                norm_row = np.sqrt((weighted_row ** 2).sum())
                norm_center = np.sqrt((center ** 2).sum())
                if norm_row > 0 and norm_center > 0:
                    sim = both / (norm_row * norm_center)
                    if sim > best_sim and sim > self.row_similarity_threshold:
                        best_sim = sim
                        best_class = j

            if best_class >= 0:
                classes[best_class].append(i)
                old_size = len(classes[best_class]) - 1
                centers[best_class] = (centers[best_class] * old_size + weighted_row) / (old_size + 1)
            else:
                classes.append([i])
                centers.append(weighted_row.copy())

        # 可同时聚到多类
        # for i, row in enumerate(X_binary):
        #     weighted_row = row * weights
        #     assigned_classes = []  # 记录该行加入的所有类索引
        #
        #     for j, center in enumerate(centers):
        #         both = (weighted_row * center).sum()
        #         norm_row = np.sqrt((weighted_row ** 2).sum())
        #         norm_center = np.sqrt((center ** 2).sum())
        #         if norm_row > 0 and norm_center > 0:
        #             sim = both / (norm_row * norm_center)
        #             if sim > self.row_similarity_threshold:
        #                 assigned_classes.append(j)
        #
        #     if assigned_classes:
        #         # 该行同时加入所有符合条件的类
        #         for j in assigned_classes:
        #             classes[j].append(i)
        #             # 增量更新该类中心（均值）
        #             old_size = len(classes[j]) - 1
        #             centers[j] = (centers[j] * old_size + weighted_row) / (old_size + 1)
        #     else:
        #         # 没有匹配的类，创建新类
        #         classes.append([i])
        #         centers.append(weighted_row.copy())

        self._log(f"    行相似度聚类（原始）: {len(classes)} 个类")

        # 过滤小类，返回每个类的DataFrame，并附加weights
        result_dfs = []
        filtered_out = []
        for i, class_indices in enumerate(classes):
            if len(class_indices) >= self.min_cluster_size:
                cluster_df = df.iloc[class_indices].copy()
                # 将weights存储到attrs中，传递给后续SVD
                cluster_df.attrs['column_weights'] = weights.copy()
                cluster_df.attrs['numeric_cols'] = numeric_cols.copy()
                result_dfs.append(cluster_df)
            else:
                filtered_out.append(i + 1)

        if filtered_out:
            self._log(f"    被过滤的小类（<{self.min_cluster_size}行）: {filtered_out}")

        self._log(f"    行相似度聚类（过滤后）: {len(result_dfs)} 个类")
        return result_dfs

    # ==================== 字段相关聚类与规则发现主函数 ====================
    def _cluster_columns_and_rules1(self, df: pd.DataFrame, numeric_cols: List[str],
                              min_nonnull_in_cluster: int = 10,
                              min_nonnull_ratio_in_cluster: float = 0.05) -> List[Dict]:
        """
        执行路径一（相关聚类），在聚类子集上发现关系。
        增强：利用行聚类时计算的 column_weights 筛选出高权重的稀疏字段，
        仅对这些字段进行关系发现（不加权 SVD），避免稠密字段干扰。
        """
        cluster_rows = len(df)

        # 获取权重（从行聚类传递过来）
        weights = df.attrs.get('column_weights', None)
        all_numeric_cols = df.attrs.get('numeric_cols', numeric_cols)

        # # 过滤稀疏字段（在行聚类中非空率太低的字段）
        # filtered_cols = []
        # for col in numeric_cols:
        #     if col not in df.columns:
        #         continue
        #     # 同时要求该字段在高权重列表中（如果高权重列表不是全量的话）
        #     if col not in top_cols:
        #         continue
        #     filtered_cols.append(col)
        #     # nonnull_count = df[col].notna().sum()
        #     # if nonnull_count >= min_nonnull_in_cluster or nonnull_count / cluster_rows >= min_nonnull_ratio_in_cluster:
        #     #     filtered_cols.append(col)
        #
        # if len(filtered_cols) < 3:
        #     self._log(f"        过滤后字段不足3个，跳过")
        #     return []

        rules = []
        # 相关聚类（使用过滤后的字段）
        corr_clusters = self._cluster_columns_by_correlation(df, all_numeric_cols)  # filtered_cols)
        self._log(f"    行类: {len(df)}行,字段聚类结果（共{len(corr_clusters)}个）:")

        for idx, cluster in enumerate(corr_clusters):
            if len(cluster) < 3:
                continue
            self._log(f"        字段聚类: {len(cluster)}个变量 - {cluster}")

            #不拆
            # sub_rules = self._discover_rules_in_subcluster(df, cluster, weights=weights,
            #                                                all_numeric_cols=all_numeric_cols)
            # if len(sub_rules) > 0:
            #     self._log(f"          规则:{sub_rules}")
            #     rules.extend(sub_rules)


            # 检查所有字段同时非空的行数
            cooccur = self._check_cooccurrence(df, cluster)

            # 如果共同非空行数 >= 100，或者字段数 <= 10，不拆分
            if cooccur >= 100 or len(cluster) <= 10:
                # 🔥 关键：不传递 weights，使用普通 SVD
                sub_rules = self._discover_rules_in_subcluster(df, cluster, weights=weights,
                                                               all_numeric_cols=all_numeric_cols)
                if len(sub_rules) > 0:
                    self._log(f"          规则:{sub_rules}")
                    rules.extend(sub_rules)
            else:
                # 否则按共同非空比例拆分子类
                subclusters = self._split_by_cooccurrence_ratio(df, cluster)
                for sub_idx, sub in enumerate(subclusters):
                    if len(sub) < 3:
                        continue
                    sub_rules = self._discover_rules_in_subcluster(df, sub, weights=weights,
                                                                   all_numeric_cols=all_numeric_cols)
                    if len(sub_rules) > 0:
                        self._log(f"          规则:{sub_rules}")
                        rules.extend(sub_rules)

        return rules

    # 执行完后再过滤后再次执行
    def _cluster_columns_and_rules(self, df: pd.DataFrame, numeric_cols: List[str],
                                   min_nonnull_in_cluster: int = 10,
                                   min_nonnull_ratio_in_cluster: float = 0.05) -> List[Dict]:
        """
        执行相关聚类，在聚类子集上发现关系。
        增强：利用行聚类时计算的 column_weights 筛选出高权重的稀疏字段，
        仅对这些字段进行关系发现（不加权 SVD），避免稠密字段干扰。
        再增加：第一次发现规则后，对每个字段聚类，剔除规则中与当前聚类重叠 ≥3 的字段，
               然后在剩余字段上再次发现规则（使用全局规则）。
        """
        cluster_rows = len(df)

        # 获取权重（从行聚类传递过来）
        weights = df.attrs.get('column_weights', None)
        all_numeric_cols = df.attrs.get('numeric_cols', numeric_cols)

        # 相关聚类
        corr_clusters = self._cluster_columns_by_correlation(df, all_numeric_cols)
        self._log(f"    行类: {len(df)}行,字段聚类结果（共{len(corr_clusters)}个）:")

        # ========== 第一阶段：收集所有第一次发现的规则 ==========
        first_rules_all = []  # 存储所有第一次发现的规则（全局）
        cluster_info = []  # 存储每个 cluster 的原始字段列表和拆分信息，供第二阶段使用

        for idx, cluster in enumerate(corr_clusters):
            if len(cluster) < 3:
                continue
            self._log(f"        字段聚类: {len(cluster)}个变量 - {cluster}")

            # 检查所有字段同时非空的行数
            cooccur = self._check_cooccurrence(df, cluster)

            # 第一次发现规则
            first_rules = []
            if cooccur >= 100 or len(cluster) <= 10:
                sub_rules = self._discover_rules_in_subcluster(df, cluster, weights=weights,
                                                               all_numeric_cols=all_numeric_cols)
                first_rules.extend(sub_rules)
            else:
                subclusters = self._split_by_cooccurrence_ratio(df, cluster)
                for sub in subclusters:
                    if len(sub) < 3:
                        continue
                    sub_rules = self._discover_rules_in_subcluster(df, sub, weights=weights,
                                                                   all_numeric_cols=all_numeric_cols)
                    first_rules.extend(sub_rules)

            if first_rules:
                self._log(f"          第一次发现规则:{first_rules}")
                first_rules_all.extend(first_rules)

            # 保存该 cluster 的相关信息，供第二阶段使用
            cluster_info.append({
                'cluster': cluster,
                'cooccur': cooccur,
                'first_rules': first_rules  # 可保留，但第二阶段基于全局规则
            })

        # ========== 第二阶段：基于全局规则，剔除字段后再次发现 ==========
        # 汇总所有第一次规则（去重，基于字段集合）
        seen_global = set()
        global_rules = []
        for rule in first_rules_all:
            key = frozenset(rule['fields'])
            if key not in seen_global:
                seen_global.add(key)
                global_rules.append(rule)

        # 收集需要剔除的字段（按 cluster 分别计算）
        all_rules = first_rules_all[:]  # 最终结果先包含第一次规则

        for info in cluster_info:
            cluster = info['cluster']
            cooccur = info['cooccur']

            # 基于全局规则，找出需要从该 cluster 中剔除的字段
            remove_fields = set()
            for rule in global_rules:
                rule_fields = set(rule.get('fields', []))
                overlap = len(rule_fields & set(cluster))
                if overlap >= 3:
                    remove_fields.update(rule_fields)

            if not remove_fields:
                self._log(f"        字段聚类 {cluster} 无剔除字段，跳过第二次发现")
                continue

            remaining_fields = [f for f in cluster if f not in remove_fields]
            self._log(f"        字段聚类 {cluster} 剔除 {remove_fields} 后剩余: {remaining_fields}")

            if len(remaining_fields) < 3:
                self._log(f"          剩余字段不足3个，跳过第二次发现")
                continue

            # 第二次发现规则（使用相同的拆分策略）
            second_rules = []
            if cooccur >= 100 or len(remaining_fields) <= 10:
                sub_rules = self._discover_rules_in_subcluster(df, remaining_fields,
                                                               weights=weights,
                                                               all_numeric_cols=all_numeric_cols)
                second_rules.extend(sub_rules)
            else:
                subclusters = self._split_by_cooccurrence_ratio(df, remaining_fields)
                for sub in subclusters:
                    if len(sub) < 3:
                        continue
                    sub_rules = self._discover_rules_in_subcluster(df, sub, weights=weights,
                                                                   all_numeric_cols=all_numeric_cols)
                    second_rules.extend(sub_rules)

            if second_rules:
                self._log(f"          第二次发现规则:{second_rules}")
                # 合并去重
                existing_keys = {frozenset(r.get('fields', [])) for r in all_rules}
                for r in second_rules:
                    key = frozenset(r.get('fields', []))
                    if key not in existing_keys:
                        existing_keys.add(key)
                        all_rules.append(r)

        return all_rules

    # ==================== 字段相关聚类 ====================
    def _cluster_columns_by_correlation(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[List[str]]:
        """
        按相关系数构建图，取连通分量作为聚类
        """
        if len(numeric_cols) < 2:
            return [[col] for col in numeric_cols]

        G = nx.Graph()
        G.add_nodes_from(numeric_cols)

        corr_matrix = df[numeric_cols].corr().abs()
        edge_count = 0

        # 1. 构建直接相连的边（1跳）
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                corr_val = corr_matrix.loc[col1, col2]
                if not pd.isna(corr_val) and corr_val > self.corr_threshold:
                    G.add_edge(col1, col2)
                    edge_count += 1

        # 2. 添加2跳连通关系（距离为2的节点也直接连边）
        # 方法：计算图的平方（graph square）
        nodes = list(G.nodes())
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                # 检查是否存在长度为2的路径 u - x - v
                if nx.has_path(G, u, v) and nx.shortest_path_length(G, u, v) == 2:
                    G.add_edge(u, v)


        components = list(nx.connected_components(G))
        clusters = [list(comp) for comp in components]
        #self._log(f"    相关图: {len(numeric_cols)} 节点, {edge_count} 边, 阈值={self.corr_threshold},相关聚类数: {len(clusters)}")
        return clusters

    # ==================== 按共同非空比例拆分子类 ====================
    def _split_by_cooccurrence_ratio(self, df: pd.DataFrame, cluster: List[str]) -> List[List[str]]:
        """按共同非空比例拆分子类"""
        if len(cluster) <= 3:
            return [cluster]

        nonnull_counts = {col: df[col].notna().sum() for col in cluster}
        sorted_cols = sorted(cluster, key=lambda x: nonnull_counts[x], reverse=True)
        subclusters = []

        for seed in sorted_cols:
            subcluster = [seed]
            seed_count = nonnull_counts[seed]

            for other in sorted_cols:
                if other == seed:
                    continue

                test_fields = subcluster + [other]
                co = self._check_cooccurrence(df, test_fields)
                min_count = min(seed_count, min(nonnull_counts[o] for o in subcluster), nonnull_counts[other])

                if  co >= self.min_cooccurrence_rows: #co >= self.cooccur_ratio * min_count and
                    subcluster.append(other)

            if len(subcluster) >= 3:
                # 去重
                is_dup = False
                for existing in subclusters:
                    if set(subcluster) == set(existing):
                        is_dup = True
                        break
                if not is_dup:
                    subclusters.append(subcluster)

        return subclusters

    # ==================== 关系发现 ====================
    #继承权重
    def _discover_rules_in_subcluster(self, df: pd.DataFrame, fields: List[str],
                                      weights: np.ndarray = None,
                                      all_numeric_cols: List[str] = None) -> List[Dict]:
        """
        在子类中发现关系，支持逐步剔除低权重字段的循环 SVD。
        每次剔除权重最低的字段，对剩余字段做普通 SVD，收集所有发现的规则。
        """

        if len(fields) < 3:
            return []

        self._log(f"调用前: weights= {weights}, all_numeric_cols= {all_numeric_cols}")

        # 如果没有权重信息，则直接做一次普通 SVD
        if weights is None or all_numeric_cols is None:
            return self._discover_rules_single_svd(df, fields)

        # 构建当前字段列表及其对应权重
        current_fields = fields.copy()
        # 获取每个字段的权重
        field_weight_dict = {f: weights[all_numeric_cols.index(f)] for f in current_fields if f in all_numeric_cols}
        if len(field_weight_dict) != len(current_fields):
            self._log(f" weights不正常:{weights}，field_weight_dict：{field_weight_dict}，current_fields：{current_fields}")
            # 权重不全，退回普通 SVD
            return self._discover_rules_single_svd(df, fields)

        all_rules = []  # 收集所有发现的规则
        seen_rules = set()  # 用于去重（基于字段集合）

        # 逐步剔除权重最低的字段，直到字段数 < 3
        while len(current_fields) >= 3:
            # 对当前字段集做普通 SVD（不加权）
            rules = self._discover_rules_single_svd(df, current_fields)

            for rule in rules:
                # 使用规则涉及的字段集合作为去重key
                rule_key = frozenset(rule.get('fields', []))
                if rule_key not in seen_rules:
                    seen_rules.add(rule_key)
                    all_rules.append(rule)
                    self._log(
                        f"        发现规则: {rule['rule']} (置信度={rule['confidence']})，当前字段数={len(current_fields)}")
            #break #只跑一次

            # 如果剩余字段数 <= 3，无法继续剔除（因为至少需要3个字段）
            if len(current_fields) <= 3:
                break

            # 找到当前字段集中权重最小的字段（IDF 最小，最稠密）
            min_weight_field = min(current_fields, key=lambda f: field_weight_dict[f])
            # 移除该字段
            current_fields.remove(min_weight_field)
            # self._log(
            #     f"        剔除低权重字段: {min_weight_field} (权重={field_weight_dict[min_weight_field]:.3f})，剩余 {len(current_fields)} 个字段")

        # 反向剔除，效果不好
        # # 逐步剔除权重最高的字段，直到字段数 < 3
        # while len(current_fields) >= 3:
        #     # 对当前字段集做普通 SVD（不加权）
        #     rules = self._discover_rules_single_svd(df, current_fields)
        #
        #     for rule in rules:
        #         # 使用规则涉及的字段集合作为去重key
        #         rule_key = frozenset(rule.get('fields', []))
        #         if rule_key not in seen_rules:
        #             seen_rules.add(rule_key)
        #             all_rules.append(rule)
        #             self._log(
        #                 f"        发现规则: {rule['rule']} (置信度={rule['confidence']})，当前字段数={len(current_fields)}")
        #     # break #只跑一次
        #
        #     # 如果剩余字段数 <= 3，无法继续剔除（因为至少需要3个字段）
        #     if len(current_fields) <= 3:
        #         break
        #
        #     # 找到当前字段集中权重最大的字段（IDF 最小，最稠密）
        #     max_weight_field = max(current_fields, key=lambda f: field_weight_dict[f])   ###仅此不同
        #     # 移除该字段
        #     current_fields.remove(max_weight_field)  ###
        #     # self._log(
        #     #     f"        剔除高权重字段: {min_weight_field} (权重={field_weight_dict[min_weight_field]:.3f})，剩余 {len(current_fields)} 个字段")

        # 返回所有收集到的规则
        return all_rules


    #只对当前类计算权重,没啥用，基本上都是【1 1 1.。。】
    def _discover_rules_in_subcluster2(self, df: pd.DataFrame, fields: List[str],
                                      weights: np.ndarray = None,
                                      all_numeric_cols: List[str] = None) -> List[Dict]:
        """
        在子类中发现关系，支持逐步剔除低权重字段的循环 SVD。
        每次剔除权重最低的字段，对剩余字段做普通 SVD，收集所有发现的规则。

        注意：权重基于当前 df（行聚类子集）重新计算，不使用传入的 weights 参数。
        """
        if len(fields) < 3:
            return []

        # ========== 新增：基于当前类重新计算权重 ==========
        # 获取当前类中的数值字段（只取 fields 中存在于 df 的列）
        available_fields = [f for f in fields if f in df.columns]
        if len(available_fields) < 3:
            return []

        # 构建当前类的0-1缺失矩阵
        X_binary = df[available_fields].notna().astype(int).values
        total_rows = X_binary.shape[0]

        if total_rows < self.min_cluster_size:
            return []

        # 计算当前类的IDF权重
        field_freq = X_binary.sum(axis=0)
        idf = np.log((total_rows + 1) / (field_freq + 1)) + 1
        # 使用三次方放大稀疏字段权重
        local_weights = idf

        # 构建当前字段的权重字典
        current_fields = available_fields.copy()
        field_weight_dict = {current_fields[i]: local_weights[i] for i in range(len(current_fields))}

        self._log(
            f"        基于当前类重新计算权重（{local_weights}）")
        # ================================================

        all_rules = []  # 收集所有发现的规则
        seen_rules = set()  # 用于去重（基于字段集合）

        # 逐步剔除权重最低的字段，直到字段数 < 3
        while len(current_fields) >= 3:
            # 对当前字段集做普通 SVD（不加权）
            rules = self._discover_rules_single_svd(df, current_fields)

            for rule in rules:
                # 使用规则涉及的字段集合作为去重key
                rule_key = frozenset(rule.get('fields', []))
                if rule_key not in seen_rules:
                    seen_rules.add(rule_key)
                    all_rules.append(rule)
                    self._log(
                        f"        发现规则: {rule['rule']} (置信度={rule['confidence']})，当前字段数={len(current_fields)}")

            # 如果剩余字段数 <= 3，无法继续剔除（因为至少需要3个字段）
            if len(current_fields) <= 3:
                break

            # 找到当前字段集中权重最小的字段（IDF 最小，最稠密）
            min_weight_field = min(current_fields, key=lambda f: field_weight_dict[f])
            # 移除该字段
            current_fields.remove(min_weight_field)
            self._log(
                f"        剔除低权重字段: {min_weight_field} (权重={field_weight_dict[min_weight_field]:.3f})，剩余 {len(current_fields)} 个字段")

        # 返回所有收集到的规则
        return all_rules

    def _discover_rules_single_svd(self, df: pd.DataFrame, fields: List[str]) -> List[Dict]:
        """
        对指定的字段集合做一次普通 SVD（不加权），返回发现的规则列表。
        该函数复用原有 SVD 逻辑（去掉加权部分）。
        """
        if len(fields) < 3:
            return []

        valid_df = self._get_valid_data(df, fields)
        valid_rows = len(valid_df)
        if valid_rows < self.min_cooccurrence_rows:
            return []

        X = valid_df[fields].values
        n_cols = len(fields)

        # 确定采样大小（全量或随机）
        actual_sample_size = max(self.ransac_sample_size, n_cols + 1)
        if actual_sample_size > valid_rows:
            actual_sample_size = valid_rows

        best_coeffs = None
        best_inlier_mask = None
        best_inlier_count = 0
        best_sv_ratio = 0

        # RANSAC 迭代
        for _ in range(self.ransac_iter):
            # if valid_rows <= actual_sample_size:
            #     sample_idx = np.arange(valid_rows)
            # else:
            #     sample_idx = np.random.choice(valid_rows, actual_sample_size, replace=False)

            sample_idx = np.arange(valid_rows)
            X_sample = X[sample_idx]

            try:
                X_sample_centered = X_sample - np.mean(X_sample, axis=0)
                U, s, Vt = np.linalg.svd(X_sample_centered, full_matrices=False)
                coeffs = Vt[-1, :]

                if np.all(np.abs(coeffs) < self.precision):
                    continue

                result = X @ coeffs
                rel_error = np.abs(result) / (np.abs(X[:, 0]) + 1)
                inlier_mask = rel_error < 0.01
                inlier_count = inlier_mask.sum()
                sv_ratio = s[-1] / s[0] if s[0] > 0 else 1

                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    best_coeffs = coeffs
                    best_inlier_mask = inlier_mask
                    best_sv_ratio = sv_ratio

                    if best_inlier_count >= valid_rows * self.inlier_ratio:
                        break
            except Exception:
                continue

        if best_coeffs is None or best_inlier_count < valid_rows * 0.5:
            return []

        # 用内点重新 SVD
        inlier_X = X[best_inlier_mask]
        inlier_X_centered = inlier_X - np.mean(inlier_X, axis=0)
        U, s, Vt = np.linalg.svd(inlier_X_centered, full_matrices=False)
        coeffs = Vt[-1, :]

        coeffs_sign = np.zeros(len(coeffs), dtype=int)
        for i, c in enumerate(coeffs):
            if c > self.precision:
                coeffs_sign[i] = 1
            elif c < -self.precision:
                coeffs_sign[i] = -1

        nonzero_indices = [i for i, c in enumerate(coeffs_sign) if c != 0]
        if len(nonzero_indices) < 3:
            return []

        # 按系数绝对值分组
        groups = []
        used = set()
        for i in nonzero_indices:
            if i in used:
                continue
            target_abs = abs(coeffs[i])
            group = [i]
            for j in nonzero_indices:
                if j != i and j not in used:
                    current_abs = abs(coeffs[j])
                    if target_abs > 0 and abs(current_abs - target_abs) / target_abs < 0.05:
                        group.append(j)
            if len(group) >= 3:
                groups.append(group)
                used.update(group)

        if not groups:
            return []

        rules = []
        for group in groups:
            subset_fields = [fields[i] for i in group]
            subset_coeffs = np.array([coeffs_sign[i] for i in group])
            subset_X = X[:, group]

            result = subset_X @ subset_coeffs
            scale = np.max(np.abs(subset_X), axis=1)
            scale = np.maximum(scale, 1)
            confidence = (np.abs(result) / scale < 1e-4).mean()

            if confidence < self.min_confidence:
                continue

            left_parts = []
            right_parts = []
            for field, coeff in zip(subset_fields, subset_coeffs):
                if coeff > 0:
                    right_parts.append(field)
                elif coeff < 0:
                    left_parts.append(field)

            if not left_parts:
                left_parts = ["0"]
            if not right_parts:
                right_parts = ["0"]

            left_expr = " + ".join(left_parts)
            right_expr = " + ".join(right_parts)
            expr = f"{left_expr} = {right_expr}"

            rules.append({
                "rule": expr,
                "confidence": round(confidence, 4),
                "priority": "高" if confidence == 1.0 else "中",
                "fields": subset_fields,
                "relation_type": "additive",
                "violation_count": 0,
                "violation_samples": []
            })
        return rules


    def _get_valid_data(self, df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
        """获取指定字段都非空的数据"""
        valid_mask = df[fields].notna().all(axis=1)
        return df[valid_mask][fields].copy()

    def _check_cooccurrence(self, df: pd.DataFrame, fields: List[str]) -> int:
        """检查一组字段共同非空的行数"""
        if not fields:
            return 0
        valid_mask = df[fields].notna().all(axis=1)
        return valid_mask.sum()

def discover_audit_rules(data: pd.DataFrame, variable_types: Dict[str, str],
                         foreign_keys: List[Dict] = None,
                         debug: bool = False, **kwargs) -> Dict[str, Any]:
    """便捷函数：发现勾稽规则"""
    discoverer = AuditRuleDiscoverer(debug=debug, **kwargs)
    return discoverer.discover_all(data, variable_types, foreign_keys)


def verify_audit_rules(data: pd.DataFrame,
                       rules: List[Tuple[List[str], List[int]]],
                       debug: bool = False) -> List[Dict]:
    """
    便捷函数：验证指定的勾稽规则

    参数:
    - data: 数据框
    - rules: 规则列表，每个元素为 (fields, coeffs)
             例如: (['A', 'B', 'C', 'D'], [1, 1, -1, -1]) 表示 A + B = C + D

    返回: 验证结果列表
    """
    results = []
    for fields, coeffs in rules:
        valid_mask = data[fields].notna().all(axis=1)
        valid_df = data[valid_mask][fields]
        valid_rows = len(valid_df)
        if valid_rows == 0:
            results.append({"success": False, "error": "无有效数据", "fields": fields})
            continue
        X = valid_df.values
        result = X @ np.array(coeffs)
        scale = np.max(np.abs(X), axis=1)
        scale = np.maximum(scale, 1)
        confidence = (np.abs(result) / scale < 1e-4).mean()
        violations = (np.abs(result) / scale >= 1e-4).sum()
        left = [f for f, c in zip(fields, coeffs) if c < 0]
        right = [f for f, c in zip(fields, coeffs) if c > 0]
        expr = f"{' + '.join(left)} = {' + '.join(right)}"
        results.append({
            "success": True,
            "rule": expr,
            "fields": fields,
            "coeffs": coeffs,
            "confidence": round(confidence, 4),
            "valid_rows": valid_rows,
            "violations": violations
        })
    return results