# autostat/core/audit.py
"""勾稽关系发现模块 - 四条路径：相关聚类 + 共同非空聚类 + 非空数相近聚类 + 行相似度聚类"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
import math
# 新增导入随机 SVD
try:
    from sklearn.utils.extmath import randomized_svd
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class AuditRuleDiscoverer:
    """勾稽规则发现器 - 四条路径"""

    def __init__(self,
                 # ========== 1. 基础参数 ==========
                 debug: bool = False,  # 是否开启调试模式，输出详细日志

                 # ========== 2. 数值精度参数 ==========
                 precision: float = 1e-6,  # 数值比较精度，判断两个数值是否相等

                 # ========== 3. 置信度参数 ==========
                 min_confidence: float = 0.5,  # 最小置信度阈值，低于此值的规则不输出

                 # ========== 4. 字段筛选参数 ==========
                 corr_threshold: float = 0.9,  # 相关性阈值，用于筛选强相关字段对
                 min_nonnull_count: int = 10,  # 最小非空值数量，字段非空值少于此时不考虑
                 min_nonnull_rate: float = 0.01,  # 最小非空率，字段非空率低于此值不考虑
                 cooccur_ratio: float = 0.9,  # 共现比例阈值，两字段同时非空的比例需达到此值
                 min_cooccurrence_rows: int = 10,  # 最小共现行数，两字段同时非空的行数需大于此值
                 max_numeric_fields: int = 100,  # 最大数值字段数量，限制分析的字段数量

                 # ========== 5. 聚类分析参数 ==========
                 nonnull_diff_ratio: float = 0.2,  # 非空差异比例，用于聚类分析
                 row_similarity_threshold: float = 0.99,  # 行相似度阈值，用于判断数据模式是否相同
                 min_cluster_size: int = 3,  # 最小聚类大小，聚类至少需要包含的行数

                 # ========== 6. 相等关系参数 ==========
                 equality_ratio_threshold: float = 0.8,  # 两字段相等比例阈值，判断是否满足相等关系
                 strong_coverage_threshold: float = 0.8,  # 强相等覆盖比例阈值，用于验证强相等关系

                 # ========== 7. RANSAC拟合参数 ==========
                 ransac_iter: int = 20,  # RANSAC迭代次数，用于线性关系拟合
                 ransac_sample_size: int = 20,  # RANSAC采样大小，每次迭代随机抽取的样本数
                 inlier_ratio: float = 0.7,  # 内点比例阈值，RANSAC拟合时的内点比例要求

                 # ========== 8. 线性关系验证参数 ==========
                 min_final_inlier_ratio: float = 0.5,  # 最终最小内点比例，验证规则时的内点比例要求
                 coeff_group_tolerance: float = 0.5,  # 系数分组容差，将相近系数归为同一组的容差范围
                 inlier_error_threshold: float = 0.01,  # 内点误差阈值，判断拟合是否准确的误差上限

                 # ========== 9. 异常处理参数 ==========
                 max_remove_ratio: float = 0.3,  # 迭代剔除最大比例，最大可剔除的异常值比例
                 small_sample_threshold: int = 50  # 小样本复制阈值，小于此值视为小样本，采用复制策略
                 ):
        """
        勾稽规则发现器初始化
        """
        # 1. 基础参数
        self.debug = debug

        # 2. 数值精度参数
        self.precision = precision

        # 3. 置信度参数
        self.min_confidence = min_confidence

        # 4. 字段筛选参数
        self.corr_threshold = corr_threshold
        self.min_nonnull_count = min_nonnull_count
        self.min_nonnull_rate = min_nonnull_rate
        self.cooccur_ratio = cooccur_ratio
        self.min_cooccurrence_rows = min_cooccurrence_rows
        self.max_numeric_fields = max_numeric_fields

        # 5. 聚类分析参数
        self.nonnull_diff_ratio = nonnull_diff_ratio
        self.row_similarity_threshold = row_similarity_threshold
        self.min_cluster_size = min_cluster_size

        # 6. 相等关系参数
        self.equality_ratio_threshold = equality_ratio_threshold
        self.strong_coverage_threshold = strong_coverage_threshold

        # 7. RANSAC拟合参数
        self.ransac_iter = ransac_iter
        self.ransac_sample_size = ransac_sample_size
        self.inlier_ratio = inlier_ratio

        # 8. 线性关系验证参数
        self.min_final_inlier_ratio = min_final_inlier_ratio
        self.coeff_group_tolerance = coeff_group_tolerance
        self.inlier_error_threshold = inlier_error_threshold

        # 9. 异常处理参数
        self.max_remove_ratio = max_remove_ratio
        self.small_sample_threshold = small_sample_threshold

    def _log(self, msg: str):
        if self.debug:
            print(f"  [DEBUG] {msg}")

    # ==================== 公共工具函数 ====================
    # ==================== 新增：向量化共现矩阵 ====================
    def _compute_cooccurrence_matrix(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        """使用矩阵乘法一次性计算所有列对的共现次数"""
        # 构造布尔矩阵：行=样本，列=字段，值为是否非空
        nonnull_bool = df[cols].notna().values.astype(int)
        # 共现矩阵 = 转置矩阵 × 原矩阵，结果 (n_cols x n_cols)
        cooccur = nonnull_bool.T @ nonnull_bool
        return cooccur

    def _parse_rule_string(self, rule_dict: Dict) -> Tuple[List[str], List[str], float]:
        """解析规则字符串，返回 (left_fields, right_fields, confidence)"""
        expr = rule_dict['rule']
        left, right = expr.split(' = ')
        left_fields = [f.strip() for f in left.split(' + ')] if left != '0' else []
        right_fields = [f.strip() for f in right.split(' + ')] if right != '0' else []
        return left_fields, right_fields, rule_dict.get('confidence', 1.0)

    def _build_rule_dict(self, left_fields: List[str], right_fields: List[str],
                         confidence: float, priority: str = None) -> Dict:
        """根据左右字段列表构建规则字典"""
        left_fields.sort()
        right_fields.sort()
        left_expr = ' + '.join(left_fields) if left_fields else '0'
        right_expr = ' + '.join(right_fields) if right_fields else '0'
        rule_str = f"{left_expr} = {right_expr}"
        all_fields = left_fields + right_fields
        if priority is None:
            priority = '高' if confidence == 1.0 else '中'
        return {
            'rule': rule_str,
            'fields': all_fields,
            'confidence': round(confidence, 4),
            'priority': priority,
            'relation_type': 'additive',
            'violation_count': 0,
            'violation_samples': []
        }

    def _deduplicate_rules(self, rules: List[Dict]) -> List[Dict]:
        """基于字段集合去重"""
        seen = set()
        result = []
        for rule in rules:
            key = frozenset(rule.get('fields', []))
            if key not in seen:
                seen.add(key)
                result.append(rule)
        return result

    def _normalize_two_field_rule(self, fields: List[str]) -> Tuple[str, List[str]]:
        """将2字段规则格式化为 小字段 = 大字段，返回 (rule_str, [small, large])"""
        f1, f2 = fields[0], fields[1]
        if f1 < f2:
            return f"{f1} = {f2}", [f1, f2]
        else:
            return f"{f2} = {f1}", [f2, f1]

    def _build_replace_map(self, strong_equality: List[Dict]) -> Dict[str, str]:
        """构建字段替换映射（大字段 -> 小字段）"""
        replace_map = {}
        for r in strong_equality:
            fields = r.get('fields', [])
            if len(fields) != 2:
                continue
            small, large = fields[0], fields[1]
            replace_map[large] = small

        changed = True
        while changed:
            changed = False
            for k, v in list(replace_map.items()):
                if v in replace_map and replace_map[v] != v:
                    replace_map[k] = replace_map[v]
                    changed = True
        return replace_map

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


    # ==================== 主流程====================
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
        # date_cols = [col for col, typ in variable_types.items()
        #              if typ == 'datetime' and col in data.columns
        #              and col in self._keep_cols]
        # self._log(f"日期变量: {len(date_cols)} 个")
        #
        # if len(date_cols) >= 2:
        #     temporal_rules = self._discover_temporal_rules(data, date_cols)
        #     rules["temporal_rules"] = temporal_rules
        #     self._log(f"发现 {len(temporal_rules)} 条时序关系")

        # 时序关系发现（已由 date_rules.py 负责，此处跳过）
        # 保留空列表，避免后续报错
        rules["temporal_rules"] = []

        self._log("=" * 60)
        return rules



    # ==================== 数值规则发现 ====================
    # ==================== 修改 _discover_arithmetic_rules 中调用 ====================
    def _discover_arithmetic_rules(self, df: pd.DataFrame,
                                   numeric_cols: List[str]) -> List[Dict]:
        """
        数值规则发现（使用向量化共现矩阵）
        """
        self._log("  [数值关系] 开始...")

        if len(numeric_cols) < 3:
            return []

        # 在原始全量数据上发现2字段相等关系
        strong_equality, weak_equality = self._discover_equality_rules(df, numeric_cols)

        all_rules = []
        all_rules.extend(weak_equality)

        self._log("\n  【行相似度聚类】")
        row_clusters = self._cluster_rows_by_similarity(df, numeric_cols)

        if self.debug and len(row_clusters) > 0:
            self._log(f"行聚类结果（共{len(row_clusters)}个行类）:")

        rules4 = []
        for i, cluster_df in enumerate(row_clusters):
            self._log(f"处理行聚类{i + 1}: {len(cluster_df)} 行")
            sub_rules = self._cluster_columns_and_rules(
                cluster_df, numeric_cols,
                min_nonnull_in_cluster=self.min_cluster_size,
                min_nonnull_ratio_in_cluster=0.01,
                strong_equality=strong_equality
            )
            rules4.extend(sub_rules)

        all_rules.extend(rules4)
        self._log(f"    发现规则数: {len(rules4)}")

        all_rules.extend(strong_equality)

        # 去重
        unique_rules = self._deduplicate_rules(all_rules)
        self._log(f"\n    合并后总计: {len(unique_rules)} 条数值关系")

        # 全量验证
        validated_rules = self._validate_rules_on_full_data(df, unique_rules)

        if self.debug:
            self._log("\n  【全量验证后的规则列表】")
            for i, rule in enumerate(validated_rules, 1):
                self._log(
                    f"    {i}. {rule['rule']} (置信度={rule['confidence']}, 有效行数={rule['valid_rows']}, 满足行数={rule['satisfied_rows']})")

        return validated_rules

    # ==================== 2字段相等规则发现 ====================
    def _discover_equality_rules(self, df: pd.DataFrame, numeric_cols: List[str]):
        """
        发现2字段相等关系：A = B
        注意：valid_rows/satisfied_rows 等由后续全量验证重新计算
        """
        strong_rules = []
        weak_rules = []
        n = len(numeric_cols)

        nonnull_counts = {col: df[col].notna().sum() for col in numeric_cols}

        for i in range(n):
            for j in range(i + 1, n):
                col1 = numeric_cols[i]
                col2 = numeric_cols[j]

                n1 = nonnull_counts[col1]
                n2 = nonnull_counts[col2]

                valid_mask = df[col1].notna() & df[col2].notna()
                common_nonnull = valid_mask.sum()

                if common_nonnull < self.min_cooccurrence_rows:
                    continue

                equal_mask = df.loc[valid_mask, col1] == df.loc[valid_mask, col2]
                equal_count = equal_mask.sum()
                equal_ratio = equal_count / common_nonnull

                if equal_ratio < self.equality_ratio_threshold:
                    continue

                rule_str, fields = self._normalize_two_field_rule([col1, col2])

                rule = {
                    'rule': rule_str,
                    'fields': fields,
                    'confidence': round(equal_ratio, 4),
                    'priority': '高' if equal_ratio == 1.0 else '中',
                    'relation_type': 'additive',
                    'violation_count': 0,
                    'violation_samples': [],
                    'valid_rows': 0,
                    'satisfied_rows': 0
                }

                if (common_nonnull >= self.strong_coverage_threshold * n1 and
                        common_nonnull >= self.strong_coverage_threshold * n2):
                    strong_rules.append(rule)
                else:
                    weak_rules.append(rule)

        return strong_rules, weak_rules

    # ==================== 行相似度聚类 ====================
    def _cluster_rows_by_similarity(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[pd.DataFrame]:
        """
        基于0-1矩阵的行相似度在线聚类（加权余弦相似度，使用IDF权重）
        """
        if len(df) < self.min_cluster_size:
            return []

        X_binary = df[numeric_cols].notna().astype(int).values
        total_rows, n_fields = X_binary.shape

        field_freq = X_binary.sum(axis=0)
        idf = np.log((total_rows + 1) / (field_freq + 1)) + 1
        self._log(f"    IDF1: {idf}")

        #weights=self._calculate_idf_weights(X_binary, numeric_cols)  #自适应IDF权重映射
        weights = idf ** 3
        self._log(f"    weights：{weights}")

        # 使用单归属聚类
        classes, centers = self._cluster_rows_single(X_binary, weights)
        #classes, centers = self._cluster_rows_multi(X_binary, weights)  #多归属聚类

        self._log(f"    行相似度聚类（原始）: {len(classes)} 个类")

        result_dfs = []
        filtered_out = []
        for i, class_indices in enumerate(classes):
            if len(class_indices) >= self.min_cluster_size:
                cluster_df = df.iloc[class_indices].copy()
                cluster_df.attrs['column_weights'] = weights.copy()
                cluster_df.attrs['numeric_cols'] = numeric_cols.copy()
                result_dfs.append(cluster_df)
            else:
                filtered_out.append(i + 1)

        if filtered_out:
            self._log(f"    被过滤的小类（<{self.min_cluster_size}行）: {filtered_out}")

        self._log(f"    行相似度聚类（过滤后）: {len(result_dfs)} 个类")
        return result_dfs

    def _calculate_idf_weights(self, X_binary: np.ndarray, numeric_cols: List[str]) -> np.ndarray:
        """
        自适应IDF权重映射（备用）：使用指数映射替代IDF³
        """
        total_rows, n_fields = X_binary.shape
        field_freq = X_binary.sum(axis=0)
        idf = np.log((total_rows + 1) / (field_freq + 1)) + 1

        idf_min = idf.min()
        idf_max = idf.max()
        if idf_max == idf_min:
            return np.ones_like(idf)

        n_fields_count = len(numeric_cols)
        max_multiplier = min(50, 10 * np.log2(max(n_fields_count, 2)))
        idf_norm = (idf - idf_min) / (idf_max - idf_min)
        k = 4.0
        exp_scale = (np.exp(k * idf_norm) - 1) / (np.exp(k) - 1)
        weights = 1 + exp_scale * (max_multiplier - 1)
        return weights

    def _cluster_rows_single(self, X_binary: np.ndarray, weights: np.ndarray) -> Tuple[
        List[List[int]], List[np.ndarray]]:
        """
        单归属行聚类：每行只归入最相似的一个类
        返回 (classes, centers)
        """
        classes = []
        centers = []

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

        return classes, centers

    def _cluster_rows_multi(self, X_binary: np.ndarray, weights: np.ndarray) -> Tuple[
        List[List[int]], List[np.ndarray]]:
        """
        多归属行聚类：一行可以属于多个相似度超过阈值的类
        返回 (classes, centers)
        """
        classes = []
        centers = []

        for i, row in enumerate(X_binary):
            weighted_row = row * weights
            assigned_classes = []

            for j, center in enumerate(centers):
                both = (weighted_row * center).sum()
                norm_row = np.sqrt((weighted_row ** 2).sum())
                norm_center = np.sqrt((center ** 2).sum())
                if norm_row > 0 and norm_center > 0:
                    sim = both / (norm_row * norm_center)
                    if sim > self.row_similarity_threshold:
                        assigned_classes.append(j)

            if assigned_classes:
                for j in assigned_classes:
                    classes[j].append(i)
                    old_size = len(classes[j]) - 1
                    centers[j] = (centers[j] * old_size + weighted_row) / (old_size + 1)
            else:
                classes.append([i])
                centers.append(weighted_row.copy())

        return classes, centers

    # ==================== 字段相关聚类与规则发现主函数 ====================
    def _cluster_columns_and_rules(self, df: pd.DataFrame, numeric_cols: List[str],
                                   min_nonnull_in_cluster: int = 10,
                                   min_nonnull_ratio_in_cluster: float = 0.05,
                                   strong_equality: List[Dict] = None) -> List[Dict]:
        """
        执行相关聚类，在聚类子集上发现关系
        """
        cluster_rows = len(df)

        weights = df.attrs.get('column_weights', None)
        all_numeric_cols = df.attrs.get('numeric_cols', numeric_cols)

        # 相关聚类
        corr_clusters = self._cluster_columns_by_correlation(df, all_numeric_cols)
        self._log(f"    行类: {len(df)}行,字段聚类结果（共{len(corr_clusters)}个）:")

        # ========== 第一阶段：收集所有第一次发现的规则 ==========
        first_rules_all = []
        cluster_info = []

        for idx, cluster in enumerate(corr_clusters):
            if len(cluster) < 3:
                continue
            self._log(f"        字段聚类: {len(cluster)}个变量 - {cluster}")

            cooccur = self._check_cooccurrence(df, cluster)

            first_rules = self._discover_rules_on_fields(df, cluster, cooccur, weights, all_numeric_cols)

            if first_rules:
                self._log(f"          第一次发现规则:{first_rules}")
                first_rules_all.extend(first_rules)

            cluster_info.append({
                'cluster': cluster,
                'cooccur': cooccur,
                'first_rules': first_rules
            })

        # ========== 第二阶段：基于全局规则，剔除字段后再次发现 ==========
        seen_global = set()
        global_rules = []
        for rule in first_rules_all:
            key = frozenset(rule['fields'])
            if key not in seen_global:
                seen_global.add(key)
                global_rules.append(rule)

        # 结果先包含第一次发现的规则
        all_rules = first_rules_all[:]

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
                continue

            remaining_fields = [f for f in cluster if f not in remove_fields]
            self._log(f"        字段聚类 {cluster} 剔除 {remove_fields} 后剩余: {remaining_fields}")

            if len(remaining_fields) < 3:
                continue

            second_rules = self._discover_rules_on_fields(df, remaining_fields, cooccur, weights, all_numeric_cols)

            if second_rules:
                self._log(f"          第二次发现规则:{second_rules}")
                existing_keys = {frozenset(r.get('fields', [])) for r in all_rules}
                for r in second_rules:
                    key = frozenset(r.get('fields', []))
                    if key not in existing_keys:
                        existing_keys.add(key)
                        all_rules.append(r)

        return all_rules

    # ==================== 两次发现流程 ====================
    def _discover_rules_on_fields(self, df: pd.DataFrame, fields: List[str],
                                  cooccur: int, weights: np.ndarray = None,
                                  all_numeric_cols: List[str] = None) -> List[Dict]:
        """
        在指定的字段列表上发现规则（根据共同非空行数和字段数决定是否拆分）
        """
        if len(fields) < 3:
            return []

            # 如果没有权重信息，则直接做一次普通 SVD
        if weights is None or all_numeric_cols is None:
            return self._discover_rules_single_svd(df, fields)

        if cooccur >= 100 or len(fields) <= 10:
            return self._discover_rules_in_subcluster(df, fields, weights=weights,
                                                      all_numeric_cols=all_numeric_cols)
        else:
            subclusters = self._split_by_cooccurrence_ratio(df, fields)
            rules = []
            for sub in subclusters:
                if len(sub) < 3:
                    continue
                sub_rules = self._discover_rules_in_subcluster(df, sub, weights=weights,
                                                               all_numeric_cols=all_numeric_cols)
                rules.extend(sub_rules)
            return rules

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

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                corr_val = corr_matrix.loc[col1, col2]
                if not pd.isna(corr_val) and corr_val > self.corr_threshold:
                    G.add_edge(col1, col2)

        nodes = list(G.nodes())
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                if nx.has_path(G, u, v) and nx.shortest_path_length(G, u, v) == 2:
                    G.add_edge(u, v)

        components = list(nx.connected_components(G))
        clusters = [list(comp) for comp in components]
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

                if co >= self.min_cooccurrence_rows:
                    subcluster.append(other)

            if len(subcluster) >= 3:
                is_dup = False
                for existing in subclusters:
                    if set(subcluster) == set(existing):
                        is_dup = True
                        break
                if not is_dup:
                    subclusters.append(subcluster)

        return subclusters

    # ==================== 关系发现 ====================
    def _discover_rules_in_subcluster(self, df: pd.DataFrame, fields: List[str],
                                      weights: np.ndarray = None,
                                      all_numeric_cols: List[str] = None) -> List[Dict]:
        """
        在子类中发现关系，支持逐步剔除低权重字段的循环 SVD
        """
        if len(fields) < 3:
            return []

        if weights is None or all_numeric_cols is None:
            return self._discover_rules_single_svd(df, fields)

        # 使用双向剔除（正向+反向）
        #return self._bidirectional_elimination(df, fields, weights, all_numeric_cols)

        return self._eliminate_by_weight(df, fields, weights, all_numeric_cols, 'low_first')

    # ==================== 双向剔除（正向/反向） ====================
    def _eliminate_by_weight(self, df: pd.DataFrame, fields: List[str],
                             weights: np.ndarray, all_numeric_cols: List[str],
                             direction: str = 'low_first') -> List[Dict]:
        """
        按权重方向剔除字段，循环SVD发现规则

        参数:
        - direction: 'low_first' 先剔除权重低的（保留稀疏字段），'high_first' 先剔除权重高的（保留稠密字段）
        """
        if len(fields) < 3:
            return []

        current_fields = fields.copy()
        field_weight_dict = {f: weights[all_numeric_cols.index(f)] for f in current_fields if f in all_numeric_cols}
        if len(field_weight_dict) != len(current_fields):
            return self._discover_rules_single_svd(df, fields)

        all_rules = []
        seen_rules = set()

        while len(current_fields) >= 3:
            rules = self._discover_rules_single_svd(df, current_fields)

            for rule in rules:
                rule_key = frozenset(rule.get('fields', []))
                if rule_key not in seen_rules:
                    seen_rules.add(rule_key)
                    all_rules.append(rule)

            if len(current_fields) <= 3:
                break

            if direction == 'low_first':
                target_field = min(current_fields, key=lambda f: field_weight_dict[f])
            else:
                target_field = max(current_fields, key=lambda f: field_weight_dict[f])

            current_fields.remove(target_field)

        return all_rules

    def _bidirectional_elimination(self, df: pd.DataFrame, fields: List[str],
                                   weights: np.ndarray, all_numeric_cols: List[str]) -> List[Dict]:
        """
        双向剔除：同时运行正向剔除和反向剔除，合并结果
        """
        low_first_rules = self._eliminate_by_weight(df, fields, weights, all_numeric_cols, 'low_first')
        high_first_rules = self._eliminate_by_weight(df, fields, weights, all_numeric_cols, 'high_first')
        all_rules = low_first_rules + high_first_rules
        return self._deduplicate_rules(all_rules)

    def _discover_rules_single_svd(self, df: pd.DataFrame, fields: List[str]) -> List[Dict]:
        """
        对指定的字段集合做一次普通 SVD（不加权），返回发现的规则列表
        """
        if len(fields) < 3:
            return []

        valid_df = self._get_valid_data(df, fields)
        valid_rows = len(valid_df)
        if valid_rows < self.min_cooccurrence_rows:
            return []

        X_orig = valid_df[fields].values

        # ========== 小样本复制增强 ==========
        #X_orig, valid_rows = self._augment_small_sample(X_orig, valid_rows, True)

        # ========== 尝试迭代剔除增强版 ==========
        #best_coeffs, best_inlier_mask, best_keep_rows_count = self._iterative_svd_refinement(X_orig, valid_rows, True)

        # 场景1：数据较干净，追求速度 - 不使用采样
        #best_coeffs, best_inlier_mask, count = self._fit_linear_relation(X_orig, valid_rows, use_ransac=False)

        # # 场景2：数据有异常值，追求鲁棒性 - 使用RANSAC采样 通不过
        # best_coeffs, best_inlier_mask, count = self._fit_linear_relation(X_orig, valid_rows, use_ransac=True)
        #
        # # 场景3：小样本数据（小于采样大小）- 自动降级为不采样
        if valid_rows < self.ransac_sample_size:
            best_coeffs, best_inlier_mask, count = self._fit_linear_relation(X_orig, valid_rows, use_ransac=False)
        else:
            best_coeffs, best_inlier_mask, count = self._fit_linear_relation(X_orig, valid_rows, use_ransac=True)


        if best_coeffs is None:
            return []

        result = X_orig @ best_coeffs
        rel_error = np.abs(result) / (np.abs(X_orig[:, 0]) + 1)
        final_inlier_mask = rel_error < self.inlier_error_threshold
        final_inlier_count = final_inlier_mask.sum()

        if final_inlier_count < valid_rows * self.min_final_inlier_ratio:
            return []

        return self._extract_rules_from_svd(X_orig, fields, final_inlier_mask)

    def _augment_small_sample(self, X_orig: np.ndarray, valid_rows: int, bb: bool):
        """
        小样本复制增强（备用）：当有效行数小于阈值时，通过复制和随机缩放增加样本量
        """
        if valid_rows < self.small_sample_threshold:
            repeat = (self.small_sample_threshold + valid_rows - 1) // valid_rows
            X_repeated = np.repeat(X_orig, repeat, axis=0)
            row_scales = np.random.uniform(1, 10, size=(X_repeated.shape[0], 1))
            X_aug = X_repeated * row_scales
            augmented_rows = X_aug.shape[0]
            if bb:
                self._log(
                    f"        有效行数 {valid_rows} < {self.small_sample_threshold}，复制 {repeat} 倍并添加随机缩放(1~10)，用于 SVD 拟合（共 {augmented_rows} 行）")
            return X_aug, augmented_rows
        else:
            return X_orig, valid_rows

    def _iterative_svd_refinement(self, X: np.ndarray, valid_rows: int, bb: bool):
        """
        迭代剔除增强版（备用）：逐次剔除残差最大的行，找到最优系数
        """
        min_keep_rows = max(5, int(valid_rows * (1 - self.max_remove_ratio)))

        best_coeffs = None
        best_inlier_mask = None
        best_keep_rows_count = 0

        current_X = X.copy()
        current_indices = np.arange(valid_rows)

        while len(current_indices) >= min_keep_rows:
            X_sample = current_X
            rows_count = len(current_indices)

            try:
                X_centered = X_sample - np.mean(X_sample, axis=0)
                U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
                coeffs = Vt[-1, :]

                if np.all(np.abs(coeffs) < self.precision):
                    break

                result = X_sample @ coeffs
                rel_error = np.abs(result) / (np.abs(X_sample[:, 0]) + 1)

                remove_count = max(1, rows_count // 10)
                largest_error_indices = np.argsort(rel_error)[-remove_count:]

                if rel_error[largest_error_indices[-1]] < self.inlier_error_threshold:
                    best_coeffs = coeffs
                    best_keep_rows_count = rows_count
                    best_inlier_mask = np.zeros(valid_rows, dtype=bool)
                    best_inlier_mask[current_indices] = True
                    break

                current_X = np.delete(current_X, largest_error_indices, axis=0)
                current_indices = np.delete(current_indices, largest_error_indices)

            except Exception as e:
                if bb:
                    self._log(f"迭代剔除异常: {e}")
                break
        else:
            if len(current_indices) >= 3:
                X_sample = current_X
                X_centered = X_sample - np.mean(X_sample, axis=0)
                U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
                best_coeffs = Vt[-1, :]
                best_keep_rows_count = len(current_indices)
                best_inlier_mask = np.zeros(valid_rows, dtype=bool)
                best_inlier_mask[current_indices] = True

        return best_coeffs, best_inlier_mask, best_keep_rows_count

    def _basic_svd_fit(self, X: np.ndarray, valid_rows: int):
        """使用随机化 SVD 加速"""
        if not HAS_SKLEARN:
            # 降级到标准 SVD
            try:
                X_centered = X - np.mean(X, axis=0)
                U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
                coeffs = Vt[-1, :]
                if np.all(np.abs(coeffs) < self.precision):
                    return None, None, 0
                result = X @ coeffs
                rel_error = np.abs(result) / (np.abs(X[:, 0]) + 1)
                inlier_mask = rel_error < self.inlier_error_threshold
                inlier_count = inlier_mask.sum()
                return coeffs, inlier_mask, inlier_count
            except Exception:
                return None, None, 0
        else:
            try:
                # 使用随机化 SVD，只取最小奇异值对应的右奇异向量
                # n_components 取列数，但随机化可以只计算少量成分
                n_components = min(X.shape[1], 10)  # 最多取10个成分，但我们需要最后一个
                U, s, Vt = randomized_svd(X, n_components=min(n_components, X.shape[1]-1), random_state=42)
                # 实际上我们需要最后一个右奇异向量，但 randomized_svd 不保证顺序，我们取最小的奇异值对应的向量
                # 更可靠：对中心化数据做 SVD，但使用随机化近似
                X_centered = X - np.mean(X, axis=0)
                U, s, Vt = randomized_svd(X_centered, n_components=min(n_components, X.shape[1]-1), random_state=42)
                # 取最小奇异值对应的向量（最后一个）
                coeffs = Vt[-1, :]
                if np.all(np.abs(coeffs) < self.precision):
                    return None, None, 0
                result = X @ coeffs
                rel_error = np.abs(result) / (np.abs(X[:, 0]) + 1)
                inlier_mask = rel_error < self.inlier_error_threshold
                inlier_count = inlier_mask.sum()
                return coeffs, inlier_mask, inlier_count
            except Exception:
                # 降级到标准 SVD
                try:
                    X_centered = X - np.mean(X, axis=0)
                    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
                    coeffs = Vt[-1, :]
                    if np.all(np.abs(coeffs) < self.precision):
                        return None, None, 0
                    result = X @ coeffs
                    rel_error = np.abs(result) / (np.abs(X[:, 0]) + 1)
                    inlier_mask = rel_error < self.inlier_error_threshold
                    inlier_count = inlier_mask.sum()
                    return coeffs, inlier_mask, inlier_count
                except Exception:
                    return None, None, 0

    def _ransac_svd_fit(self, X: np.ndarray, valid_rows: int):
        """RANSAC 迭代中使用随机 SVD"""
        best_coeffs = None
        best_inlier_mask = None
        best_inlier_count = 0

        for _ in range(self.ransac_iter):
            sample_idx = np.random.choice(valid_rows, size=self.ransac_sample_size, replace=False)
            X_sample = X[sample_idx]
            try:
                # 使用基本 SVD（内部可能使用随机化）
                coeffs, _, _ = self._basic_svd_fit(X_sample, len(sample_idx))
                if coeffs is None:
                    continue
                if np.all(np.abs(coeffs) < self.precision):
                    continue
                result = X @ coeffs
                rel_error = np.abs(result) / (np.abs(X[:, 0]) + 1)
                inlier_mask = rel_error < self.inlier_error_threshold
                inlier_count = inlier_mask.sum()
                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    best_coeffs = coeffs
                    best_inlier_mask = inlier_mask
                    if best_inlier_count >= valid_rows * self.inlier_ratio:
                        break
            except Exception:
                continue

        if best_coeffs is None:
            return None, None, 0
        return best_coeffs, best_inlier_mask, best_inlier_count

    def _fit_linear_relation(self, X: np.ndarray, valid_rows: int, use_ransac: bool = True):
        """
        线性关系拟合（统一入口）

        Parameters
        ----------
        X : np.ndarray
            输入数据矩阵，每列为一个字段
        valid_rows : int
            有效行数
        use_ransac : bool, default=True
            是否使用 RANSAC 采样拟合
            - True: 使用 RANSAC 迭代采样（适合有异常值的数据）
            - False: 使用全量 SVD 拟合（适合干净数据，速度更快）

        Returns
        -------
        coeffs : np.ndarray
            拟合系数
        inlier_mask : np.ndarray
            内点掩码
        inlier_count : int
            内点数量
        """
        if use_ransac:
            return self._ransac_svd_fit(X, valid_rows)
        else:
            return self._basic_svd_fit(X, valid_rows)

    def _extract_rules_from_svd(self, X: np.ndarray, fields: List[str],
                                inlier_mask: np.ndarray) -> List[Dict]:
        """
        共用后续处理：基于内点重新 SVD，分组，生成规则
        """
        inlier_X = X[inlier_mask]
        if len(inlier_X) < 3:
            return []

        total_valid_rows = X.shape[0]
        satisfied_rows = inlier_mask.sum()

        inlier_X_centered = inlier_X - np.mean(inlier_X, axis=0)
        U, s, Vt = np.linalg.svd(inlier_X_centered, full_matrices=False)
        coeffs = Vt[-1, :]

        groups = self._group_coefficients(coeffs, fields)

        if not groups:
            return []

        rules = []
        for group in groups:
            subset_fields = group['fields']
            subset_coeffs = group['coeffs']
            subset_X = X[:, [fields.index(f) for f in subset_fields]]

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

            satisfied_rows_calc = int(round(confidence * total_valid_rows))

            rules.append({
                'rule': expr,
                'confidence': round(confidence, 4),
                'priority': '高' if confidence == 1.0 else '中',
                'fields': subset_fields,
                'relation_type': 'additive',
                'violation_count': total_valid_rows - satisfied_rows_calc,
                'violation_samples': [],
                'valid_rows': total_valid_rows,
                'satisfied_rows': satisfied_rows_calc
            })

        return rules

    def _group_coefficients(self, coeffs: np.ndarray, fields: List[str]) -> List[Dict]:
        """
        对系数进行符号化和分组
        """
        coeffs_sign = np.zeros(len(coeffs), dtype=int)
        for i, c in enumerate(coeffs):
            if c > self.precision:
                coeffs_sign[i] = 1
            elif c < -self.precision:
                coeffs_sign[i] = -1

        nonzero_indices = [i for i, c in enumerate(coeffs_sign) if c != 0]
        if len(nonzero_indices) < 3:
            return []

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
                    if target_abs > 0 and abs(current_abs - target_abs) / target_abs < self.coeff_group_tolerance:
                        group.append(j)
            if len(group) >= 3:
                groups.append({
                    'fields': [fields[idx] for idx in group],
                    'coeffs': [coeffs_sign[idx] for idx in group]
                })
                used.update(group)

        return groups



    def _validate_rule_on_full_data(self, df: pd.DataFrame, rule: Dict) -> Dict:
        """
        在全量数据上重新验证单条规则，更新 valid_rows、satisfied_rows、confidence、violation_count
        """
        fields = rule.get('fields', [])
        if len(fields) < 2:
            return rule

        # 获取全量数据中所有字段都非空的行
        valid_mask = df[fields].notna().all(axis=1)
        valid_rows = valid_mask.sum()

        if valid_rows == 0:
            rule['valid_rows'] = 0
            rule['satisfied_rows'] = 0
            rule['confidence'] = 0.0
            rule['violation_count'] = 0
            rule['priority'] = '低'
            return rule

        # 解析规则，计算满足条件的行数
        rule_str = rule['rule']
        left, right = rule_str.split(' = ')
        left_fields = [f.strip() for f in left.split(' + ')] if left != '0' else []
        right_fields = [f.strip() for f in right.split(' + ')] if right != '0' else []

        # 计算左值和右值
        if left_fields:
            left_sum = df.loc[valid_mask, left_fields].sum(axis=1)
        else:
            left_sum = 0

        if right_fields:
            right_sum = df.loc[valid_mask, right_fields].sum(axis=1)
        else:
            right_sum = 0

        # 计算误差
        diff = np.abs(left_sum - right_sum)
        scale = np.maximum(np.abs(left_sum), np.abs(right_sum))
        scale = np.maximum(scale, 1)
        satisfied_mask = (diff / scale < 1e-4)
        satisfied_rows = satisfied_mask.sum()

        confidence = satisfied_rows / valid_rows if valid_rows > 0 else 0

        # 更新规则
        rule['valid_rows'] = int(valid_rows)
        rule['satisfied_rows'] = int(satisfied_rows)
        rule['confidence'] = round(confidence, 4)
        rule['violation_count'] = int(valid_rows - satisfied_rows)
        rule['priority'] = '高' if confidence == 1.0 else '中'

        return rule

    def _validate_rules_on_full_data(self, df: pd.DataFrame, rules: List[Dict]) -> List[Dict]:
        """
        在全量数据上重新验证所有规则
        """
        validated_rules = []
        for rule in rules:
            validated_rule = self._validate_rule_on_full_data(df, rule)
            # 重新计算后的置信度必须大于等于最小置信度阈值
            if validated_rule.get('confidence', 0) >= self.min_confidence:
                validated_rules.append(validated_rule)
        return validated_rules

    # ==================== 规则简化（保留备用） ====================
    def reduce_arithmetic_rules(self, rules: List[Dict]) -> List[Dict]:
        """
        对加法规则列表进行分层简化（保留备用，当前未使用）

        注意：2字段规则不参与消元，只作为结果输出
        消元只发生在3+字段规则之间
        """
        from copy import deepcopy

        def parse_rule(rule_dict):
            expr = rule_dict['rule']
            left, right = expr.split(' = ')
            left_fields = [f.strip() for f in left.split(' + ')] if left != '0' else []
            right_fields = [f.strip() for f in right.split(' + ')] if right != '0' else []
            return left_fields, right_fields, rule_dict.get('confidence', 1.0)

        def build_rule(left_fields, right_fields, confidence):
            left_fields.sort()
            right_fields.sort()
            left_expr = ' + '.join(left_fields) if left_fields else '0'
            right_expr = ' + '.join(right_fields) if right_fields else '0'
            rule_str = f"{left_expr} = {right_expr}"
            all_fields = left_fields + right_fields
            return {
                'rule': rule_str,
                'fields': all_fields,
                'confidence': confidence,
                'priority': '高' if confidence == 1.0 else '中',
                'relation_type': 'additive',
                'violation_count': 0,
                'violation_samples': []
            }

        # 分离2字段规则
        two_field_rules = []
        other_rules = []

        for r in rules:
            if len(r.get('fields', [])) == 2:
                two_field_rules.append(r)
            else:
                other_rules.append(r)

        if not other_rules:
            return two_field_rules

        # 按字段数分组
        rules_by_len = {}
        for r in other_rules:
            field_count = len(r.get('fields', []))
            if field_count not in rules_by_len:
                rules_by_len[field_count] = []
            rules_by_len[field_count].append(r)

        # 3字段规则去重
        if 3 in rules_by_len:
            seen = set()
            unique_3 = []
            for r in rules_by_len[3]:
                left, right, conf = parse_rule(r)
                key = frozenset(left + right)
                if key not in seen:
                    seen.add(key)
                    unique_3.append({
                        'left': left,
                        'right': right,
                        'confidence': conf
                    })
            rules_by_len[3] = unique_3

        # 构建3字段规则索引
        three_field_index = {}
        if 3 in rules_by_len:
            for r in rules_by_len[3]:
                key = frozenset(r['left'] + r['right'])
                three_field_index[key] = r

        # 逐层消元
        for field_count in sorted(rules_by_len.keys()):
            if field_count <= 3:
                continue

            simplified_rules = []
            for r in rules_by_len[field_count]:
                left, right, conf = parse_rule(r)
                changed = False
                new_left, new_right = left[:], right[:]

                for three_key, three_rule in three_field_index.items():
                    three_left = three_rule['left']
                    three_right = three_rule['right']
                    three_fields = three_left + three_right

                    current_fields = set(left + right)
                    if not three_key.issubset(current_fields):
                        continue

                    if set(three_left).issubset(set(left)) and set(three_right).issubset(set(left)):
                        new_left = [f for f in left if f not in three_fields]
                        new_left.append(three_right[0])
                        new_left = list(set(new_left))
                        new_right = right[:]
                        conf = min(conf, three_rule['confidence'])
                        changed = True
                        break
                    elif set(three_left).issubset(set(right)) and set(three_right).issubset(set(left)):
                        new_left = [f for f in left if f not in three_right]
                        new_right = [f for f in right if f not in three_left]
                        new_left.append(three_left[0])
                        new_left = list(set(new_left))
                        new_right = list(set(new_right))
                        conf = min(conf, three_rule['confidence'])
                        changed = True
                        break
                    elif set(three_left).issubset(set(left)) and set(three_right).issubset(set(right)):
                        new_left = [f for f in left if f not in three_left]
                        new_right = [f for f in right if f not in three_right]
                        conf = min(conf, three_rule['confidence'])
                        changed = True
                        break
                    elif set(three_left).issubset(set(right)) and set(three_right).issubset(set(right)):
                        new_left = left[:]
                        new_right = [f for f in right if f not in three_fields]
                        new_right.append(three_left[0])
                        new_right = list(set(new_right))
                        conf = min(conf, three_rule['confidence'])
                        changed = True
                        break

                if changed:
                    new_field_count = len(new_left + new_right)
                    if new_field_count == 3:
                        new_left, new_right = new_left[:], new_right[:]
                        new_left.sort()
                        new_right.sort()
                        new_key = frozenset(new_left + new_right)
                        if new_key not in three_field_index:
                            three_field_index[new_key] = {
                                'left': new_left,
                                'right': new_right,
                                'confidence': conf
                            }
                            if 3 not in rules_by_len:
                                rules_by_len[3] = []
                            rules_by_len[3].append({
                                'left': new_left,
                                'right': new_right,
                                'confidence': conf
                            })
                    else:
                        new_left.sort()
                        new_right.sort()
                        simplified_rules.append({
                            'left': new_left,
                            'right': new_right,
                            'confidence': conf
                        })
                else:
                    simplified_rules.append({
                        'left': left,
                        'right': right,
                        'confidence': conf
                    })

            if simplified_rules:
                rules_by_len[field_count] = simplified_rules

        # 构建最终结果
        result = two_field_rules[:]

        for field_count in sorted(rules_by_len.keys()):
            for r in rules_by_len[field_count]:
                result.append(build_rule(r['left'], r['right'], r['confidence']))

        return self._deduplicate_rules(result)


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

        return self._deduplicate_rules(rules)

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

        return self._deduplicate_rules(rules)


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