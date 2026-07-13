# autostat/core/audit.py
"""勾稽关系发现模块 - 四条路径：相关聚类 + 共同非空聚类 + 非空数相近聚类 + 行相似度聚类"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
import math

# ============================================================
# 随机化 SVD（优先使用 sklearn，降级到标准 SVD）
# ============================================================
try:
    from sklearn.utils.extmath import randomized_svd
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class AuditRuleDiscoverer:
    """勾稽规则发现器 - 四条路径（融合优化版）"""

    def __init__(self,
                 # ========== 1. 基础参数 ==========
                 debug: bool = False,

                 # ========== 2. 数值精度参数 ==========
                 precision: float = 1e-6,

                 # ========== 3. 置信度参数 ==========
                 min_confidence: float = 0.5,

                 # ========== 4. 字段筛选参数 ==========
                 corr_threshold: float = 0.9,
                 min_nonnull_count: int = 10,
                 min_nonnull_rate: float = 0.01,
                 cooccur_ratio: float = 0.9,
                 min_cooccurrence_rows: int = 10,
                 max_numeric_fields: int = 100,

                 # ========== 5. 聚类分析参数 ==========
                 nonnull_diff_ratio: float = 0.2,
                 row_similarity_threshold: float = 0.99,
                 min_cluster_size: int = 3,

                 # ========== 6. 相等关系参数 ==========
                 equality_ratio_threshold: float = 0.8,
                 strong_coverage_threshold: float = 0.8,

                 # ========== 7. RANSAC拟合参数 ==========
                 ransac_iter: int = 20,
                 ransac_sample_size: int = 20,
                 inlier_ratio: float = 0.7,

                 # ========== 8. 线性关系验证参数 ==========
                 min_final_inlier_ratio: float = 0.5,
                 coeff_group_tolerance: float = 0.5,
                 inlier_error_threshold: float = 0.01,

                 # ========== 9. 异常处理参数 ==========
                 max_remove_ratio: float = 0.3,
                 small_sample_threshold: int = 50,

                 # ========== 10. 优化控制参数 ==========
                 use_randomized_svd: bool = True,
                 use_svd_cache: bool = True,
                 use_vectorized_clustering: bool = True):
        """
        勾稽规则发现器初始化（融合优化版）
        """
        # 基础参数
        self.debug = debug
        self.precision = precision
        self.min_confidence = min_confidence
        self.corr_threshold = corr_threshold
        self.min_nonnull_count = min_nonnull_count
        self.min_nonnull_rate = min_nonnull_rate
        self.cooccur_ratio = cooccur_ratio
        self.min_cooccurrence_rows = min_cooccurrence_rows
        self.max_numeric_fields = max_numeric_fields
        self.nonnull_diff_ratio = nonnull_diff_ratio
        self.row_similarity_threshold = row_similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.equality_ratio_threshold = equality_ratio_threshold
        self.strong_coverage_threshold = strong_coverage_threshold
        self.ransac_iter = ransac_iter
        self.ransac_sample_size = ransac_sample_size
        self.inlier_ratio = inlier_ratio
        self.min_final_inlier_ratio = min_final_inlier_ratio
        self.coeff_group_tolerance = coeff_group_tolerance
        self.inlier_error_threshold = inlier_error_threshold
        self.max_remove_ratio = max_remove_ratio
        self.small_sample_threshold = small_sample_threshold

        # 优化参数
        self.use_randomized_svd = use_randomized_svd and HAS_SKLEARN
        self.use_svd_cache = use_svd_cache
        self.use_vectorized_clustering = use_vectorized_clustering

        # SVD 缓存（字段组合 -> 规则列表）
        self._svd_cache = {}
        self._keep_cols = []

        if self.debug:
            print(f"  [AUDIT] 随机化SVD: {'启用' if self.use_randomized_svd else '禁用'}")
            print(f"  [AUDIT] SVD缓存: {'启用' if self.use_svd_cache else '禁用'}")
            print(f"  [AUDIT] 向量化聚类: {'启用' if self.use_vectorized_clustering else '禁用'}")

    def _log(self, msg: str):
        if self.debug:
            print(f"  [DEBUG] {msg}")

    def _clear_cache(self):
        """清空SVD缓存"""
        self._svd_cache = {}

    # ==================== 公共工具函数 ====================

    def _parse_rule_string(self, rule_dict: Dict) -> Tuple[List[str], List[str], float]:
        expr = rule_dict['rule']
        left, right = expr.split(' = ')
        left_fields = [f.strip() for f in left.split(' + ')] if left != '0' else []
        right_fields = [f.strip() for f in right.split(' + ')] if right != '0' else []
        return left_fields, right_fields, rule_dict.get('confidence', 1.0)

    def _build_rule_dict(self, left_fields: List[str], right_fields: List[str],
                         confidence: float, priority: str = None) -> Dict:
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
        seen = set()
        result = []
        for rule in rules:
            key = frozenset(rule.get('fields', []))
            if key not in seen:
                seen.add(key)
                result.append(rule)
        return result

    def _normalize_two_field_rule(self, fields: List[str]) -> Tuple[str, List[str]]:
        f1, f2 = fields[0], fields[1]
        if f1 < f2:
            return f"{f1} = {f2}", [f1, f2]
        else:
            return f"{f2} = {f1}", [f2, f1]

    def _build_replace_map(self, strong_equality: List[Dict]) -> Dict[str, str]:
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
        valid_mask = df[fields].notna().all(axis=1)
        return df[valid_mask][fields].copy()

    def _check_cooccurrence(self, df: pd.DataFrame, fields: List[str]) -> int:
        if not fields:
            return 0
        valid_mask = df[fields].notna().all(axis=1)
        return valid_mask.sum()

    def _compute_cooccurrence_matrix(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        """使用矩阵乘法一次性计算所有列对的共现次数"""
        nonnull_bool = df[cols].notna().values.astype(int)
        cooccur = nonnull_bool.T @ nonnull_bool
        return cooccur

    # ==================== 主流程 ====================

    def discover_all(self, data: pd.DataFrame, variable_types: Dict[str, str],
                     foreign_keys: List[Dict] = None) -> Dict[str, Any]:
        """发现所有类型的勾稽规则"""
        self._clear_cache()
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

        # 时序关系发现（由 date_rules.py 负责，保留空列表）
        rules["temporal_rules"] = []

        self._log("=" * 60)
        return rules

    # ==================== 数值规则发现 ====================

    def _discover_arithmetic_rules(self, df: pd.DataFrame,
                                   numeric_cols: List[str]) -> List[Dict]:
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

    # ==================== 行相似度聚类（融合向量化优化） ====================

    def _cluster_rows_by_similarity(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[pd.DataFrame]:
        if len(df) < self.min_cluster_size:
            return []

        X_binary = df[numeric_cols].notna().astype(int).values
        total_rows, n_fields = X_binary.shape

        field_freq = X_binary.sum(axis=0)
        idf = np.log((total_rows + 1) / (field_freq + 1)) + 1
        weights = idf ** 3

        # 使用向量化聚类或原版聚类
        if self.use_vectorized_clustering:
            classes, centers = self._cluster_rows_single_vectorized(X_binary, weights)
        else:
            classes, centers = self._cluster_rows_single(X_binary, weights)

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

    def _cluster_rows_single(self, X_binary: np.ndarray, weights: np.ndarray) -> Tuple[
        List[List[int]], List[np.ndarray]]:
        """原版单归属行聚类（保留作为降级方案）"""
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

    def _cluster_rows_single_vectorized(self, X_binary: np.ndarray, weights: np.ndarray) -> Tuple[
        List[List[int]], List[np.ndarray]]:
        """向量化版单归属行聚类"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            X_weighted = X_binary * weights
            classes = []
            centers = []

            for i, row in enumerate(X_weighted):
                weighted_row = row.reshape(1, -1)

                if not centers:
                    classes.append([i])
                    centers.append(weighted_row.flatten().copy())
                    continue

                centers_matrix = np.array(centers)
                sims = cosine_similarity(weighted_row, centers_matrix).flatten()

                best_j = np.argmax(sims)
                best_sim = sims[best_j]

                if best_sim > self.row_similarity_threshold:
                    classes[best_j].append(i)
                    old_size = len(classes[best_j]) - 1
                    centers[best_j] = (centers[best_j] * old_size + weighted_row.flatten()) / (old_size + 1)
                else:
                    classes.append([i])
                    centers.append(weighted_row.flatten().copy())

            return classes, centers

        except ImportError:
            if self.debug:
                print("  ⚠️ sklearn 不可用，降级到原版聚类方法")
            return self._cluster_rows_single(X_binary, weights)

    # ==================== 字段相关聚类与规则发现 ====================

    def _cluster_columns_and_rules(self, df: pd.DataFrame, numeric_cols: List[str],
                                   min_nonnull_in_cluster: int = 10,
                                   min_nonnull_ratio_in_cluster: float = 0.05,
                                   strong_equality: List[Dict] = None) -> List[Dict]:
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

        all_rules = first_rules_all[:]

        for info in cluster_info:
            cluster = info['cluster']
            cooccur = info['cooccur']

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

    def _discover_rules_on_fields(self, df: pd.DataFrame, fields: List[str],
                                  cooccur: int, weights: np.ndarray = None,
                                  all_numeric_cols: List[str] = None) -> List[Dict]:
        if len(fields) < 3:
            return []

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

    # ==================== 列相关聚类 ====================

    def _cluster_columns_by_correlation(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[List[str]]:
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
        if len(fields) < 3:
            return []

        if weights is None or all_numeric_cols is None:
            return self._discover_rules_single_svd(df, fields)

        return self._eliminate_by_weight(df, fields, weights, all_numeric_cols, 'low_first')

    def _eliminate_by_weight(self, df: pd.DataFrame, fields: List[str],
                             weights: np.ndarray, all_numeric_cols: List[str],
                             direction: str = 'low_first') -> List[Dict]:
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

    # ==================== SVD 发现规则（融合随机化SVD + 缓存） ====================

    def _compute_svd_coeffs(self, X_centered: np.ndarray) -> Optional[np.ndarray]:
        """
        计算 SVD 的最小奇异值对应的右奇异向量
        融合：使用随机化SVD（如果可用且启用）
        """
        if self.use_randomized_svd:
            try:
                n_cols = X_centered.shape[1]
                n_comp = min(n_cols - 1, 10)
                if n_comp < 1:
                    n_comp = 1

                U, s, Vt = randomized_svd(X_centered, n_components=n_comp, random_state=42)
                return Vt[-1, :]
            except Exception as e:
                if self.debug:
                    self._log(f"  ⚠️ 随机化SVD失败，降级到标准SVD: {e}")
                try:
                    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
                    return Vt[-1, :]
                except Exception:
                    return None
        else:
            try:
                U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
                return Vt[-1, :]
            except Exception:
                return None

    def _discover_rules_single_svd(self, df: pd.DataFrame, fields: List[str]) -> List[Dict]:
        """
        对指定的字段集合做一次 SVD，返回发现的规则列表
        融合：随机化SVD + 缓存
        """
        if len(fields) < 3:
            return []

        # ========== 缓存检查 ==========
        if self.use_svd_cache:
            cache_key = frozenset(fields)
            if cache_key in self._svd_cache:
                if self.debug:
                    self._log(f"        [缓存命中] 字段组合: {fields}")
                return self._svd_cache[cache_key].copy()

        # ========== 执行 SVD ==========
        valid_df = self._get_valid_data(df, fields)
        valid_rows = len(valid_df)
        if valid_rows < self.min_cooccurrence_rows:
            return []

        X_orig = valid_df[fields].values

        # 小样本复制增强（已禁用）
        # X_orig, valid_rows = self._augment_small_sample(X_orig, valid_rows, True)

        if valid_rows < self.ransac_sample_size:
            best_coeffs, best_inlier_mask, count = self._fit_linear_relation(X_orig, valid_rows, use_ransac=False)
        else:
            best_coeffs, best_inlier_mask, count = self._fit_linear_relation(X_orig, valid_rows, use_ransac=True)

        if best_coeffs is None:
            if self.use_svd_cache:
                self._svd_cache[cache_key] = []
            return []

        result = X_orig @ best_coeffs
        rel_error = np.abs(result) / (np.abs(X_orig[:, 0]) + 1)
        final_inlier_mask = rel_error < self.inlier_error_threshold
        final_inlier_count = final_inlier_mask.sum()

        if final_inlier_count < valid_rows * self.min_final_inlier_ratio:
            if self.use_svd_cache:
                self._svd_cache[cache_key] = []
            return []

        rules = self._extract_rules_from_svd(X_orig, fields, final_inlier_mask)

        # ========== 缓存保存 ==========
        if self.use_svd_cache:
            self._svd_cache[cache_key] = rules.copy()

        return rules

    def _augment_small_sample(self, X_orig: np.ndarray, valid_rows: int, bb: bool):
        """小样本复制增强（已禁用，保留方法但返回原数据）"""
        return X_orig, valid_rows

    def _iterative_svd_refinement(self, X: np.ndarray, valid_rows: int, bb: bool):
        """迭代剔除增强版（保留原版）"""
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
                coeffs = self._compute_svd_coeffs(X_centered)

                if coeffs is None or np.all(np.abs(coeffs) < self.precision):
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
                best_coeffs = self._compute_svd_coeffs(X_centered)
                best_keep_rows_count = len(current_indices)
                best_inlier_mask = np.zeros(valid_rows, dtype=bool)
                best_inlier_mask[current_indices] = True

        return best_coeffs, best_inlier_mask, best_keep_rows_count

    # ==================== SVD 拟合函数（融合随机化SVD） ====================

    def _basic_svd_fit(self, X: np.ndarray, valid_rows: int):
        """基础 SVD 拟合（融合随机化SVD）"""
        try:
            X_centered = X - np.mean(X, axis=0)
            coeffs = self._compute_svd_coeffs(X_centered)

            if coeffs is None or np.all(np.abs(coeffs) < self.precision):
                return None, None, 0

            result = X @ coeffs
            rel_error = np.abs(result) / (np.abs(X[:, 0]) + 1)
            inlier_mask = rel_error < self.inlier_error_threshold
            inlier_count = inlier_mask.sum()

            return coeffs, inlier_mask, inlier_count
        except Exception:
            return None, None, 0

    def _ransac_svd_fit(self, X: np.ndarray, valid_rows: int):
        """RANSAC SVD 拟合（融合随机化SVD）"""
        best_coeffs = None
        best_inlier_mask = None
        best_inlier_count = 0

        for _ in range(self.ransac_iter):
            sample_idx = np.random.choice(valid_rows, size=self.ransac_sample_size, replace=False)
            X_sample = X[sample_idx]

            try:
                X_centered = X_sample - np.mean(X_sample, axis=0)
                coeffs = self._compute_svd_coeffs(X_centered)

                if coeffs is None or np.all(np.abs(coeffs) < self.precision):
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
        if use_ransac:
            return self._ransac_svd_fit(X, valid_rows)
        else:
            return self._basic_svd_fit(X, valid_rows)

    # ==================== 规则提取 ====================

    def _extract_rules_from_svd(self, X: np.ndarray, fields: List[str],
                                inlier_mask: np.ndarray) -> List[Dict]:
        inlier_X = X[inlier_mask]
        if len(inlier_X) < 3:
            return []

        total_valid_rows = X.shape[0]
        satisfied_rows = inlier_mask.sum()

        inlier_X_centered = inlier_X - np.mean(inlier_X, axis=0)
        coeffs = self._compute_svd_coeffs(inlier_X_centered)

        if coeffs is None:
            return []

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

    # ==================== 全量验证 ====================

    def _validate_rule_on_full_data(self, df: pd.DataFrame, rule: Dict) -> Dict:
        fields = rule.get('fields', [])
        if len(fields) < 2:
            return rule

        valid_mask = df[fields].notna().all(axis=1)
        valid_rows = valid_mask.sum()

        if valid_rows == 0:
            rule['valid_rows'] = 0
            rule['satisfied_rows'] = 0
            rule['confidence'] = 0.0
            rule['violation_count'] = 0
            rule['priority'] = '低'
            return rule

        rule_str = rule['rule']
        left, right = rule_str.split(' = ')
        left_fields = [f.strip() for f in left.split(' + ')] if left != '0' else []
        right_fields = [f.strip() for f in right.split(' + ')] if right != '0' else []

        if left_fields:
            left_sum = df.loc[valid_mask, left_fields].sum(axis=1)
        else:
            left_sum = 0

        if right_fields:
            right_sum = df.loc[valid_mask, right_fields].sum(axis=1)
        else:
            right_sum = 0

        diff = np.abs(left_sum - right_sum)
        scale = np.maximum(np.abs(left_sum), np.abs(right_sum))
        scale = np.maximum(scale, 1)
        satisfied_mask = (diff / scale < 1e-4)
        satisfied_rows = satisfied_mask.sum()

        confidence = satisfied_rows / valid_rows if valid_rows > 0 else 0

        rule['valid_rows'] = int(valid_rows)
        rule['satisfied_rows'] = int(satisfied_rows)
        rule['confidence'] = round(confidence, 4)
        rule['violation_count'] = int(valid_rows - satisfied_rows)
        rule['priority'] = '高' if confidence == 1.0 else '中'

        return rule

    def _validate_rules_on_full_data(self, df: pd.DataFrame, rules: List[Dict]) -> List[Dict]:
        validated_rules = []
        for rule in rules:
            validated_rule = self._validate_rule_on_full_data(df, rule)
            if validated_rule.get('confidence', 0) >= self.min_confidence:
                validated_rules.append(validated_rule)
        return validated_rules

    # ==================== 规则简化（保留备用） ====================

    def reduce_arithmetic_rules(self, rules: List[Dict]) -> List[Dict]:
        """规则简化（与原版完全相同）"""
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

    # ==================== 时序关系（保留但未使用） ====================

    def _discover_temporal_rules(self, df: pd.DataFrame,
                                 date_cols: List[str]) -> List[Dict]:
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


# ========================================================================
# 便捷函数
# ========================================================================

def discover_audit_rules(data: pd.DataFrame, variable_types: Dict[str, str],
                         foreign_keys: List[Dict] = None,
                         debug: bool = False,
                         use_randomized_svd: bool = True,
                         use_svd_cache: bool = True,
                         use_vectorized_clustering: bool = True,
                         **kwargs) -> Dict[str, Any]:
    """
    发现勾稽规则（融合优化版）

    参数：
        data: 数据框
        variable_types: 变量类型字典
        foreign_keys: 外键列表
        debug: 调试模式
        use_randomized_svd: 是否使用随机化SVD（默认True）
        use_svd_cache: 是否缓存SVD结果（默认True）
        use_vectorized_clustering: 是否使用向量化聚类（默认True）
        **kwargs: 其他 AuditRuleDiscoverer 参数

    返回：
        勾稽规则字典
    """
    discoverer = AuditRuleDiscoverer(
        debug=debug,
        use_randomized_svd=use_randomized_svd,
        use_svd_cache=use_svd_cache,
        use_vectorized_clustering=use_vectorized_clustering,
        **kwargs
    )
    return discoverer.discover_all(data, variable_types, foreign_keys or [])


def verify_audit_rules(data: pd.DataFrame,
                       rules: List[Tuple[List[str], List[int]]],
                       debug: bool = False) -> List[Dict]:
    """验证指定的勾稽规则"""
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