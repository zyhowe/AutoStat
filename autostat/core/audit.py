# autostat/core/audit.py
"""勾稽关系发现模块 - 按相关图聚类，按共同非空拆分子类，RANSAC+SVD发现关系，支持等式验证"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
from math import gcd
from functools import reduce


class AuditRuleDiscoverer:
    """勾稽规则发现器"""

    def __init__(self,
                 precision: float = 1e-6,
                 min_confidence: float = 0.5,
                 corr_threshold: float = 0.9,
                 min_nonnull_count: int = 10,
                 min_nonnull_rate: float = 0.01,
                 cooccur_ratio: float = 0.9,
                 min_cooccurrence_rows: int = 10,
                 max_numeric_fields: int = 100,
                 ransac_iter: int = 50,
                 ransac_sample_size: int = 20,
                 inlier_ratio: float = 0.7,
                 debug: bool = False):
        """
        参数:
        - precision: 浮点数比较精度
        - min_confidence: 最小置信度阈值
        - corr_threshold: 相关系数阈值
        - min_nonnull_count: 最小非空数（低于此值排除）
        - min_nonnull_rate: 最小非空率（低于此值排除）
        - cooccur_ratio: 共同非空比例阈值（>= 此值 * min(非空数) 才合并）
        - min_cooccurrence_rows: 最小共同非空行数
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
        self._log("开始勾稽关系发现")
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

        # 步骤1：过滤高缺失字段
        self._keep_cols = [
            col for col in data.columns
            if nonnull_counts[col] >= self.min_nonnull_count
               or nonnull_rates[col] >= self.min_nonnull_rate
        ]
        excluded = [col for col in data.columns if col not in self._keep_cols]
        if excluded:
            self._log(
                f"排除字段（非空数<{self.min_nonnull_count} 且 非空率<{self.min_nonnull_rate:.0%}）: {excluded[:5]}{'...' if len(excluded) > 5 else ''}")
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

    # ==================== 步骤2：按相关图聚类 ====================
    def _cluster_by_correlation(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[List[str]]:
        """
        按相关系数构建图，取连通分量作为聚类
        """
        if len(numeric_cols) < 2:
            return [[col] for col in numeric_cols]

        G = nx.Graph()
        G.add_nodes_from(numeric_cols)

        corr_matrix = df[numeric_cols].corr().abs()
        edge_count = 0

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                corr_val = corr_matrix.loc[col1, col2]
                if not pd.isna(corr_val) and corr_val > self.corr_threshold:
                    G.add_edge(col1, col2)
                    edge_count += 1

        self._log(f"    相关图: {len(numeric_cols)} 节点, {edge_count} 边, 阈值={self.corr_threshold}")

        components = list(nx.connected_components(G))
        clusters = [list(comp) for comp in components]

        self._log(f"    连通分量数: {len(clusters)}")
        return clusters

    # ==================== 步骤3：按共同非空拆分子类（非互斥） ====================
    def _split_by_cooccurrence_non_exclusive(self, df: pd.DataFrame, cluster: List[str]) -> List[List[str]]:
        """
        按共同非空条件拆分子类（非互斥）

        每个字段可以属于多个子类，只要满足条件就创建子类

        条件：子类内所有字段同时非空的行数 >= cooccur_ratio * min(各字段非空数)
        """
        if len(cluster) <= 3:
            return [cluster]

        # 计算非空数
        nonnull_counts = {col: df[col].notna().sum() for col in cluster}

        # 按非空数从大到小排序（大的作为种子，更容易找到关系）
        sorted_cols = sorted(cluster, key=lambda x: nonnull_counts[x], reverse=True)

        subclusters = []

        for seed in sorted_cols:
            subcluster = [seed]
            seed_count = nonnull_counts[seed]

            for other in sorted_cols:
                if other == seed:
                    continue

                # 测试加入 other 后的共同非空数
                test_fields = subcluster + [other]
                cooccur = self._check_cooccurrence(df, test_fields)

                # 计算最小非空数
                min_count = min(seed_count, min(nonnull_counts[o] for o in subcluster), nonnull_counts[other])

                # 判断条件
                if cooccur >= self.cooccur_ratio * min_count and cooccur >= self.min_cooccurrence_rows:
                    subcluster.append(other)

            if len(subcluster) >= 3:
                # 检查是否与已有子类重复
                is_dup = False
                for existing in subclusters:
                    if set(subcluster) == set(existing):
                        is_dup = True
                        break
                if not is_dup:
                    subclusters.append(subcluster)
                    self._log(
                        f"      子类: {len(subcluster)}个变量, 共同非空={self._check_cooccurrence(df, subcluster)}")

        return subclusters

    # ==================== 步骤4：发现关系 ====================
    def _discover_rules_in_subcluster(self, df: pd.DataFrame, fields: List[str]) -> List[Dict]:
        """
        在子类中发现关系
        """
        if len(fields) < 3:
            return []

        # 取共同非空的数据
        valid_df = self._get_valid_data(df, fields)
        valid_rows = len(valid_df)
        self._log(f"        有效数据行数: {valid_rows}")

        if valid_rows < self.min_cooccurrence_rows:
            self._log(f"        有效数据行数 < {self.min_cooccurrence_rows}，跳过")
            return []

        X = valid_df[fields].values
        n_cols = len(fields)

        # 确保采样数大于列数
        actual_sample_size = max(self.ransac_sample_size, n_cols + 1)
        if actual_sample_size > valid_rows:
            actual_sample_size = valid_rows

        best_coeffs = None
        best_inlier_mask = None
        best_inlier_count = 0
        best_sv_ratio = 0

        for _ in range(self.ransac_iter):
            if valid_rows <= actual_sample_size:
                sample_idx = np.arange(valid_rows)
            else:
                sample_idx = np.random.choice(valid_rows, actual_sample_size, replace=False)

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

        if best_coeffs is None:
            self._log(f"        RANSAC 未找到有效解")
            return []

        if best_inlier_count < valid_rows * 0.5:
            self._log(f"        内点率过低 ({best_inlier_count}/{valid_rows} = {best_inlier_count / valid_rows:.1%})")
            return []

        self._log(
            f"        RANSAC: 内点 {best_inlier_count}/{valid_rows} ({best_inlier_count / valid_rows:.1%}), 奇异值比={best_sv_ratio:.2e}")

        # 用内点重新 SVD
        inlier_X = X[best_inlier_mask]
        inlier_X_centered = inlier_X - np.mean(inlier_X, axis=0)
        U, s, Vt = np.linalg.svd(inlier_X_centered, full_matrices=False)
        coeffs = Vt[-1, :]

        # 符号系数
        coeffs_sign = np.zeros(len(coeffs), dtype=int)
        for i, c in enumerate(coeffs):
            if c > self.precision:
                coeffs_sign[i] = 1
            elif c < -self.precision:
                coeffs_sign[i] = -1

        self._log(f"        符号系数: {coeffs_sign}")

        # 找出非零系数
        nonzero_indices = [i for i, c in enumerate(coeffs_sign) if c != 0]

        if len(nonzero_indices) < 3:
            self._log(f"        非零系数太少 ({len(nonzero_indices)})")
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
            self._log(f"        无有效分组")
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

            self._log(f"        子集 {subset_fields}: 置信度={confidence:.4f}")

            if confidence < self.min_confidence:
                self._log(f"        跳过: 置信度不足")
                continue

            # 构建表达式
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

            self._log(f"        表达式: {expr}")

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

    # ==================== 主流程 ====================
    def _discover_arithmetic_rules(self, df: pd.DataFrame,
                                   numeric_cols: List[str]) -> List[Dict]:
        """
        主流程：
        1. 按相关图聚类
        2. 对每个聚类按共同非空拆分子类（非互斥）
        3. 对每个子类发现关系
        """
        self._log("  [数值关系] 开始...")

        if len(numeric_cols) < 3:
            return []

        # 步骤2：相关图聚类
        clusters = self._cluster_by_correlation(df, numeric_cols)
        self._log(f"    相关聚类数: {len(clusters)}")

        all_rules = []

        for i, cluster in enumerate(clusters):
            if len(cluster) < 3:
                continue

            self._log(f"    聚类{i + 1}: {len(cluster)}个变量")

            # 步骤3：按共同非空拆分子类（非互斥）
            subclusters = self._split_by_cooccurrence_non_exclusive(df, cluster)
            self._log(f"      拆分为 {len(subclusters)} 个子类")

            # 步骤4：对每个子类发现关系
            for subcluster in subclusters:
                if len(subcluster) < 3:
                    continue

                rules = self._discover_rules_in_subcluster(df, subcluster)
                all_rules.extend(rules)

        # 步骤5：去重
        seen = set()
        unique_rules = []

        for rule in all_rules:
            key = frozenset(rule["fields"])
            if key not in seen:
                seen.add(key)
                unique_rules.append(rule)

        self._log(f"    总计: {len(unique_rules)} 条数值关系")
        return unique_rules

    # ==================== 等式验证 ====================
    def verify_rule(self, df: pd.DataFrame, fields: List[str], coeffs: List[int]) -> Dict:
        """
        验证指定等式是否成立

        参数:
        - fields: 字段列表 [A, B, C, D]
        - coeffs: 系数列表 [1, 1, -1, -1] 表示 A + B = C + D

        返回: {
            "success": bool,
            "fields": List[str],
            "coeffs": List[int],
            "confidence": float,
            "valid_rows": int,
            "total_rows": int,
            "violations": int
        }
        """
        if len(fields) != len(coeffs):
            return {
                "success": False,
                "error": "fields 和 coeffs 长度不匹配"
            }

        # 检查字段是否存在
        missing_fields = [f for f in fields if f not in df.columns]
        if missing_fields:
            return {
                "success": False,
                "error": f"字段不存在: {missing_fields}"
            }

        # 取共同非空的数据
        valid_df = self._get_valid_data(df, fields)
        valid_rows = len(valid_df)

        if valid_rows == 0:
            return {
                "success": False,
                "fields": fields,
                "coeffs": coeffs,
                "confidence": 0.0,
                "valid_rows": 0,
                "total_rows": len(df),
                "violations": 0,
                "error": "没有共同非空的行"
            }

        X = valid_df[fields].values
        result = X @ np.array(coeffs)

        # 计算相对误差
        scale = np.max(np.abs(X), axis=1)
        scale = np.maximum(scale, 1)
        rel_error = np.abs(result) / scale
        violations = (rel_error >= 1e-4).sum()
        confidence = (rel_error < 1e-4).mean()

        # 构建表达式
        left_parts = []
        right_parts = []
        for field, coeff in zip(fields, coeffs):
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

        return {
            "success": True,
            "rule": expr,
            "fields": fields,
            "coeffs": coeffs,
            "confidence": round(confidence, 4),
            "valid_rows": valid_rows,
            "total_rows": len(df),
            "violations": violations,
            "violation_rate": round(violations / valid_rows, 4) if valid_rows > 0 else 0
        }

    def verify_rules_batch(self, df: pd.DataFrame, rules: List[Tuple[List[str], List[int]]]) -> List[Dict]:
        """
        批量验证多个等式

        参数:
        - rules: 规则列表，每个元素为 (fields, coeffs)

        返回: 验证结果列表
        """
        results = []
        for fields, coeffs in rules:
            result = self.verify_rule(df, fields, coeffs)
            results.append(result)
            if self.debug:
                if result["success"]:
                    self._log(
                        f"      验证规则 {result['rule']}: 置信度={result['confidence']:.4f}, 有效行数={result['valid_rows']}, 违反数={result['violations']}")
                else:
                    self._log(f"      验证规则失败: {result.get('error', 'unknown')}")
        return results

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
    discoverer = AuditRuleDiscoverer(debug=debug)
    return discoverer.verify_rules_batch(data, rules)