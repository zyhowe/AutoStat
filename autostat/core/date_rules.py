# date_rules.py
"""
日期关系发现模块 - 优化版
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Dict, Optional, Any, Tuple
from datetime import timedelta, datetime
import re
from collections import Counter

try:
    import chinese_calendar as cc

    HAS_CHINESE_CALENDAR = True
except ImportError:
    HAS_CHINESE_CALENDAR = False


class DateRuleDiscoverer:
    """日期关系发现器 - 优化版"""

    def __init__(self,
                 debug: bool = False,
                 min_confidence: float = 0.7,
                 min_coverage: float = 0.5,
                 min_nonnull: int = 10,
                 min_cooccurrence: int = 50,
                 holidays: List[str] = None,
                 use_chinese_calendar: bool = True,
                 consider_workday: bool = True,
                 consider_shifted: bool = False,
                 consider_conditional: bool = True,
                 max_categorical_cardinality: int = 20,
                 effect_size_threshold: float = 0.1):
        self.debug = debug
        self.min_confidence = min_confidence
        self.min_coverage = min_coverage
        self.min_nonnull = min_nonnull
        self.min_cooccurrence = min_cooccurrence
        self.custom_holidays = holidays if holidays is not None else []
        self.use_chinese_calendar = use_chinese_calendar and HAS_CHINESE_CALENDAR
        self.consider_workday = consider_workday
        self.consider_shifted = consider_shifted
        self.consider_conditional = consider_conditional
        self.max_categorical_cardinality = max_categorical_cardinality
        self.effect_size_threshold = effect_size_threshold

        self._workday_cache = {}
        self._workday_series_cache = {}
        self._cache_max_size = 100

    def _log(self, msg: str):
        if self.debug:
            print(f"  [DateRule] {datetime.now().strftime('%H:%M:%S')} {msg}")

    def _is_workday(self, d: pd.Timestamp) -> bool:
        if pd.isnull(d):
            return False

        if hasattr(d, 'date'):
            date_key = d.date()
        else:
            date_key = d

        if date_key in self._workday_cache:
            return self._workday_cache[date_key]

        date_str = d.strftime('%Y-%m-%d')
        if date_str in self.custom_holidays:
            self._workday_cache[date_key] = False
            return False

        if self.use_chinese_calendar:
            try:
                result = cc.is_workday(d.date())
                self._workday_cache[date_key] = result
                return result
            except Exception:
                result = d.weekday() < 5
                self._workday_cache[date_key] = result
                return result

        result = d.weekday() < 5
        self._workday_cache[date_key] = result
        return result

    def _get_workday_mask(self, min_date: pd.Timestamp, max_date: pd.Timestamp) -> np.ndarray:
        """获取日期范围内的工作日掩码（带缓存）"""
        cache_key = (min_date, max_date)
        if cache_key in self._workday_series_cache:
            return self._workday_series_cache[cache_key]

        # 限制缓存大小
        if len(self._workday_series_cache) >= self._cache_max_size:
            # 移除最早的一个
            oldest_key = next(iter(self._workday_series_cache))
            del self._workday_series_cache[oldest_key]

        date_range = pd.date_range(min_date, max_date, freq='D')
        is_workday = np.array([self._is_workday(d) for d in date_range])
        self._workday_series_cache[cache_key] = is_workday
        return is_workday

    def _workday_count_series(self, starts: pd.Series, ends: pd.Series) -> np.ndarray:
        """向量化计算工作日间隔（使用预计算工作日掩码）"""
        if len(starts) == 0:
            return np.array([])

        starts = starts.reset_index(drop=True)
        ends = ends.reset_index(drop=True)

        starts = pd.to_datetime(starts)
        ends = pd.to_datetime(ends)

        result = np.full(len(starts), -1, dtype=np.int64)

        # 找出全局最小和最大日期
        min_date = min(starts.min(), ends.min())
        max_date = max(starts.max(), ends.max())

        # 如果日期范围太大（超过10年），直接返回-1
        if (max_date - min_date).days > 3650:
            return result

        # 获取预计算的工作日掩码
        is_workday = self._get_workday_mask(min_date, max_date)
        date_range = pd.date_range(min_date, max_date, freq='D')

        for i in range(len(starts)):
            start = starts.iloc[i]
            end = ends.iloc[i]
            if pd.isnull(start) or pd.isnull(end):
                continue
            if end < start:
                continue
            if start == end:
                result[i] = 0
                continue

            # 使用预计算掩码快速求和
            mask = (date_range > start) & (date_range <= end)
            count = is_workday[mask].sum()
            result[i] = count

        return result

    def _make_rule(self, rule_str: str, confidence: float, fields: List[str],
                   valid_rows: int = 0, satisfied_rows: int = 0) -> Dict:
        return {
            'rule': rule_str,
            'fields': fields,
            'confidence': round(confidence, 4),
            'priority': '高' if confidence >= 0.95 else '中',
            'relation_type': 'temporal',
            'violation_count': valid_rows - satisfied_rows,
            'violation_samples': [],
            'valid_rows': valid_rows,
            'satisfied_rows': satisfied_rows
        }

    def _deduplicate_rules(self, rules: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for r in rules:
            key = (r['rule'], tuple(sorted(r['fields'])))
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def _filter_valid_date_columns(self, df: pd.DataFrame, date_cols: List[str]) -> List[str]:
        """过滤无效日期列：只保留非空率足够高的列"""
        valid_cols = []
        for col in date_cols:
            if col not in df.columns:
                continue
            nonnull_ratio = df[col].notna().mean()
            nonnull_count = df[col].notna().sum()

            if nonnull_count >= self.min_nonnull and nonnull_ratio >= 0.1:
                valid_cols.append(col)
            else:
                self._log(f"  跳过稀疏日期列 {col}: 非空={nonnull_count}, 非空率={nonnull_ratio:.1%}")

        return valid_cols

    def _filter_date_pairs(self, df: pd.DataFrame, date_cols: List[str]) -> List[Tuple[str, str]]:
        """筛选日期对：只保留共现行达到阈值的组合"""
        if len(date_cols) < 2:
            return []

        date_min = {}
        for col in date_cols:
            if col not in df.columns:
                continue
            valid = df[col].dropna()
            if len(valid) > 0:
                date_min[col] = valid.min()
            else:
                date_min[col] = None

        sorted_cols = [col for col in sorted(date_min.keys(),
                                             key=lambda x: date_min[x] if date_min[x] is not None else pd.Timestamp.max)
                       if date_min[col] is not None]

        if len(sorted_cols) < 2:
            return []

        min_col = sorted_cols[0]
        pairs = [(min_col, col) for col in sorted_cols[1:]]
        adjacent_pairs = [(sorted_cols[i], sorted_cols[i + 1]) for i in range(len(sorted_cols) - 1)]

        all_pairs = pairs + adjacent_pairs
        unique_pairs = []
        seen = set()
        for p in all_pairs:
            if p not in seen:
                seen.add(p)
                unique_pairs.append(p)

        filtered_pairs = []
        for col1, col2 in unique_pairs:
            if col1 not in df.columns or col2 not in df.columns:
                continue
            cooccurrence = (df[col1].notna() & df[col2].notna()).sum()
            if cooccurrence >= self.min_cooccurrence:
                filtered_pairs.append((col1, col2))
            else:
                self._log(f"  跳过 {col1} → {col2}: 共现行={cooccurrence} < {self.min_cooccurrence}")

        if self.debug:
            self._log(f"日期对筛选: {len(date_cols)}个字段 → {len(filtered_pairs)}个候选对")
            self._log(f"  最小日期: {min_col}")
            self._log(f"  排序: {sorted_cols}")

        return filtered_pairs

    def _filter_conditional_vars(self, df: pd.DataFrame, date_cols: List[str],
                                 categorical_cols: List[str]) -> List[Tuple[str, List[Any]]]:
        """筛选有显著影响的条件变量（使用采样加速）"""
        if not categorical_cols or not self.consider_conditional:
            return []

        # 排除日期派生字段
        date_derived_suffixes = ('_year', '_month', '_quarter', '_week', '_weekday', '_day', '_is_weekend')
        filtered_categorical_cols = []
        for col in categorical_cols:
            if not any(col.endswith(suffix) for suffix in date_derived_suffixes):
                filtered_categorical_cols.append(col)
            else:
                self._log(f"  排除日期派生字段: {col}")

        if not filtered_categorical_cols:
            self._log("  无有效分类变量（已排除日期派生字段）")
            return []

        result = []

        pairs = self._filter_date_pairs(df, date_cols)
        if not pairs:
            return []

        best_pair = None
        best_count = 0
        for col1, col2 in pairs:
            cnt = (df[col1].notna() & df[col2].notna()).sum()
            if cnt > best_count:
                best_count = cnt
                best_pair = (col1, col2)

        if best_pair is None:
            return []

        col1, col2 = best_pair

        valid_mask = df[col1].notna() & df[col2].notna()
        valid_indices = df[valid_mask].index

        if len(valid_indices) < self.min_nonnull:
            return []

        # 采样计算 η²（采样2000行，既保证精度又加速）
        sample_size = min(2000, len(valid_indices))
        sample_indices = np.random.choice(valid_indices, size=sample_size, replace=False)

        intervals = self._workday_count_series(
            df.loc[sample_indices, col1],
            df.loc[sample_indices, col2]
        )
        valid_interval_mask = intervals >= 0
        valid_interval_indices = sample_indices[valid_interval_mask]

        if len(valid_interval_indices) < self.min_nonnull:
            return []

        intervals_series = pd.Series(
            intervals[valid_interval_mask].tolist(),
            index=valid_interval_indices
        )

        for cat_col in filtered_categorical_cols:
            if cat_col not in df.columns:
                continue

            n_unique = df[cat_col].nunique()
            if n_unique > self.max_categorical_cardinality:
                self._log(f"  跳过 {cat_col}: 基数={n_unique} > 阈值{self.max_categorical_cardinality}")
                continue

            cat_values = df.loc[intervals_series.index, cat_col]
            if cat_values.isna().all():
                continue

            unique_cats = cat_values.dropna().unique()
            if len(unique_cats) < 2:
                continue

            group_means = {}
            group_counts = {}
            for cat in unique_cats:
                mask = cat_values == cat
                group_intervals = intervals_series[mask]
                if len(group_intervals) >= 3:
                    group_means[cat] = group_intervals.mean()
                    group_counts[cat] = len(group_intervals)

            if len(group_means) < 2:
                continue

            total_mean = np.mean(intervals_series)
            ss_between = sum(group_counts[cat] * (mean - total_mean) ** 2
                             for cat, mean in group_means.items())
            ss_total = sum((x - total_mean) ** 2 for x in intervals_series)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            self._log(f"  {cat_col}: η²={eta_squared:.4f}")

            if eta_squared >= self.effect_size_threshold:
                values = df[cat_col].dropna().unique().tolist()
                result.append((cat_col, values))
                self._log(f"    ✅ 保留 {cat_col} (效应量={eta_squared:.4f})")

        return result

    def _discover_basic_temporal(self, df: pd.DataFrame, date_cols: List[str]) -> List[Dict]:
        """发现基本时序关系：≤, =, +n天（自然日，使用众数+阈值）"""
        self._log("   发现基本时序关系...")
        rules = []
        pairs = self._filter_date_pairs(df, date_cols)

        for col1, col2 in pairs:
            valid_mask = df[col1].notna() & df[col2].notna()
            if valid_mask.sum() < self.min_nonnull:
                continue

            c1 = pd.to_datetime(df.loc[valid_mask, col1])
            c2 = pd.to_datetime(df.loc[valid_mask, col2])

            if (c1 <= c2).all():
                rule = self._make_rule(f"{col1} ≤ {col2}", 1.0, [col1, col2])
                rules.append(rule)
                continue

            if (c1 == c2).all():
                rule = self._make_rule(f"{col1} = {col2}", 1.0, [col1, col2])
                rules.append(rule)
                continue

            diff = (c2 - c1).dt.days
            if len(diff) > 0:
                counter = Counter(diff)
                most_common_days, most_common_count = counter.most_common(1)[0]
                most_common_ratio = most_common_count / len(diff)

                if self.debug:
                    unique_vals = sorted(counter.keys())
                    self._log(
                        f"      {col1} → {col2}: 自然日众数={most_common_days}({most_common_count}/{len(diff)}={most_common_ratio:.1%}), 唯一值数={len(unique_vals)}")

                if most_common_ratio >= self.min_confidence and most_common_days != 0:
                    rule = self._make_rule(
                        f"{col2} = {col1} + {most_common_days}天",
                        most_common_ratio, [col1, col2],
                        valid_rows=len(diff),
                        satisfied_rows=most_common_count
                    )
                    rules.append(rule)

        return rules

    def _discover_workday_interval(self, df: pd.DataFrame, date_cols: List[str]) -> List[Dict]:
        """发现工作日间隔关系（采样预检查+向量化计算）"""
        if not self.consider_workday:
            return []
        self._log("   发现工作日间隔关系...")
        rules = []
        pairs = self._filter_date_pairs(df, date_cols)

        for col1, col2 in pairs:
            valid_mask = df[col1].notna() & df[col2].notna()
            if valid_mask.sum() < self.min_cooccurrence:
                continue

            # 快速预检查：采样100行判断是否有规律
            sample_size = min(100, valid_mask.sum())
            sample_indices = df[valid_mask].sample(n=sample_size, random_state=42).index

            sample_wdays = self._workday_count_series(
                df.loc[sample_indices, col1],
                df.loc[sample_indices, col2]
            )
            sample_valid = sample_wdays[sample_wdays >= 0]
            if len(sample_valid) == 0:
                continue

            sample_counter = Counter(sample_valid.tolist())
            sample_most_common, sample_count = sample_counter.most_common(1)[0]
            sample_ratio = sample_count / len(sample_valid)

            if sample_ratio < self.min_confidence * 0.8:
                if self.debug:
                    self._log(
                        f"      {col1} → {col2}: 采样预检失败 (众数={sample_most_common}, 占比={sample_ratio:.1%} < {self.min_confidence * 0.8:.1%})，跳过")
                continue

            if self.debug:
                self._log(
                    f"      {col1} → {col2}: 采样预检通过 (众数={sample_most_common}, 占比={sample_ratio:.1%})，进行全量计算")

            # 全量计算（向量化）
            valid_indices = df[valid_mask].index
            wdays_array = self._workday_count_series(
                df.loc[valid_indices, col1],
                df.loc[valid_indices, col2]
            )
            valid_idx = wdays_array >= 0
            wdays_list = wdays_array[valid_idx].tolist()

            if not wdays_list:
                continue

            counter = Counter(wdays_list)
            most_common_gap, most_common_count = counter.most_common(1)[0]
            most_common_ratio = most_common_count / len(wdays_list)

            if self.debug:
                unique_vals = sorted(counter.keys())
                self._log(
                    f"      {col1} → {col2}: 众数={most_common_gap}({most_common_count}/{len(wdays_list)}={most_common_ratio:.1%}), 唯一值数={len(unique_vals)}")

            if most_common_ratio >= self.min_confidence and most_common_gap != 0:
                rule = self._make_rule(
                    f"{col2} = {col1} + {most_common_gap}个工作日",
                    most_common_ratio, [col1, col2],
                    valid_rows=len(wdays_list),
                    satisfied_rows=most_common_count
                )
                rules.append(rule)

        # 全量验证
        validated = []
        for r in rules:
            match = re.search(r'\+ (\d+)个工作日', r['rule'])
            if match:
                wdays = int(match.group(1))
                conf, vrows, srows = self._validate_workday_rule(df, r['fields'][0], r['fields'][1], wdays)
                if conf >= self.min_confidence:
                    r.update({'confidence': round(conf, 4), 'valid_rows': vrows,
                              'satisfied_rows': srows, 'violation_count': vrows - srows})
                    validated.append(r)

        self._log(f"     发现 {len(validated)} 条工作日间隔规则")
        return validated

    def _discover_shifted_relation(self, df: pd.DataFrame, date_cols: List[str]) -> List[Dict]:
        """发现顺延关系（默认关闭）"""
        if not self.consider_shifted:
            return []
        self._log("   发现顺延关系...")
        rules = []

        pairs = self._filter_date_pairs(df, date_cols)

        for col1, col2 in pairs:
            valid_mask = df[col1].notna() & df[col2].notna()
            if valid_mask.sum() < self.min_cooccurrence:
                continue

            valid_indices = df[valid_mask].index

            for shift in [1, 2, 3]:
                match_count = 0
                for idx in valid_indices:
                    a = df.loc[idx, col1]
                    b = df.loc[idx, col2]
                    if pd.isnull(a) or pd.isnull(b):
                        continue
                    shifted = self._next_workday(a + timedelta(days=shift))
                    if shifted == b:
                        match_count += 1

                confidence = match_count / len(valid_indices) if len(valid_indices) > 0 else 0
                if confidence >= self.min_confidence:
                    rule_str = f"{col2} 是 {col1} 顺延 {shift} 个工作日后的下一个工作日"
                    rule = self._make_rule(
                        rule_str, confidence, [col1, col2],
                        valid_rows=len(valid_indices),
                        satisfied_rows=match_count
                    )
                    rules.append(rule)

        self._log(f"     发现 {len(rules)} 条顺延关系规则")
        return rules

    def _discover_conditional_temporal(self, df: pd.DataFrame,
                                       date_cols: List[str],
                                       categorical_cols: List[str]) -> List[Dict]:
        """发现条件时序关系"""
        if not self.consider_conditional or not categorical_cols:
            return []

        self._log("   发现条件时序关系...")

        significant_vars = self._filter_conditional_vars(df, date_cols, categorical_cols)

        if not significant_vars:
            self._log("      无显著影响的分类变量")
            return []

        rules = []
        pairs = self._filter_date_pairs(df, date_cols)

        for cat_col, values in significant_vars:
            self._log(f"    处理分类变量: {cat_col}")

            for val in values:
                subset = df[df[cat_col] == val]
                if len(subset) < self.min_nonnull:
                    continue

                for col1, col2 in pairs:
                    mask = subset[col1].notna() & subset[col2].notna()
                    if mask.sum() < self.min_cooccurrence:
                        continue

                    c1 = pd.to_datetime(subset.loc[mask, col1])
                    c2 = pd.to_datetime(subset.loc[mask, col2])
                    if (c1 <= c2).all():
                        rule_str = f"当 {cat_col} = {val} 时，{col1} ≤ {col2}"
                        rule = self._make_rule(rule_str, 1.0, [cat_col, col1, col2])
                        rules.append(rule)

                    if self.consider_workday:
                        subset_indices = subset[mask].index
                        wdays_array = self._workday_count_series(
                            subset.loc[subset_indices, col1],
                            subset.loc[subset_indices, col2]
                        )
                        valid_idx = wdays_array >= 0
                        wdays_list = wdays_array[valid_idx].tolist()

                        if wdays_list:
                            counter = Counter(wdays_list)
                            most_common_gap, most_common_count = counter.most_common(1)[0]
                            most_common_ratio = most_common_count / len(wdays_list)

                            if most_common_ratio >= self.min_confidence and most_common_gap != 0:
                                rule_str = f"当 {cat_col} = {val} 时，{col2} = {col1} + {most_common_gap}个工作日"
                                rule = self._make_rule(
                                    rule_str, most_common_ratio, [cat_col, col1, col2],
                                    valid_rows=len(wdays_list),
                                    satisfied_rows=most_common_count
                                )
                                rules.append(rule)

        self._log(f"      发现 {len(rules)} 条条件时序规则")
        return self._deduplicate_rules(rules)

    def _validate_workday_rule(self, df: pd.DataFrame, col1: str, col2: str,
                               wdays: int) -> Tuple[float, int, int]:
        mask = df[col1].notna() & df[col2].notna()
        valid_indices = df[mask].index
        valid_rows = len(valid_indices)
        if valid_rows == 0:
            return 0.0, 0, 0

        wdays_array = self._workday_count_series(
            df.loc[valid_indices, col1],
            df.loc[valid_indices, col2]
        )
        valid_idx = wdays_array >= 0
        total_valid = valid_idx.sum()
        satisfied = (wdays_array[valid_idx] == wdays).sum()

        confidence = satisfied / total_valid if total_valid > 0 else 0.0
        return confidence, total_valid, satisfied

    def discover_all(self, df: pd.DataFrame, date_cols: List[str],
                     categorical_cols: Optional[List[str]] = None) -> List[Dict]:
        """发现所有日期关系规则"""
        self._log("开始发现日期关系")
        if self.use_chinese_calendar:
            self._log("使用中国节假日库")

        valid_date_cols = self._filter_valid_date_columns(df, date_cols)
        if len(valid_date_cols) < 2:
            self._log(f"有效日期列不足2个: {len(valid_date_cols)}")
            return []

        self._log(f"有效日期列: {valid_date_cols}")

        all_rules = []

        basic_rules = self._discover_basic_temporal(df, valid_date_cols)
        all_rules.extend(basic_rules)

        if self.consider_workday:
            workday_rules = self._discover_workday_interval(df, valid_date_cols)
            all_rules.extend(workday_rules)

        if self.consider_shifted:
            shifted_rules = self._discover_shifted_relation(df, valid_date_cols)
            all_rules.extend(shifted_rules)

        if categorical_cols and self.consider_conditional:
            cond_rules = self._discover_conditional_temporal(df, valid_date_cols, categorical_cols)
            all_rules.extend(cond_rules)

        unique_rules = self._deduplicate_rules(all_rules)
        filtered = [r for r in unique_rules if r.get('confidence', 0) >= self.min_confidence]

        self._log(f"总计发现 {len(filtered)} 条日期关系规则")
        return filtered


def discover_date_rules(data: pd.DataFrame,
                        date_columns: List[str],
                        categorical_columns: List[str] = None,
                        debug: bool = False,
                        **kwargs) -> List[Dict]:
    """便捷函数：发现日期关系规则"""
    discoverer = DateRuleDiscoverer(debug=debug, **kwargs)
    return discoverer.discover_all(data, date_columns, categorical_columns)