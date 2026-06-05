# date_rules.py
"""
日期关系发现模块

发现数据集中日期字段之间的各种关系，包括：
- 基本时序关系：先后顺序、相等、固定自然日间隔
- 工作日间隔关系：固定工作日间隔
- 顺延关系：非工作日自动顺延到下一个工作日后形成的间隔
- 条件时序关系：在分类变量不同取值下，日期关系的变化

输出规则格式与 audit.py 保持一致，便于集成。
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Dict, Optional, Any, Tuple
from datetime import timedelta
import re

# 尝试导入节假日库
try:
    import chinese_calendar as cc

    HAS_CHINESE_CALENDAR = True
except ImportError:
    HAS_CHINESE_CALENDAR = False
    print("⚠️ 未安装 chinese_calendar，将使用简单的周末判断")


class DateRuleDiscoverer:
    """日期关系发现器"""

    def __init__(self,
                 debug: bool = False,
                 min_confidence: float = 0.9,
                 min_coverage: float = 0.5,
                 min_nonnull: int = 10,
                 holidays: List[str] = None,
                 use_chinese_calendar: bool = True,
                 tolerance_days: int = 1,
                 consider_workday: bool = True,
                 consider_shifted: bool = True,
                 consider_conditional: bool = True):
        """
        参数
        ----------
        debug : bool
            是否输出调试信息
        min_confidence : float
            最小置信度阈值，低于此值的规则不输出
        min_coverage : float
            最小覆盖比例，规则至少需要覆盖的数据比例
        min_nonnull : int
            最小非空行数，参与检查的日期对至少需要这么多行
        holidays : List[str]
            自定义节假日列表（可选），格式如 ['2025-01-01', '2025-01-28']
        use_chinese_calendar : bool
            是否使用中国节假日库（需要安装 chinese_calendar）
        tolerance_days : int
            日期比较时的允许误差天数（用于固定间隔）
        consider_workday : bool
            是否发现工作日间隔关系
        consider_shifted : bool
            是否发现顺延关系
        consider_conditional : bool
            是否发现条件时序关系（与分类变量结合）
        """
        self.debug = debug
        self.min_confidence = min_confidence
        self.min_coverage = min_coverage
        self.min_nonnull = min_nonnull
        self.custom_holidays = holidays if holidays is not None else []
        self.use_chinese_calendar = use_chinese_calendar and HAS_CHINESE_CALENDAR
        self.tolerance_days = tolerance_days
        self.consider_workday = consider_workday
        self.consider_shifted = consider_shifted
        self.consider_conditional = consider_conditional

    def _log(self, msg: str):
        if self.debug:
            print(f"  [DateRule] {msg}")

    def _is_workday(self, d: pd.Timestamp) -> bool:
        """
        判断单个日期是否为工作日
        使用 chinese_calendar 库判断中国法定节假日和调休
        """
        if pd.isnull(d):
            return False

        # 转换为 date 对象
        date_obj = d.date()

        # 检查自定义节假日
        date_str = d.strftime('%Y-%m-%d')
        if date_str in self.custom_holidays:
            return False

        # 使用中国节假日库
        if self.use_chinese_calendar:
            try:
                # chinese_calendar 的 is_workday 会自动处理法定假日和调休
                return cc.is_workday(date_obj)
            except Exception:
                # 降级：简单周末判断
                return d.weekday() < 5

        # 简单周末判断
        return d.weekday() < 5

    def _next_workday(self, d: pd.Timestamp) -> pd.Timestamp:
        """返回 d 之后（含 d）的第一个工作日"""
        if pd.isnull(d):
            return d
        result = d
        while not self._is_workday(result):
            result += timedelta(days=1)
        return result

    def _workday_count(self, start: pd.Timestamp, end: pd.Timestamp) -> int:
        """
        返回从 start 到 end 之间的工作日数（不含 start，含 end）

        例如：周一 to 周三 = 2个工作日（周二、周三）
             周五 to 下周一 = 1个工作日（周一，因为周末不算）
        """
        if pd.isnull(start) or pd.isnull(end):
            return np.nan
        if end < start:
            return -1

        count = 0
        current = start
        while current < end:
            current += timedelta(days=1)
            if self._is_workday(current):
                count += 1
        return count

    def _make_rule(self,
                   rule_str: str,
                   confidence: float,
                   fields: List[str],
                   valid_rows: int = 0,
                   satisfied_rows: int = 0) -> Dict:
        """构建标准规则字典"""
        return {
            'rule': rule_str,
            'fields': fields,
            'confidence': round(confidence, 4),
            'priority': '高' if confidence == 1.0 else '中',
            'relation_type': 'temporal',
            'violation_count': valid_rows - satisfied_rows,
            'violation_samples': [],
            'valid_rows': valid_rows,
            'satisfied_rows': satisfied_rows
        }

    def _deduplicate_rules(self, rules: List[Dict]) -> List[Dict]:
        """基于规则字符串和字段集合去重"""
        seen = set()
        unique = []
        for r in rules:
            key = (r['rule'], tuple(sorted(r['fields'])))
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def _validate_basic_rule(self,
                             df: pd.DataFrame,
                             col1: str,
                             col2: str,
                             rel_type: str,
                             days: Optional[int] = None) -> Tuple[float, int, int]:
        """验证基本时序规则"""
        mask = df[col1].notna() & df[col2].notna()
        valid_df = df.loc[mask, [col1, col2]]
        valid_rows = len(valid_df)
        if valid_rows == 0:
            return 0.0, 0, 0

        c1 = pd.to_datetime(valid_df[col1])
        c2 = pd.to_datetime(valid_df[col2])

        if rel_type == 'le':
            satisfied = (c1 <= c2).sum()
        elif rel_type == 'eq':
            satisfied = (c1 == c2).sum()
        elif rel_type == 'diff':
            diff = (c2 - c1).dt.days
            satisfied = (diff == days).sum()
        else:
            return 0.0, 0, 0

        confidence = satisfied / valid_rows if valid_rows > 0 else 0.0
        return confidence, valid_rows, satisfied

    def _validate_workday_rule(self,
                               df: pd.DataFrame,
                               col1: str,
                               col2: str,
                               wdays: int) -> Tuple[float, int, int]:
        """验证工作日间隔规则"""
        mask = df[col1].notna() & df[col2].notna()
        valid_df = df.loc[mask, [col1, col2]]
        valid_rows = len(valid_df)
        if valid_rows == 0:
            return 0.0, 0, 0

        satisfied = 0
        for idx, row in valid_df.iterrows():
            cnt = self._workday_count(row[col1], row[col2])
            if cnt == wdays:
                satisfied += 1
        confidence = satisfied / valid_rows
        return confidence, valid_rows, satisfied

    def _validate_shifted_rule(self,
                               df: pd.DataFrame,
                               col1: str,
                               col2: str,
                               days: int) -> Tuple[float, int, int]:
        """验证顺延规则"""
        mask = df[col1].notna() & df[col2].notna()
        valid_df = df.loc[mask, [col1, col2]]
        valid_rows = len(valid_df)
        if valid_rows == 0:
            return 0.0, 0, 0

        satisfied = 0
        for idx, row in valid_df.iterrows():
            shifted = self._next_workday(row[col2])
            if pd.notnull(shifted):
                diff = (shifted - row[col1]).days
                if diff == days:
                    satisfied += 1
        confidence = satisfied / valid_rows
        return confidence, valid_rows, satisfied

    # ---------- 关系发现方法 ----------
    def _discover_basic_temporal(self, df: pd.DataFrame, date_cols: List[str]) -> List[Dict]:
        """发现基本时序关系：≤, =, +n天"""
        self._log("   发现基本时序关系...")
        rules = []
        for col1, col2 in combinations(date_cols, 2):
            valid_mask = df[col1].notna() & df[col2].notna()
            if valid_mask.sum() < self.min_nonnull:
                continue

            c1 = pd.to_datetime(df.loc[valid_mask, col1])
            c2 = pd.to_datetime(df.loc[valid_mask, col2])

            # 1. 检查 ≤
            if (c1 <= c2).all():
                rule = self._make_rule(f"{col1} ≤ {col2}", 1.0, [col1, col2])
                rules.append(rule)
                continue

            # 2. 检查相等
            if (c1 == c2).all():
                rule = self._make_rule(f"{col1} = {col2}", 1.0, [col1, col2])
                rules.append(rule)
                continue

            # 3. 检查固定自然日间隔
            diff = (c2 - c1).dt.days
            if diff.nunique() == 1 and len(diff) > 0:
                days = int(diff.iloc[0])
                if days != 0:
                    rule = self._make_rule(f"{col2} = {col1} + {days}天", 1.0, [col1, col2])
                    rules.append(rule)

        # 全量验证
        validated = []
        for r in rules:
            rule_str = r['rule']
            fields = r['fields']
            col1, col2 = fields[0], fields[1]
            if '≤' in rule_str:
                conf, vrows, srows = self._validate_basic_rule(df, col1, col2, 'le')
            elif '=' in rule_str and '+' not in rule_str:
                conf, vrows, srows = self._validate_basic_rule(df, col1, col2, 'eq')
            elif '+' in rule_str and '天' in rule_str:
                match = re.search(r'\+ (\d+)天', rule_str)
                if match:
                    days = int(match.group(1))
                    conf, vrows, srows = self._validate_basic_rule(df, col1, col2, 'diff', days)
                else:
                    continue
            else:
                continue
            if conf >= self.min_confidence:
                r.update({'confidence': round(conf, 4), 'valid_rows': vrows, 'satisfied_rows': srows,
                          'violation_count': vrows - srows})
                validated.append(r)
        return validated

    def _discover_workday_interval(self, df: pd.DataFrame, date_cols: List[str]) -> List[Dict]:
        """发现固定工作日间隔关系"""
        if not self.consider_workday:
            return []
        self._log("   发现工作日间隔关系...")
        rules = []
        for col1, col2 in combinations(date_cols, 2):
            mask = df[col1].notna() & df[col2].notna()
            if mask.sum() < self.min_nonnull:
                continue

            wdays_list = []
            for idx in df[mask].index:
                cnt = self._workday_count(df.loc[idx, col1], df.loc[idx, col2])
                if not np.isnan(cnt):
                    wdays_list.append(cnt)

            if not wdays_list:
                continue

            unique_vals = np.unique(wdays_list)

            # 调试输出
            if self.debug and len(unique_vals) <= 5:
                self._log(f"      {col1} → {col2}: 唯一值={unique_vals.tolist()}, 样本前5={wdays_list[:5]}")

            if len(unique_vals) == 1:
                wdays = int(unique_vals[0])
                if wdays == 0:
                    continue
                rule = self._make_rule(f"{col2} = {col1} + {wdays}个工作日", 1.0, [col1, col2])
                rules.append(rule)

        validated = []
        for r in rules:
            match = re.search(r'\+ (\d+)个工作日', r['rule'])
            if match:
                wdays = int(match.group(1))
                conf, vrows, srows = self._validate_workday_rule(df, r['fields'][0], r['fields'][1], wdays)
                if conf >= self.min_confidence:
                    r.update({'confidence': round(conf, 4), 'valid_rows': vrows, 'satisfied_rows': srows,
                              'violation_count': vrows - srows})
                    validated.append(r)
        self._log(f"     发现 {len(validated)} 条")
        return validated

    def _discover_shifted_relation(self, df: pd.DataFrame, date_cols: List[str]) -> List[Dict]:
        """发现顺延关系"""
        if not self.consider_shifted:
            return []
        self._log("   发现顺延关系...")
        rules = []
        for col1, col2 in combinations(date_cols, 2):
            mask = df[col1].notna() & df[col2].notna()
            if mask.sum() < self.min_nonnull:
                continue

            diff_list = []
            for idx in df[mask].index:
                d1 = df.loc[idx, col1]
                d2 = df.loc[idx, col2]
                shifted = self._next_workday(d2)
                if pd.notnull(shifted):
                    diff = (shifted - d1).days
                    diff_list.append(diff)
            if not diff_list:
                continue
            unique_vals = np.unique(diff_list)
            if len(unique_vals) == 1:
                days = int(unique_vals[0])
                rule = self._make_rule(f"顺延后 {col2} = {col1} + {days}天", 1.0, [col1, col2])
                rules.append(rule)

        validated = []
        for r in rules:
            match = re.search(r'\+ (\d+)天', r['rule'])
            if match:
                days = int(match.group(1))
                conf, vrows, srows = self._validate_shifted_rule(df, r['fields'][0], r['fields'][1], days)
                if conf >= self.min_confidence:
                    r.update({'confidence': round(conf, 4), 'valid_rows': vrows, 'satisfied_rows': srows,
                              'violation_count': vrows - srows})
                    validated.append(r)
        self._log(f"     发现 {len(validated)} 条")
        return validated

    def _discover_conditional_temporal(self, df: pd.DataFrame,
                                       date_cols: List[str],
                                       categorical_cols: List[str]) -> List[Dict]:
        """发现条件时序关系"""
        if not self.consider_conditional or not categorical_cols:
            return []
        self._log("   发现条件时序关系...")
        rules = []

        for cat_col in categorical_cols:
            values = df[cat_col].dropna().unique()
            if len(values) < 2:
                continue
            for val in values:
                subset = df[df[cat_col] == val]
                if len(subset) < self.min_nonnull:
                    continue

                for col1, col2 in combinations(date_cols, 2):
                    mask = subset[col1].notna() & subset[col2].notna()
                    if mask.sum() < self.min_nonnull:
                        continue
                    c1 = pd.to_datetime(subset.loc[mask, col1])
                    c2 = pd.to_datetime(subset.loc[mask, col2])

                    # ≤ 关系
                    if (c1 <= c2).all():
                        inner = {'type': 'basic', 'rel_type': 'le', 'col1': col1, 'col2': col2}
                        conf, vrows, srows = self._validate_conditional_rule(df, cat_col, val, inner)
                        if conf >= self.min_confidence and vrows >= self.min_nonnull:
                            rule_str = f"当 {cat_col} = {val} 时，{col1} ≤ {col2}"
                            rule = self._make_rule(rule_str, conf, [cat_col, col1, col2], vrows, srows)
                            rules.append(rule)

                    # 相等关系
                    if (c1 == c2).all():
                        inner = {'type': 'basic', 'rel_type': 'eq', 'col1': col1, 'col2': col2}
                        conf, vrows, srows = self._validate_conditional_rule(df, cat_col, val, inner)
                        if conf >= self.min_confidence and vrows >= self.min_nonnull:
                            rule_str = f"当 {cat_col} = {val} 时，{col1} = {col2}"
                            rule = self._make_rule(rule_str, conf, [cat_col, col1, col2], vrows, srows)
                            rules.append(rule)

                    # 固定自然日间隔
                    diff = (c2 - c1).dt.days
                    if diff.nunique() == 1 and len(diff) > 0:
                        days = int(diff.iloc[0])
                        if days != 0:
                            inner = {'type': 'basic', 'rel_type': 'diff', 'col1': col1, 'col2': col2, 'days': days}
                            conf, vrows, srows = self._validate_conditional_rule(df, cat_col, val, inner)
                            if conf >= self.min_confidence and vrows >= self.min_nonnull:
                                rule_str = f"当 {cat_col} = {val} 时，{col2} = {col1} + {days}天"
                                rule = self._make_rule(rule_str, conf, [cat_col, col1, col2], vrows, srows)
                                rules.append(rule)

                    # 工作日间隔
                    if self.consider_workday:
                        wdays_list = []
                        for idx in subset[mask].index:
                            cnt = self._workday_count(subset.loc[idx, col1], subset.loc[idx, col2])
                            if not np.isnan(cnt):
                                wdays_list.append(cnt)
                        if wdays_list and np.unique(wdays_list).size == 1:
                            wdays = int(wdays_list[0])
                            if wdays != 0:
                                inner = {'type': 'workday', 'col1': col1, 'col2': col2, 'wdays': wdays}
                                conf, vrows, srows = self._validate_conditional_rule(df, cat_col, val, inner)
                                if conf >= self.min_confidence and vrows >= self.min_nonnull:
                                    rule_str = f"当 {cat_col} = {val} 时，{col2} = {col1} + {wdays}个工作日"
                                    rule = self._make_rule(rule_str, conf, [cat_col, col1, col2], vrows, srows)
                                    rules.append(rule)

                    # 顺延关系
                    if self.consider_shifted:
                        diff_shift = []
                        for idx in subset[mask].index:
                            d1 = subset.loc[idx, col1]
                            d2 = subset.loc[idx, col2]
                            shifted = self._next_workday(d2)
                            if pd.notnull(shifted):
                                diff_shift.append((shifted - d1).days)
                        if diff_shift and np.unique(diff_shift).size == 1:
                            days = int(diff_shift[0])
                            inner = {'type': 'shifted', 'col1': col1, 'col2': col2, 'days': days}
                            conf, vrows, srows = self._validate_conditional_rule(df, cat_col, val, inner)
                            if conf >= self.min_confidence and vrows >= self.min_nonnull:
                                rule_str = f"当 {cat_col} = {val} 时，顺延后 {col2} = {col1} + {days}天"
                                rule = self._make_rule(rule_str, conf, [cat_col, col1, col2], vrows, srows)
                                rules.append(rule)

        self._log(f"     发现 {len(rules)} 条")
        return self._deduplicate_rules(rules)

    def _validate_conditional_rule(self,
                                   df: pd.DataFrame,
                                   cat_col: str,
                                   cat_val: Any,
                                   inner_rule: Dict) -> Tuple[float, int, int]:
        """验证条件时序规则"""
        subset = df[df[cat_col] == cat_val]
        if len(subset) < self.min_nonnull:
            return 0.0, 0, 0

        itype = inner_rule.get('type')
        col1 = inner_rule['col1']
        col2 = inner_rule['col2']

        if itype == 'basic':
            rel_type = inner_rule['rel_type']
            days = inner_rule.get('days', None)
            conf, vrows, srows = self._validate_basic_rule(subset, col1, col2, rel_type, days)
        elif itype == 'workday':
            wdays = inner_rule['wdays']
            conf, vrows, srows = self._validate_workday_rule(subset, col1, col2, wdays)
        elif itype == 'shifted':
            days = inner_rule['days']
            conf, vrows, srows = self._validate_shifted_rule(subset, col1, col2, days)
        else:
            return 0.0, 0, 0

        return conf, vrows, srows

    # ---------- 主入口 ----------
    def discover_all(self,
                     df: pd.DataFrame,
                     date_cols: List[str],
                     categorical_cols: Optional[List[str]] = None) -> List[Dict]:
        """
        发现所有日期关系规则
        """
        self._log("开始发现日期关系")
        if self.use_chinese_calendar:
            self._log("使用中国节假日库 (chinese_calendar)")
        all_rules = []

        # 1. 基本时序关系
        basic_rules = self._discover_basic_temporal(df, date_cols)
        all_rules.extend(basic_rules)

        # 2. 工作日间隔关系
        if self.consider_workday:
            workday_rules = self._discover_workday_interval(df, date_cols)
            all_rules.extend(workday_rules)

        # 3. 顺延关系
        if self.consider_shifted:
            shifted_rules = self._discover_shifted_relation(df, date_cols)
            all_rules.extend(shifted_rules)

        # 4. 条件时序关系
        if categorical_cols and self.consider_conditional:
            cond_rules = self._discover_conditional_temporal(df, date_cols, categorical_cols)
            all_rules.extend(cond_rules)

        unique_rules = self._deduplicate_rules(all_rules)
        self._log(f"总计发现 {len(unique_rules)} 条日期关系规则")
        return unique_rules


def discover_date_rules(data: pd.DataFrame,
                        date_columns: List[str],
                        categorical_columns: List[str] = None,
                        debug: bool = False,
                        **kwargs) -> List[Dict]:
    """
    便捷函数：发现日期字段间的勾稽关系

    参数
    ----------
    data : pd.DataFrame
        输入数据
    date_columns : List[str]
        日期字段列表
    categorical_columns : List[str], optional
        分类字段列表，用于发现条件规则
    debug : bool
        是否打印调试信息
    **kwargs
        传递给 DateRuleDiscoverer 的其他参数

    返回
    -------
    List[Dict]
        规则列表
    """
    discoverer = DateRuleDiscoverer(debug=debug, **kwargs)
    return discoverer.discover_all(data, date_columns, categorical_columns)