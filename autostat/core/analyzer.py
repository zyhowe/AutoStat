"""
主分析器类 - 整合所有模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os
from typing import Dict, Optional, List

from autostat.loader import DataLoader
from autostat.checker import ConditionChecker
from autostat.core.base import BaseAnalyzer, TYPE_DESCRIPTION_MAP
from autostat.core.timeseries import TimeSeriesAnalyzer
from autostat.core.relationship import RelationshipAnalyzer
from autostat.core.recommendation import RecommendationAnalyzer
from autostat.core.plots import plot_categorical, plot_continuous, plot_timeseries, plot_correlation, plot_categorical_correlation, plot_numeric_categorical_eta

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DATE_FEATURES_CONFIG = {
    "none": [],
    "basic": ["year", "month", "quarter"],
    "full": ["year", "month", "quarter", "week", "weekday", "day", "is_weekend"]
}


class AutoStatisticalAnalyzer:
    """智能统计分析器 - 自动识别数据类型并选择方法"""

    def __init__(self, data, target_col=None, source_table_name=None,
                 predefined_types=None, auto_clean=False, quiet=False,
                 parse_dates=True, date_columns=None, date_features_level="basic",
                 skip_auto_inference=False):
        """
        初始化分析器

        参数:
        - data: DataFrame 或文件路径
        - target_col: 目标列名
        - source_table_name: 源表名
        - predefined_types: 预定义的变量类型字典 {col: type}
        - auto_clean: 是否自动清洗
        - quiet: 静默模式
        - parse_dates: 是否解析日期
        - date_columns: 日期列名列表
        - date_features_level: 日期特征级别
        - skip_auto_inference: 是否跳过自动类型识别（Web模式使用）
        """
        # 处理空数据
        if data is None:
            raise ValueError("data 参数不能为 None")

        if isinstance(data, str):
            print(f"📁 加载文件: {data}")
            if not os.path.exists(data):
                raise FileNotFoundError(f"文件不存在: {data}")
            try:
                self.raw_data = DataLoader.load_from_file(data, parse_dates=parse_dates,
                                                          date_columns=date_columns).copy()
            except Exception as e:
                raise ValueError(f"文件加载失败: {e}")
        elif isinstance(data, pd.DataFrame):
            if data.empty:
                raise ValueError("DataFrame 不能为空")
            self.raw_data = data.copy()
        else:
            raise ValueError("data 参数必须是 DataFrame 或文件路径字符串")

        self.data = self.raw_data.copy()
        self.target_col = target_col
        self.source_table_name = source_table_name
        self.quiet = quiet
        self.date_features_level = date_features_level

        self.variable_types = {}
        self.type_reasons = {}
        self.date_features = {}
        self.date_original_columns = set()
        self.date_derived_columns = set()
        self.date_column_mapping = {}
        self.time_series_diagnostics = {}
        self.quality_report = {}
        self.cleaning_suggestions = []
        self.condition_checker = None

        if not quiet:
            print("\n" + "=" * 70)
            print("🔍 启动智能分析流程")
            print("=" * 70)

        # 阶段1：快速初筛（总是执行，只做数据清洗）
        if not quiet:
            print("\n【阶段1】快速初筛...")
        base = BaseAnalyzer(self.data, quiet=quiet)
        base._quick_pre_screen()
        self.data = base.data
        if not quiet:
            print("  ✅ 完成")

        # 阶段2：类型识别（根据参数决定）
        if not quiet:
            print("\n【阶段2】类型识别...")

        if predefined_types and skip_auto_inference:
            # Web模式：直接使用用户定义的类型
            self.variable_types = predefined_types.copy()
            self.type_reasons = {col: '用户定义' for col in predefined_types.keys()}
            # 过滤掉标记为排除的字段
            excluded_cols = [col for col, typ in predefined_types.items() if typ == 'exclude']
            if excluded_cols:
                self.data = self.data.drop(columns=excluded_cols, errors='ignore')
                if not quiet:
                    print(f"  🗑️ 已排除字段: {excluded_cols}")
            if not quiet:
                print("  ✅ 使用用户定义类型")
        else:
            # MCP/CLI模式：自动识别
            base.variable_types = self.variable_types
            base.type_reasons = self.type_reasons
            base._infer_variable_types()
            self.variable_types = base.variable_types
            self.type_reasons = base.type_reasons
            # 如果有部分预定义，覆盖
            if predefined_types:
                self.variable_types.update(predefined_types)
                self.type_reasons.update({col: '用户定义' for col in predefined_types.keys()})
            if not quiet:
                print("  ✅ 自动类型识别完成")

        # 阶段3：日期特征提取（派生列类型在提取时直接设置，不需要二次识别）
        if not quiet:
            print("\n【阶段3】日期特征提取...")
        self._extract_all_date_features()
        if not quiet:
            print("  ✅ 完成")

        # 阶段4：数据质量体检（总是执行）
        if not quiet:
            print("\n【阶段4】深度数据质量体检...")
        base.variable_types = self.variable_types
        base.quality_report = base._comprehensive_quality_check()
        self.quality_report = base.quality_report
        if not quiet:
            base._print_quality_summary()

        # 阶段5：生成清洗建议（总是执行）
        if not quiet:
            print("\n【阶段5】生成清洗建议...")
        base.cleaning_suggestions = base._generate_cleaning_suggestions()
        self.cleaning_suggestions = base.cleaning_suggestions
        if not quiet:
            base._print_cleaning_suggestions()

        if auto_clean:
            base._auto_clean()
            self.data = base.data
            self.quality_report = base.quality_report
            if not quiet:
                print("  ✅ 已执行自动清洗")

        self.condition_checker = ConditionChecker(
            self.data, self.variable_types,
            self.date_derived_columns, self.date_column_mapping
        )

        self.timeseries_analyzer = TimeSeriesAnalyzer(
            self.data, self.variable_types, self.date_derived_columns, quiet
        )
        self.relationship_analyzer = RelationshipAnalyzer(
            self.data, self.variable_types, self.condition_checker,
            self.date_original_columns, self.date_derived_columns,
            self.date_column_mapping, quiet
        )
        self.recommendation_analyzer = RecommendationAnalyzer(
            self.data, self.variable_types, self.quality_report,
            self.time_series_diagnostics
        )
        self.recommendation_analyzer.set_date_info(
            self.date_derived_columns,
            self.date_column_mapping,
            self.date_original_columns
        )

        if not quiet:
            print("\n" + "=" * 70)
            print("✅ 初始化完成")
            print("=" * 70)

    def _extract_all_date_features(self):
        date_cols = [col for col, typ in self.variable_types.items() if typ == 'datetime']
        for date_col in date_cols:
            self._extract_date_features(date_col)

    def _extract_date_features(self, date_col):
        if date_col not in self.data.columns:
            return

        features_to_extract = DATE_FEATURES_CONFIG.get(self.date_features_level, ["year", "month", "quarter"])
        if not features_to_extract:
            return

        dates = pd.to_datetime(self.data[date_col], errors='coerce')
        if dates.isna().all():
            return

        if not hasattr(self, 'date_features'):
            self.date_features = {}

        features = {}
        new_columns = []

        if 'year' in features_to_extract:
            col_name = f'{date_col}_year'
            features[col_name] = dates.dt.year
            new_columns.append(col_name)
            # 固定为数值型分类变量
            self.variable_types[col_name] = 'ordinal'
            self.type_reasons[col_name] = '日期派生列（年份）'

        if 'month' in features_to_extract:
            col_name = f'{date_col}_month'
            features[col_name] = dates.dt.month.astype('category')
            new_columns.append(col_name)
            # 固定为分类变量
            self.variable_types[col_name] = 'categorical'
            self.type_reasons[col_name] = '日期派生列（月份）'

        if 'quarter' in features_to_extract:
            col_name = f'{date_col}_quarter'
            features[col_name] = dates.dt.quarter.astype('category')
            new_columns.append(col_name)
            # 固定为分类变量
            self.variable_types[col_name] = 'categorical'
            self.type_reasons[col_name] = '日期派生列（季度）'

        if 'week' in features_to_extract:
            col_name = f'{date_col}_week'
            features[col_name] = dates.dt.isocalendar().week.astype('category')
            new_columns.append(col_name)
            # 固定为数值型分类变量
            self.variable_types[col_name] = 'ordinal'
            self.type_reasons[col_name] = '日期派生列（周数）'

        if 'weekday' in features_to_extract:
            col_name = f'{date_col}_weekday'
            features[col_name] = dates.dt.dayofweek.astype('category')
            new_columns.append(col_name)
            # 固定为分类变量
            self.variable_types[col_name] = 'categorical'
            self.type_reasons[col_name] = '日期派生列（星期几）'

        if 'day' in features_to_extract:
            col_name = f'{date_col}_day'
            features[col_name] = dates.dt.day
            new_columns.append(col_name)
            # 固定为数值型分类变量
            self.variable_types[col_name] = 'ordinal'
            self.type_reasons[col_name] = '日期派生列（日）'

        if 'is_weekend' in features_to_extract:
            col_name = f'{date_col}_is_weekend'
            features[col_name] = (dates.dt.dayofweek >= 5).astype('category')
            new_columns.append(col_name)
            # 固定为分类变量
            self.variable_types[col_name] = 'categorical'
            self.type_reasons[col_name] = '日期派生列（是否周末）'

        for name, values in features.items():
            if name not in self.data.columns:
                self.data[name] = values
                if not self.quiet:
                    print(f"  ✅ 自动添加特征列: {name}")

        self.date_features[date_col] = {'columns': new_columns, 'original': date_col}
        self.date_derived_columns.update(new_columns)
        for col in new_columns:
            self.date_column_mapping[col] = date_col

    def _get_type_description(self, var_type):
        """获取类型描述 - 调用 BaseAnalyzer 静态方法"""
        return BaseAnalyzer.get_type_description(var_type)

    def _check_normality(self, x):
        """检查正态性 - 调用 BaseAnalyzer 静态方法"""
        return BaseAnalyzer.check_normality(x)

    # ================== 图表base64方法 ==================

    def get_plot_base64(self, plot_type, col=None, **kwargs):
        """获取指定图表的base64编码"""
        if col is None or col not in self.data.columns:
            return None

        buf = io.BytesIO()

        try:
            if plot_type == 'categorical' and col:
                series = self.data[col].dropna()
                if len(series) > 0:
                    plot_categorical(series, col, buf)
                else:
                    return None
            elif plot_type == 'continuous' and col:
                series = self.data[col].dropna()
                if len(series) > 0:
                    plot_continuous(series, col, buf)
                else:
                    return None
            elif plot_type == 'timeseries' and col:
                date_cols = [c for c, typ in self.variable_types.items() if typ == 'datetime' and c in self.data.columns]
                if date_cols:
                    date_col = date_cols[0]
                    ts_data = self.data.groupby(date_col)[col].mean().reset_index()
                    ts_data = ts_data.dropna()
                    if len(ts_data) >= 3:
                        ts_data = ts_data.set_index(date_col)
                        series = ts_data[col]
                        plot_timeseries(series, col, buf)
                    else:
                        return None
            elif plot_type == 'correlation':
                numeric_vars = [c for c, typ in self.variable_types.items() if typ == 'continuous' and c in self.data.columns]
                if len(numeric_vars) >= 2:
                    plot_correlation(self.data, numeric_vars, buf)
                else:
                    return None
            else:
                return None

            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            return img_base64
        except Exception as e:
            if not self.quiet:
                print(f"⚠️ 生成图表失败: {e}")
            return None

    def get_numeric_correlation_base64(self):
        """获取数值变量相关性热力图base64"""
        numeric_vars = [col for col, typ in self.variable_types.items() if typ == 'continuous' and col in self.data.columns]
        if len(numeric_vars) < 2:
            return None

        buf = io.BytesIO()
        plot_correlation(self.data, numeric_vars, buf)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_base64

    def get_categorical_correlation_base64(self):
        """获取分类变量关联热力图base64"""
        categorical_vars = [col for col, typ in self.variable_types.items()
                            if typ in ['categorical', 'categorical_numeric', 'ordinal'] and col in self.data.columns]
        if len(categorical_vars) < 2:
            return None

        buf = io.BytesIO()
        plot_categorical_correlation(self.data, categorical_vars, buf)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_base64

    def get_numeric_categorical_eta_base64(self):
        """获取数值-分类变量关联热力图base64"""
        numeric_vars = [col for col, typ in self.variable_types.items() if typ == 'continuous' and col in self.data.columns]
        categorical_vars = [col for col, typ in self.variable_types.items()
                            if typ in ['categorical', 'categorical_numeric', 'ordinal'] and col in self.data.columns]
        if not numeric_vars or not categorical_vars:
            return None

        buf = io.BytesIO()
        plot_numeric_categorical_eta(self.data, numeric_vars, categorical_vars, buf)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_base64

    # ================== 辅助方法 ==================

    def _get_high_correlations(self, numeric_vars, threshold=0.7):
        """获取强相关对 - 调用 BaseAnalyzer 静态方法"""
        if not numeric_vars:
            return []
        return BaseAnalyzer.get_high_correlations(self.data, numeric_vars, threshold)

    def _get_skewed_vars(self, threshold=2):
        """获取偏态变量 - 调用 BaseAnalyzer 静态方法"""
        return BaseAnalyzer.get_skewed_vars(self.data, self.variable_types, threshold)

    def _get_imbalanced_vars(self, threshold=0.8):
        """获取不平衡分类变量 - 调用 BaseAnalyzer 静态方法"""
        return BaseAnalyzer.get_imbalanced_vars(self.data, self.variable_types, threshold)

    # ================== 对外接口 ==================

    def identify_time_series_grouping(self, date_col=None):
        return self.timeseries_analyzer.identify_time_series_grouping(date_col)

    def auto_time_series_analysis(self, max_numeric=10, group_by='auto'):
        self.timeseries_analyzer.auto_time_series_analysis(max_numeric, group_by)
        self.time_series_diagnostics = self.timeseries_analyzer.time_series_diagnostics
        # 同步更新 recommendation_analyzer
        self.recommendation_analyzer.time_series_diagnostics = self.time_series_diagnostics

    def auto_analyze_relationships(self):
        self.relationship_analyzer.auto_analyze_relationships()

    def recommend_scenarios(self):
        self.recommendation_analyzer.recommend_scenarios()

    def auto_describe(self):
        """自动统计描述报告"""
        print("\n" + "=" * 70)
        print("📊 自动统计描述报告")
        if self.source_table_name:
            print(f"  源表: {self.source_table_name}")
        print("=" * 70)

        identifier_cols = [col for col, typ in self.variable_types.items() if typ == 'identifier']
        if len(identifier_cols) == len(self.data.columns):
            print("\n⚠️ 此表只包含标识符列（如ID、编码等），没有可用于分析的业务字段。")
            print("   建议关联其他表进行有意义的分析。")
            return

        analyzable_cols = [col for col, typ in self.variable_types.items()
                           if typ in ['continuous', 'categorical', 'categorical_numeric', 'ordinal', 'datetime']]
        if not analyzable_cols:
            print("\n⚠️ 此表没有可用于分析的数值变量、分类变量或日期变量。")
            print("   建议检查数据或关联其他表。")
            return

        base = BaseAnalyzer(self.data, self.variable_types, self.type_reasons, self.quiet)

        for col in list(self.data.columns)[:20]:
            if col == self.target_col:
                continue

            summary = base._get_variable_summary(col)
            var_type = summary['type']

            print(f"\n【变量】{col}")
            print(f"类型: {self._get_type_description(var_type)}")
            print(f"样本量: {summary['n']} (缺失: {summary['n_missing']}, {summary['missing_pct']:.1f}%)")

            if var_type == 'identifier':
                n_unique = summary.get('n_unique', 'N/A')
                min_val = summary.get('min', 'N/A')
                max_val = summary.get('max', 'N/A')
                print(f"唯一值数: {n_unique}")
                print(f"范围: [{min_val}, {max_val}]")
                print(f"⚠️ 标识符列，统计指标无实际意义")

            elif var_type == 'datetime':
                min_date = summary.get('min_date', 'N/A')
                max_date = summary.get('max_date', 'N/A')
                date_range = summary.get('date_range', 'N/A')
                n_unique = summary.get('n_unique', 'N/A')
                print(f"日期范围: {min_date} 到 {max_date}")
                print(f"时间跨度: {date_range}")
                print(f"唯一日期数: {n_unique}")

            elif var_type in ['categorical', 'categorical_numeric', 'ordinal']:
                n_unique = summary.get('n_unique', 'N/A')
                mode_val = summary.get('mode', 'N/A')
                mode_freq = summary.get('mode_freq', 0)
                mode_pct = summary.get('mode_pct', 0)
                print(f"类别数: {n_unique}")
                print(f"众数: {mode_val} (频数: {mode_freq}, {mode_pct:.1f}%)")

                if isinstance(n_unique, int) and n_unique <= 10:
                    print("\n类别分布:")
                    value_counts = summary.get('value_counts', {})
                    for val, count in list(value_counts.items())[:5]:
                        if isinstance(val, (int, float)):
                            pct = count / summary['n'] * 100 if summary['n'] > 0 else 0
                            print(f"  {val}: {count} ({pct:.1f}%)")
                elif isinstance(n_unique, int) and n_unique > 10:
                    print("\n(前5个类别)")
                    value_counts = summary.get('value_counts', {})
                    for val, count in list(value_counts.items())[:5]:
                        if isinstance(val, (int, float)):
                            pct = count / summary['n'] * 100 if summary['n'] > 0 else 0
                            print(f"  {val}: {count} ({pct:.1f}%)")

                try:
                    plot_categorical(self.data[col].dropna(), col)
                except Exception as e:
                    if not self.quiet:
                        print(f"  ⚠️ 图表生成失败: {e}")

            elif var_type == 'continuous':
                mean_val = summary.get('mean', 0)
                std_val = summary.get('std', 0)
                median_val = summary.get('median', 0)
                q1_val = summary.get('q1', 0)
                q3_val = summary.get('q3', 0)
                min_val = summary.get('min', 0)
                max_val = summary.get('max', 0)
                skew_val = summary.get('skew', 0)
                kurtosis_val = summary.get('kurtosis', 0)
                is_normal = summary.get('is_normal', False)
                normality_p = summary.get('normality_p', 1.0)

                print(f"均值 ± 标准差: {mean_val:.2f} ± {std_val:.2f}")
                print(f"中位数 [Q1, Q3]: {median_val:.2f} [{q1_val:.2f}, {q3_val:.2f}]")
                print(f"范围: [{min_val:.2f}, {max_val:.2f}]")
                print(f"偏度: {skew_val:.2f}, 峰度: {kurtosis_val:.2f}")

                norm_status = "✅ 正态分布" if is_normal else "⚠️ 非正态分布"
                print(f"正态性: {norm_status} (p={normality_p:.4f})")

                if is_normal:
                    print(f"推荐描述: {mean_val:.2f} ± {std_val:.2f}")
                else:
                    print(f"推荐描述: {median_val:.2f} [{q1_val:.2f}, {q3_val:.2f}]")

                try:
                    plot_continuous(self.data[col].dropna(), col)
                except Exception as e:
                    if not self.quiet:
                        print(f"  ⚠️ 图表生成失败: {e}")

            else:
                n_unique = summary.get('n_unique', 'N/A')
                print(f"唯一值数: {n_unique}")

    def generate_full_report(self, show_outlier_details=False):
        """生成完整报告"""
        print("\n" + "=" * 70)
        print("📋 完整自动分析报告")
        print("=" * 70)

        print("\n【数据质量摘要】")
        base = BaseAnalyzer(self.data, self.variable_types, self.type_reasons, self.quiet)
        base.quality_report = self.quality_report
        base._print_quality_summary()

        print("\n【1. 变量类型识别】")
        for col, var_type in list(self.variable_types.items())[:20]:
            type_desc = self._get_type_description(var_type)
            reason = self.type_reasons.get(col, '')
            print(f"  {col}: {type_desc} ({reason})")

        if self.cleaning_suggestions:
            print("\n【清洗建议】")
            for suggestion in self.cleaning_suggestions[:5]:
                print(f"  {suggestion}")

        self.auto_describe()
        self.auto_time_series_analysis()
        self.auto_analyze_relationships()
        self.recommend_scenarios()

        try:
            plt.show(block=False)
            plt.pause(0.5)
        except Exception as e:
            if not self.quiet:
                print(f"⚠️ 图表显示失败: {e}")

    def to_json(self, output_file=None, indent=2, ensure_ascii=False):
        import json
        from datetime import datetime

        numeric_vars = [col for col, typ in self.variable_types.items() if typ == 'continuous']
        categorical_vars = [col for col, typ in self.variable_types.items()
                            if typ in ['categorical', 'categorical_numeric', 'ordinal']]
        date_cols = [col for col, typ in self.variable_types.items() if typ == 'datetime']

        base = BaseAnalyzer(self.data, self.variable_types, self.type_reasons, self.quiet)
        variable_summaries = {}
        for col in self.data.columns:
            summary = base._get_variable_summary(col)
            var_type = summary['type']

            if var_type == 'continuous':
                variable_summaries[col] = {
                    'type': var_type,
                    'type_desc': self._get_type_description(var_type),
                    'count': summary.get('n', 0),
                    'missing': summary.get('n_missing', 0),
                    'missing_pct': summary.get('missing_pct', 0),
                    'mean': summary.get('mean', 0),
                    'std': summary.get('std', 0),
                    'median': summary.get('median', 0),
                    'q1': summary.get('q1', 0),
                    'q3': summary.get('q3', 0),
                    'min': summary.get('min', 0),
                    'max': summary.get('max', 0),
                    'skew': summary.get('skew', 0),
                    'kurtosis': summary.get('kurtosis', 0),
                    'is_normal': summary.get('is_normal', False)
                }
            elif var_type in ['categorical', 'categorical_numeric', 'ordinal']:
                top_categories = {}
                for val, count in list(summary.get('value_counts', {}).items())[:10]:
                    top_categories[str(val)] = count
                variable_summaries[col] = {
                    'type': var_type,
                    'type_desc': self._get_type_description(var_type),
                    'count': summary.get('n', 0),
                    'missing': summary.get('n_missing', 0),
                    'missing_pct': summary.get('missing_pct', 0),
                    'n_unique': summary.get('n_unique', 0),
                    'mode': str(summary.get('mode', '')) if summary.get('mode') else None,
                    'mode_freq': summary.get('mode_freq', 0),
                    'mode_pct': summary.get('mode_pct', 0),
                    'top_categories': top_categories
                }
            elif var_type == 'datetime':
                variable_summaries[col] = {
                    'type': var_type,
                    'type_desc': self._get_type_description(var_type),
                    'count': summary.get('n', 0),
                    'missing': summary.get('n_missing', 0),
                    'missing_pct': summary.get('missing_pct', 0),
                    'min_date': str(summary.get('min_date', '')) if summary.get('min_date') else None,
                    'max_date': str(summary.get('max_date', '')) if summary.get('max_date') else None,
                    'date_range_days': summary.get('date_range').days if hasattr(summary.get('date_range'),
                                                                                 'days') else None,
                    'n_unique': summary.get('n_unique', 0)
                }
            else:
                variable_summaries[col] = {
                    'type': var_type,
                    'type_desc': self._get_type_description(var_type),
                    'count': summary.get('n', 0),
                    'missing': summary.get('n_missing', 0),
                    'missing_pct': summary.get('missing_pct', 0),
                    'n_unique': summary.get('n_unique', 0)
                }

        correlation_matrix = None
        if len(numeric_vars) >= 2:
            valid_numeric = [col for col in numeric_vars if col in self.data.columns and self.data[col].notna().any()]
            if len(valid_numeric) >= 2:
                correlation_matrix = self.data[valid_numeric].corr().round(4).to_dict()

        high_correlations = self._get_high_correlations(numeric_vars, threshold=0.7) if numeric_vars else []
        skewed_vars = self._get_skewed_vars(threshold=2)
        imbalanced_vars = self._get_imbalanced_vars(threshold=0.8)

        time_series_forecastable = False
        if self.time_series_diagnostics:
            for diag in self.time_series_diagnostics.values():
                if diag.get('has_autocorrelation'):
                    time_series_forecastable = True
                    break

        normal_numeric = []
        nonnormal_numeric = []
        for col in numeric_vars:
            if col in self.data.columns:
                is_normal, _, _ = self._check_normality(self.data[col].dropna())
                if is_normal:
                    normal_numeric.append(col)
                else:
                    nonnormal_numeric.append(col)

        model_recommendations = self.recommendation_analyzer._get_model_recommendations(
            numeric_vars, categorical_vars, date_cols
        )

        result = {
            'analysis_time': datetime.now().isoformat(),
            'source_table': self.source_table_name,
            'data_shape': {'rows': len(self.data), 'columns': len(self.data.columns)},
            'column_names': list(self.data.columns),
            'variable_types': {
                col: {'type': var_type, 'type_desc': self._get_type_description(var_type)}
                for col, var_type in self.variable_types.items()
            },
            'variable_summaries': variable_summaries,
            'quality_report': {
                'missing': self.quality_report.get('missing', []),
                'outliers': {
                    col: {'count': info.get('count', 0), 'percent': info.get('percent', 0),
                          'lower_bound': info.get('lower_bound'), 'upper_bound': info.get('upper_bound')}
                    for col, info in self.quality_report.get('outliers', {}).items()
                },
                'duplicates': self.quality_report.get('duplicates', {}),
                'inconsistent_types': self.quality_report.get('inconsistent_types', {}),
                'invalid_values': self.quality_report.get('invalid_values', {})
            },
            'cleaning_suggestions': self.cleaning_suggestions,
            'correlations': {
                'matrix': correlation_matrix,
                'high_correlations': high_correlations
            },
            'distribution_insights': {
                'skewed_variables': skewed_vars,
                'imbalanced_categoricals': imbalanced_vars
            },
            'time_series_diagnostics': self.time_series_diagnostics,
            'time_series_forecastable': time_series_forecastable,
            'model_recommendations': model_recommendations
        }

        json_str = json.dumps(result, indent=indent, ensure_ascii=ensure_ascii, default=str)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"✅ 分析结果已保存为JSON: {output_file}")
            return output_file
        return json_str