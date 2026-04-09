"""
主分析器类 - 整合所有模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, Optional, List

from autostat.loader import DataLoader
from autostat.checker import ConditionChecker
from autostat.core.base import BaseAnalyzer
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
                 parse_dates=True, date_columns=None, date_features_level="basic"):
        """初始化分析器"""
        if isinstance(data, str):
            print(f"📁 加载文件: {data}")
            self.raw_data = DataLoader.load_from_file(data, parse_dates=parse_dates, date_columns=date_columns).copy()
        elif isinstance(data, pd.DataFrame):
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

        if not quiet:
            print("\n【阶段1】快速初筛...")
        base = BaseAnalyzer(self.data, quiet=quiet)
        base._quick_pre_screen()
        self.data = base.data
        if not quiet:
            print("  ✅ 完成")

        if not quiet:
            print("\n【阶段2】首次类型识别...")
        if predefined_types:
            self.variable_types = predefined_types
            self.type_reasons = {col: f'从源表 {source_table_name} 继承' for col in predefined_types.keys()}
        else:
            base.variable_types = self.variable_types
            base.type_reasons = self.type_reasons
            base._infer_variable_types()
            self.variable_types = base.variable_types
            self.type_reasons = base.type_reasons
        if not quiet:
            print("  ✅ 基础字段识别完成")

        if not quiet:
            print("\n【阶段3】日期特征提取...")
        self._extract_all_date_features()
        if not quiet:
            print("  ✅ 完成")

        if self._has_new_date_features():
            if not quiet:
                print("\n【阶段4】二次类型识别...")
            base.variable_types = self.variable_types
            base.type_reasons = self.type_reasons
            base._infer_variable_types()
            self.variable_types = base.variable_types
            self.type_reasons = base.type_reasons
            if not quiet:
                print("  ✅ 日期特征列识别完成")

        if not quiet:
            print("\n【阶段5】深度数据质量体检...")
        base.variable_types = self.variable_types
        base.quality_report = base._comprehensive_quality_check()
        self.quality_report = base.quality_report
        if not quiet:
            base._print_quality_summary()

        if not quiet:
            print("\n【阶段6】生成清洗建议...")
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

        if not quiet:
            print("\n" + "=" * 70)
            print("✅ 初始化完成")
            print("=" * 70)

    def _has_new_date_features(self):
        if not hasattr(self, 'date_features'):
            return False
        for date_col, info in self.date_features.items():
            if isinstance(info, dict) and 'columns' in info:
                for col in info['columns']:
                    if col not in self.variable_types:
                        return True
        return False

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

        if 'month' in features_to_extract:
            col_name = f'{date_col}_month'
            features[col_name] = dates.dt.month.astype('category')
            new_columns.append(col_name)

        if 'quarter' in features_to_extract:
            col_name = f'{date_col}_quarter'
            features[col_name] = dates.dt.quarter.astype('category')
            new_columns.append(col_name)

        if 'week' in features_to_extract:
            col_name = f'{date_col}_week'
            features[col_name] = dates.dt.isocalendar().week.astype('category')
            new_columns.append(col_name)

        if 'weekday' in features_to_extract:
            col_name = f'{date_col}_weekday'
            features[col_name] = dates.dt.dayofweek.astype('category')
            new_columns.append(col_name)

        if 'day' in features_to_extract:
            col_name = f'{date_col}_day'
            features[col_name] = dates.dt.day
            new_columns.append(col_name)

        if 'is_weekend' in features_to_extract:
            col_name = f'{date_col}_is_weekend'
            features[col_name] = (dates.dt.dayofweek >= 5).astype('category')
            new_columns.append(col_name)

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
        type_map = {
            'categorical': '分类变量', 'categorical_numeric': '数值型分类变量',
            'ordinal': '有序分类变量', 'continuous': '连续变量',
            'text': '文本变量', 'identifier': '标识符列',
            'datetime': '日期时间变量', 'other': '其他', 'empty': '空变量'
        }
        return type_map.get(var_type, var_type)

    def _check_normality(self, x):
        from scipy.stats import shapiro, normaltest
        x = x.dropna()
        if len(x) < 8 or len(x) > 5000:
            return False, 1.0, {'skew': 0, 'kurtosis': 0}
        try:
            _, p_shapiro = shapiro(x)
        except:
            p_shapiro = 0
        try:
            _, p_normaltest = normaltest(x)
        except:
            p_normaltest = 0
        skewness = abs(x.skew())
        kurtosis = abs(x.kurtosis())
        p_value = max(p_shapiro, p_normaltest)
        is_normal = (p_value > 0.05) and (skewness < 2) and (kurtosis < 7)
        return is_normal, p_value, {'skew': skewness, 'kurtosis': kurtosis}

    # ================== 图表base64方法 ==================

    def get_plot_base64(self, plot_type, col=None, **kwargs):
        """获取指定图表的base64编码"""
        buf = io.BytesIO()

        try:
            if plot_type == 'categorical' and col:
                plot_categorical(self.data[col].dropna(), col, buf)
            elif plot_type == 'continuous' and col:
                plot_continuous(self.data[col].dropna(), col, buf)
            elif plot_type == 'timeseries' and col:
                date_cols = [c for c, typ in self.variable_types.items() if typ == 'datetime']
                if date_cols:
                    date_col = date_cols[0]
                    ts_data = self.data.groupby(date_col)[col].mean().reset_index()
                    ts_data = ts_data.dropna()
                    if len(ts_data) >= 3:
                        ts_data = ts_data.set_index(date_col)
                        series = ts_data[col]
                        plot_timeseries(series, col, buf)
            elif plot_type == 'correlation':
                numeric_vars = [c for c, typ in self.variable_types.items() if typ == 'continuous']
                plot_correlation(self.data, numeric_vars, buf)
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
        numeric_vars = [col for col, typ in self.variable_types.items() if typ == 'continuous']
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
                            if typ in ['categorical', 'categorical_numeric', 'ordinal']]
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
        numeric_vars = [col for col, typ in self.variable_types.items() if typ == 'continuous']
        categorical_vars = [col for col, typ in self.variable_types.items()
                            if typ in ['categorical', 'categorical_numeric', 'ordinal']]
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
        correlations = []
        if len(numeric_vars) >= 2:
            corr_data = self.data[numeric_vars].corr()
            for i in range(len(corr_data.columns)):
                for j in range(i + 1, len(corr_data.columns)):
                    val = corr_data.iloc[i, j]
                    if abs(val) >= threshold:
                        correlations.append({
                            'var1': corr_data.columns[i],
                            'var2': corr_data.columns[j],
                            'value': round(val, 3)
                        })
        return sorted(correlations, key=lambda x: abs(x['value']), reverse=True)

    def _get_skewed_vars(self, threshold=2):
        skewed = []
        for col, typ in self.variable_types.items():
            if typ == 'continuous':
                data = self.data[col].dropna()
                if len(data) > 0:
                    skew = data.skew()
                    if abs(skew) >= threshold:
                        skewed.append({'name': col, 'skew': round(skew, 2)})
        return sorted(skewed, key=lambda x: abs(x['skew']), reverse=True)

    def _get_imbalanced_vars(self, threshold=0.8):
        imbalanced = []
        for col, typ in self.variable_types.items():
            if typ in ['categorical', 'categorical_numeric', 'ordinal']:
                vc = self.data[col].value_counts(normalize=True)
                if len(vc) > 0 and vc.max() >= threshold:
                    imbalanced.append({
                        'name': col,
                        'top_category': str(vc.index[0]),
                        'top_pct': round(vc.max() * 100, 1)
                    })
        return imbalanced

    # ================== 对外接口 ==================

    def identify_time_series_grouping(self, date_col=None):
        return self.timeseries_analyzer.identify_time_series_grouping(date_col)

    def auto_time_series_analysis(self, max_numeric=10, group_by='auto'):
        self.timeseries_analyzer.auto_time_series_analysis(max_numeric, group_by)
        self.time_series_diagnostics = self.timeseries_analyzer.time_series_diagnostics

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
                print(f"唯一值数: {summary['n_unique']}")
                print(f"范围: [{summary['min']}, {summary['max']}]")
                print(f"⚠️ 标识符列，统计指标无实际意义")

            elif var_type == 'datetime':
                print(f"日期范围: {summary['min_date']} 到 {summary['max_date']}")
                print(f"时间跨度: {summary['date_range']}")
                print(f"唯一日期数: {summary['n_unique']}")

            elif var_type in ['categorical', 'categorical_numeric', 'ordinal']:
                print(f"类别数: {summary['n_unique']}")
                print(f"众数: {summary['mode']} (频数: {summary['mode_freq']}, {summary['mode_pct']:.1f}%)")

                if summary['n_unique'] <= 10:
                    print("\n类别分布:")
                    for val, count in summary['value_counts'].head().items():
                        pct = count / summary['n'] * 100
                        print(f"  {val}: {count} ({pct:.1f}%)")
                else:
                    print("\n(前5个类别)")
                    for val, count in summary['value_counts'].head().items():
                        pct = count / summary['n'] * 100
                        print(f"  {val}: {count} ({pct:.1f}%)")

                plot_categorical(self.data[col].dropna(), col)

            elif var_type == 'continuous':
                print(f"均值 ± 标准差: {summary['mean']:.2f} ± {summary['std']:.2f}")
                print(f"中位数 [Q1, Q3]: {summary['median']:.2f} [{summary['q1']:.2f}, {summary['q3']:.2f}]")
                print(f"范围: [{summary['min']:.2f}, {summary['max']:.2f}]")
                print(f"偏度: {summary['skew']:.2f}, 峰度: {summary['kurtosis']:.2f}")

                norm_status = "✅ 正态分布" if summary['is_normal'] else "⚠️ 非正态分布"
                print(f"正态性: {norm_status} (p={summary['normality_p']:.4f})")

                if summary['is_normal']:
                    print(f"推荐描述: {summary['mean']:.2f} ± {summary['std']:.2f}")
                else:
                    print(f"推荐描述: {summary['median']:.2f} [{summary['q1']:.2f}, {summary['q3']:.2f}]")

                plot_continuous(self.data[col].dropna(), col)

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

        plt.show(block=False)
        plt.pause(0.5)

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
                    'count': summary['n'],
                    'missing': summary['n_missing'],
                    'missing_pct': summary['missing_pct'],
                    'mean': summary['mean'],
                    'std': summary['std'],
                    'median': summary['median'],
                    'q1': summary['q1'],
                    'q3': summary['q3'],
                    'min': summary['min'],
                    'max': summary['max'],
                    'skew': summary['skew'],
                    'kurtosis': summary['kurtosis'],
                    'is_normal': summary['is_normal']
                }
            elif var_type in ['categorical', 'categorical_numeric', 'ordinal']:
                top_categories = {}
                for val, count in list(summary.get('value_counts', {}).items())[:10]:
                    top_categories[str(val)] = count
                variable_summaries[col] = {
                    'type': var_type,
                    'type_desc': self._get_type_description(var_type),
                    'count': summary['n'],
                    'missing': summary['n_missing'],
                    'missing_pct': summary['missing_pct'],
                    'n_unique': summary['n_unique'],
                    'mode': str(summary['mode']) if summary['mode'] else None,
                    'mode_freq': summary['mode_freq'],
                    'mode_pct': summary['mode_pct'],
                    'top_categories': top_categories
                }
            elif var_type == 'datetime':
                variable_summaries[col] = {
                    'type': var_type,
                    'type_desc': self._get_type_description(var_type),
                    'count': summary['n'],
                    'missing': summary['n_missing'],
                    'missing_pct': summary['missing_pct'],
                    'min_date': str(summary['min_date']),
                    'max_date': str(summary['max_date']),
                    'date_range_days': summary['date_range'].days if hasattr(summary['date_range'], 'days') else None,
                    'n_unique': summary['n_unique']
                }
            else:
                variable_summaries[col] = {
                    'type': var_type,
                    'type_desc': self._get_type_description(var_type),
                    'count': summary['n'],
                    'missing': summary['n_missing'],
                    'missing_pct': summary['missing_pct']
                }

        correlation_matrix = None
        if len(numeric_vars) >= 2:
            correlation_matrix = self.data[numeric_vars].corr().round(4).to_dict()

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