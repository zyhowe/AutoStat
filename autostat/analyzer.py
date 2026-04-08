"""
智能统计分析器模块
自动识别数据类型并选择方法
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, normaltest, pearsonr, spearmanr
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
from datetime import datetime
from typing import Dict, List, Optional

from autostat.loader import DataLoader
from autostat.checker import ConditionChecker

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DATE_FEATURES_CONFIG = {
    "none": [],
    "basic": ["year", "month", "quarter"],
    "full": ["year", "month", "quarter", "week", "weekday", "day", "is_weekend"]
}


class AutoStatisticalAnalyzer:
    """智能统计分析器"""

    def __init__(self, data, source_table_name=None, predefined_types=None,
                 auto_clean=False, quiet=False, parse_dates=True,
                 date_columns=None, date_features_level="basic"):
        """
        参数:
        - data: DataFrame 或 文件路径
        - source_table_name: 源表名
        - predefined_types: 预定义变量类型
        - auto_clean: 是否自动清洗
        - quiet: 是否静默模式
        - parse_dates: 是否解析日期
        - date_columns: 日期列名列表
        - date_features_level: 日期派生级别
        """
        # 加载数据
        if isinstance(data, str):
            print(f"📁 加载文件: {data}")
            self.raw_data = DataLoader.load_from_file(data, parse_dates=parse_dates, date_columns=date_columns).copy()
        elif isinstance(data, pd.DataFrame):
            self.raw_data = data.copy()
        else:
            raise ValueError("data 参数必须是 DataFrame 或文件路径字符串")

        self.data = self.raw_data.copy()
        self.target_col = None
        self.source_table_name = source_table_name
        self.results = {}
        self.date_features = {}
        self.date_original_columns = set()
        self.date_derived_columns = set()
        self.date_column_mapping = {}
        self.type_inference_warnings = {}
        self.quality_issues = {}
        self.auto_clean = auto_clean
        self.quiet = quiet
        self.time_series_diagnostics = {}
        self.time_series_grouping = {}
        self.date_features_level = date_features_level

        if not quiet:
            print("\n" + "=" * 70)
            print("🔍 启动智能分析流程")
            print("=" * 70)

        # 快速初筛
        self._quick_pre_screen()

        # 类型识别
        if predefined_types:
            self.variable_types = predefined_types
            self.type_reasons = {col: f'从源表 {source_table_name} 继承' for col in predefined_types.keys()}
        else:
            self.variable_types = {}
            self.type_reasons = {}
            self._infer_variable_types()

        # 日期特征提取
        self._extract_all_date_features()

        if self._has_new_date_features():
            self._infer_variable_types()

        if not quiet:
            self._print_type_summary()

        # 质量检查
        self.quality_report = self._comprehensive_quality_check()
        if not quiet:
            self._print_quality_summary()

        # 清洗建议
        self.cleaning_suggestions = self._generate_cleaning_suggestions()
        if not quiet:
            self._print_cleaning_suggestions()

        if auto_clean:
            self._auto_clean()

        self.condition_checker = ConditionChecker(
            self.data, self.variable_types,
            self.date_derived_columns, self.date_column_mapping
        )

        if not quiet:
            print("\n✅ 初始化完成")

    def _quick_pre_screen(self):
        """快速初筛"""
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                try:
                    if len(self.data[col]) > 0:
                        sample = self.data[col].iloc[0]
                        if isinstance(sample, bytes):
                            self.data[col] = self.data[col].apply(
                                lambda x: x.decode('gbk', errors='ignore') if isinstance(x, bytes) else x
                            )
                    empty_mask = self.data[col].astype(str).str.strip() == ''
                    if empty_mask.any():
                        self.data.loc[empty_mask, col] = np.nan
                except:
                    pass

    def _has_new_date_features(self):
        """检查是否有新的日期特征列"""
        if not hasattr(self, 'date_features'):
            return False
        for date_col, info in self.date_features.items():
            if isinstance(info, dict) and 'columns' in info:
                for col in info['columns']:
                    if col not in self.variable_types:
                        return True
        return False

    def _infer_variable_types(self):
        """自动推断变量类型"""
        for col in self.data.columns:
            if col in self.variable_types:
                continue

            values = self.data[col].dropna()
            if len(values) == 0:
                self.variable_types[col] = 'empty'
                self.type_reasons[col] = '空变量'
                continue

            n_unique = values.nunique()
            unique_ratio = n_unique / len(values) if len(values) > 0 else 0

            # 日期类型检测
            if pd.api.types.is_string_dtype(values) or pd.api.types.is_object_dtype(values):
                try:
                    converted = pd.to_datetime(values.head(100), errors='coerce')
                    if converted.notna().sum() > 80:
                        self.variable_types[col] = 'datetime'
                        self.type_reasons[col] = '日期时间类型'
                        self.date_original_columns.add(col)
                        self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                        continue
                except:
                    pass

            if pd.api.types.is_datetime64_any_dtype(values):
                self.variable_types[col] = 'datetime'
                self.type_reasons[col] = '日期时间类型'
                self.date_original_columns.add(col)
                continue

            # 数值类型
            if pd.api.types.is_numeric_dtype(values):
                is_integer = pd.api.types.is_integer_dtype(values)

                if is_integer and unique_ratio > 0.95:
                    self.variable_types[col] = 'identifier'
                    self.type_reasons[col] = f'标识符列 (唯一性{unique_ratio:.1%})'
                elif n_unique <= 5 or (unique_ratio < 0.05 and n_unique < 20):
                    self.variable_types[col] = 'categorical_numeric'
                    self.type_reasons[col] = f'数值型分类变量，{n_unique}个取值'
                elif is_integer and n_unique < 25:
                    self.variable_types[col] = 'ordinal'
                    self.type_reasons[col] = f'有序分类变量，{n_unique}个等级'
                else:
                    self.variable_types[col] = 'continuous'
                    is_normal, _, norm_stats = self._check_normality(values)
                    self.type_reasons[col] = f'连续变量，{"正态" if is_normal else "非正态"}分布'
            elif pd.api.types.is_string_dtype(values) or pd.api.types.is_categorical_dtype(values):
                if n_unique <= 20:
                    self.variable_types[col] = 'categorical'
                    self.type_reasons[col] = f'分类变量，{n_unique}个类别'
                else:
                    self.variable_types[col] = 'text'
                    self.type_reasons[col] = f'文本变量'
            elif pd.api.types.is_bool_dtype(values):
                self.variable_types[col] = 'categorical'
                self.type_reasons[col] = '布尔型分类变量'
            else:
                self.variable_types[col] = 'other'
                self.type_reasons[col] = f'其他类型'

    def _check_normality(self, x):
        """检查正态性"""
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

    def _extract_all_date_features(self):
        """提取所有日期特征"""
        date_cols = [col for col, typ in self.variable_types.items() if typ == 'datetime']
        for date_col in date_cols:
            self._extract_date_features(date_col)

    def _extract_date_features(self, date_col):
        """提取单个日期特征"""
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

        self.date_features[date_col] = {
            'columns': new_columns,
            'original': date_col
        }

    def _get_type_description(self, var_type):
        """获取类型描述"""
        type_map = {
            'categorical': '分类变量',
            'categorical_numeric': '数值型分类变量',
            'ordinal': '有序分类变量',
            'continuous': '连续变量',
            'text': '文本变量',
            'identifier': '标识符列',
            'datetime': '日期时间变量',
            'other': '其他',
            'empty': '空变量'
        }
        return type_map.get(var_type, var_type)

    def _print_type_summary(self):
        """打印类型汇总"""
        if self.quiet:
            return
        type_counts = {}
        for typ in self.variable_types.values():
            type_counts[typ] = type_counts.get(typ, 0) + 1
        print("\n  类型汇总:")
        for typ, count in type_counts.items():
            print(f"    {self._get_type_description(typ)}: {count}列")

    def _comprehensive_quality_check(self) -> Dict:
        """综合质量检查"""
        report = {
            'missing': self._check_missing_values(),
            'outliers': self._check_outliers_by_type(),
            'duplicates': self._check_duplicates_by_key(),
            'inconsistent_types': self._check_type_consistency(),
            'invalid_values': self._check_invalid_by_type()
        }
        return report

    def _check_missing_values(self) -> List[Dict]:
        """检查缺失值"""
        missing_list = []
        for col in self.data.columns:
            missing_count = self.data[col].isna().sum()
            if missing_count > 0:
                missing_list.append({
                    'column': col,
                    'count': missing_count,
                    'percent': missing_count / len(self.data) * 100
                })
        return sorted(missing_list, key=lambda x: x['percent'], reverse=True)

    def _check_outliers_by_type(self) -> Dict:
        """检查异常值"""
        outlier_report = {}
        for col, typ in self.variable_types.items():
            if typ == 'continuous':
                data = self.data[col].dropna()
                if len(data) > 0:
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    outlier_mask = (data < lower) | (data > upper)
                    if outlier_mask.sum() > 0:
                        outlier_report[col] = {
                            'count': outlier_mask.sum(),
                            'percent': outlier_mask.sum() / len(data) * 100,
                            'lower_bound': lower,
                            'upper_bound': upper
                        }
        return outlier_report

    def _check_duplicates_by_key(self) -> Dict:
        """检查重复值"""
        id_cols = [col for col, typ in self.variable_types.items() if typ == 'identifier']
        if id_cols:
            duplicates = self.data.duplicated(subset=id_cols, keep=False)
            return {
                'count': duplicates.sum(),
                'percent': duplicates.sum() / len(self.data) * 100 if len(self.data) > 0 else 0,
                'based_on': id_cols
            }
        else:
            duplicates = self.data.duplicated(keep=False)
            return {
                'count': duplicates.sum(),
                'percent': duplicates.sum() / len(self.data) * 100 if len(self.data) > 0 else 0,
                'based_on': 'all_columns'
            }

    def _check_type_consistency(self) -> Dict:
        """检查类型一致性"""
        inconsistency = {}
        for col in self.data.columns:
            types = self.data[col].apply(type).unique()
            if len(types) > 1:
                non_nan_types = [t for t in types if t != float or not pd.isna(self.data[col]).all()]
                if len(non_nan_types) > 1:
                    inconsistency[col] = {'types': [str(t) for t in non_nan_types]}
        return inconsistency

    def _check_invalid_by_type(self) -> Dict:
        """检查无效值"""
        invalid = {}
        for col, typ in self.variable_types.items():
            data = self.data[col].dropna()
            if len(data) == 0:
                continue
            if typ == 'continuous' and pd.api.types.is_numeric_dtype(data):
                inf_mask = np.isinf(data)
                if inf_mask.any():
                    invalid[f'{col}_inf'] = {'type': '无限值', 'count': inf_mask.sum()}
            elif typ == 'datetime':
                try:
                    future = data > pd.Timestamp.now()
                    if future.any():
                        invalid[f'{col}_future'] = {'type': '未来日期', 'count': future.sum()}
                except:
                    pass
        return invalid

    def _generate_cleaning_suggestions(self) -> List[str]:
        """生成清洗建议"""
        suggestions = []
        for item in self.quality_report['missing']:
            col = item['column']
            pct = item['percent']
            if pct > 80:
                suggestions.append(f"⚠️ 严重: {col} 缺失率 {pct:.1f}%，建议删除该列")
            elif pct > 50:
                suggestions.append(f"⚠️ 较高: {col} 缺失率 {pct:.1f}%，建议评估是否保留")
            elif pct > 20:
                suggestions.append(f"📌 {col} 缺失率 {pct:.1f}%，建议填充")
        for col, info in self.quality_report['outliers'].items():
            if info['percent'] > 10:
                suggestions.append(f"⚠️ {col} 异常值比例 {info['percent']:.1f}%，建议检查")
        dup_info = self.quality_report['duplicates']
        if dup_info['count'] > 0:
            suggestions.append(f"📌 发现 {dup_info['count']} 条重复记录 ({dup_info['percent']:.1f}%)")
        return suggestions

    def _print_quality_summary(self):
        """打印质量摘要"""
        if self.quiet:
            return
        missing_count = len(self.quality_report['missing'])
        outlier_count = len(self.quality_report['outliers'])
        dup_count = self.quality_report['duplicates']['count']
        print(f"\n  📊 缺失值: {missing_count}列存在缺失")
        print(f"  📊 异常值: {outlier_count}列存在异常值")
        print(f"  📊 重复值: {dup_count}条")

    def _print_cleaning_suggestions(self):
        """打印清洗建议"""
        if self.quiet or not self.cleaning_suggestions:
            return
        print("\n  💡 清洗建议:")
        for i, suggestion in enumerate(self.cleaning_suggestions[:5], 1):
            print(f"    {i}. {suggestion}")

    def _auto_clean(self):
        """自动清洗"""
        if not self.quiet:
            print("\n  执行自动清洗...")
        cols_to_drop = []
        for item in self.quality_report['missing']:
            if item['percent'] > 80:
                cols_to_drop.append(item['column'])
        if cols_to_drop:
            self.data = self.data.drop(columns=cols_to_drop)
        dup_info = self.quality_report['duplicates']
        if dup_info['count'] > 0:
            if dup_info['based_on'] != 'all_columns':
                self.data = self.data.drop_duplicates(subset=dup_info['based_on'], keep='first')
            else:
                self.data = self.data.drop_duplicates(keep='first')
        self.quality_report = self._comprehensive_quality_check()

    def auto_describe(self):
        """自动描述统计"""
        print("\n" + "=" * 70)
        print("📊 自动统计描述报告")
        if self.source_table_name:
            print(f"  源表: {self.source_table_name}")
        print("=" * 70)

        for col in self.data.columns:
            if col == self.target_col:
                continue
            summary = self._get_variable_summary(col)
            var_type = summary['type']
            print(f"\n【变量】{col}")
            print(f"类型: {self._get_type_description(var_type)}")
            print(f"样本量: {summary['n']} (缺失: {summary['n_missing']}, {summary['missing_pct']:.1f}%)")

            if var_type == 'identifier':
                print(f"唯一值数: {summary['n_unique']}")
                print(f"范围: [{summary['min']}, {summary['max']}]")
            elif var_type == 'datetime':
                print(f"日期范围: {summary['min_date']} 到 {summary['max_date']}")
                print(f"时间跨度: {summary['date_range']}")
            elif var_type in ['categorical', 'categorical_numeric', 'ordinal']:
                print(f"类别数: {summary['n_unique']}")
                print(f"众数: {summary['mode']} (频数: {summary['mode_freq']}, {summary['mode_pct']:.1f}%)")
                if summary['n_unique'] <= 10:
                    print("\n类别分布:")
                    for val, count in summary['value_counts'].head().items():
                        pct = count / summary['n'] * 100
                        print(f"  {val}: {count} ({pct:.1f}%)")
            elif var_type == 'continuous':
                print(f"均值 ± 标准差: {summary['mean']:.2f} ± {summary['std']:.2f}")
                print(f"中位数 [Q1, Q3]: {summary['median']:.2f} [{summary['q1']:.2f}, {summary['q3']:.2f}]")
                print(f"范围: [{summary['min']:.2f}, {summary['max']:.2f}]")
                print(f"偏度: {summary['skew']:.2f}, 峰度: {summary['kurtosis']:.2f}")
                norm_status = "✅ 正态分布" if summary['is_normal'] else "⚠️ 非正态分布"
                print(f"正态性: {norm_status} (p={summary['normality_p']:.4f})")

    def _get_variable_summary(self, col):
        """获取变量摘要"""
        data = self.data[col].dropna()
        var_type = self.variable_types.get(col, 'unknown')
        summary = {
            'name': col,
            'type': var_type,
            'n': len(data),
            'n_missing': self.data[col].isna().sum(),
            'missing_pct': self.data[col].isna().mean() * 100
        }
        if var_type in ['categorical', 'categorical_numeric', 'ordinal']:
            value_counts = data.value_counts()
            summary.update({
                'n_unique': data.nunique(),
                'mode': data.mode().iloc[0] if not data.mode().empty else None,
                'mode_freq': value_counts.iloc[0] if not value_counts.empty else 0,
                'mode_pct': (value_counts.iloc[0] / len(data) * 100) if not value_counts.empty else 0,
                'value_counts': value_counts
            })
        elif var_type == 'continuous':
            is_normal, p_norm, norm_stats = self._check_normality(data)
            summary.update({
                'mean': data.mean(),
                'std': data.std(),
                'median': data.median(),
                'q1': data.quantile(0.25),
                'q3': data.quantile(0.75),
                'min': data.min(),
                'max': data.max(),
                'skew': data.skew(),
                'kurtosis': data.kurtosis(),
                'is_normal': is_normal,
                'normality_p': p_norm
            })
        elif var_type == 'identifier':
            summary.update({
                'n_unique': data.nunique(),
                'min': data.min(),
                'max': data.max()
            })
        elif var_type == 'datetime':
            summary.update({
                'min_date': data.min(),
                'max_date': data.max(),
                'date_range': data.max() - data.min(),
                'n_unique': data.nunique()
            })
        return summary

    def generate_full_report(self, show_outlier_details=False):
        """生成完整报告"""
        print("\n" + "=" * 70)
        print("📋 完整自动分析报告")
        print("=" * 70)

        print("\n【数据质量摘要】")
        self._print_quality_summary()

        print("\n【1. 变量类型识别】")
        for col, var_type in self.variable_types.items():
            type_desc = self._get_type_description(var_type)
            reason = getattr(self, 'type_reasons', {}).get(col, '')
            print(f"  {col}: {type_desc} ({reason})")

        if self.cleaning_suggestions:
            print("\n【清洗建议】")
            for suggestion in self.cleaning_suggestions[:5]:
                print(f"  {suggestion}")

        self.auto_describe()
        self.auto_time_series_analysis()
        self.auto_analyze_relationships()
        self.recommend_scenarios()

    def auto_time_series_analysis(self, max_numeric=10, group_by='auto'):
        """自动时间序列分析"""
        date_cols = [col for col, typ in self.variable_types.items() if typ == 'datetime']
        if not date_cols:
            print("\n📅 未发现日期变量，跳过时间序列分析")
            return

        date_col = date_cols[0]
        print("\n" + "=" * 70)
        print("📅 自动时间序列分析")
        print("=" * 70)

        numeric_cols = [col for col, typ in self.variable_types.items() if typ == 'continuous']
        if len(numeric_cols) > max_numeric:
            print(f"   ℹ️ 发现 {len(numeric_cols)} 个数值变量，分析前 {max_numeric} 个")
            numeric_cols = numeric_cols[:max_numeric]

        for num_col in numeric_cols:
            ts_data = self.data.groupby(date_col)[num_col].mean().reset_index()
            ts_data = ts_data.dropna()
            if len(ts_data) < 10:
                print(f"\n  ⚠️ {num_col}: 样本量不足 (n={len(ts_data)}<10)")
                continue
            print(f"\n📈 分析变量: {num_col}")
            self._analyze_time_series(ts_data, date_col, num_col)

    def _analyze_time_series(self, ts_data, date_col, value_col, entity_name=""):
        """分析单个时间序列"""
        ts_data = ts_data.set_index(date_col)
        series = ts_data[value_col]

        try:
            adf_result = adfuller(series, autolag='AIC', regression='ct')
            adf_p = adf_result[1]
            is_stationary = adf_p < 0.05
            print(f"\n  【平稳性检验】p值: {adf_p:.4f} → {'平稳' if is_stationary else '非平稳'}")
        except Exception as e:
            print(f"  ⚠️ 平稳性检验失败: {e}")

        try:
            max_lag = min(20, len(series) // 5)
            if max_lag >= 2:
                lb_result = acorr_ljungbox(series, lags=[max_lag], return_df=True)
                lb_p = lb_result['lb_pvalue'].iloc[0]
                has_autocorrelation = lb_p < 0.05
                print(f"  【自相关检验】p值: {lb_p:.4f} → {'存在自相关' if has_autocorrelation else '白噪声'}")
        except:
            pass

        print(f"\n  【基本统计量】")
        print(f"     均值: {series.mean():.2f}, 标准差: {series.std():.2f}")
        print(f"     最小值: {series.min():.2f}, 最大值: {series.max():.2f}")

    def auto_analyze_relationships(self):
        """自动关系分析"""
        print("\n" + "=" * 70)
        print("🔗 自动关系分析报告")
        print("=" * 70)

        numeric_vars = [col for col in self.data.columns if self.variable_types.get(col) == 'continuous']
        categorical_vars = [col for col in self.data.columns
                           if self.variable_types.get(col) in ['categorical', 'categorical_numeric', 'ordinal']]

        if len(numeric_vars) >= 2:
            print("\n【数值变量相关系数】")
            corr_data = self.data[numeric_vars].corr()
            print(corr_data.round(4))

        if len(categorical_vars) >= 2:
            print("\n【分类变量关联】")
            for i in range(min(5, len(categorical_vars))):
                for j in range(i+1, min(5, len(categorical_vars))):
                    col1, col2 = categorical_vars[i], categorical_vars[j]
                    condition = self.condition_checker.check_categorical_relationship(col1, col2)
                    if condition['suitable']:
                        crosstab = pd.crosstab(self.data[col1], self.data[col2])
                        print(f"  {col1} ↔ {col2}: {condition['method']}")

    def recommend_scenarios(self):
        """场景推荐"""
        print("\n" + "=" * 80)
        print("💡 智能场景推荐")
        print("=" * 80)

        numeric_vars = [col for col in self.data.columns if self.variable_types.get(col) == 'continuous']
        categorical_vars = [col for col in self.data.columns
                           if self.variable_types.get(col) in ['categorical', 'categorical_numeric', 'ordinal']]
        datetime_vars = [col for col in self.data.columns if self.variable_types.get(col) == 'datetime']

        print(f"\n【数据概况】")
        print(f"  • 数值变量: {len(numeric_vars)} 个")
        print(f"  • 分类变量: {len(categorical_vars)} 个")
        print(f"  • 日期变量: {len(datetime_vars)} 个")

        if categorical_vars:
            print("\n【分类模型推荐】")
            for cat_col in categorical_vars[:3]:
                n_classes = self.data[cat_col].nunique()
                if 2 <= n_classes <= 20:
                    print(f"  • {cat_col} ({n_classes}类) → 分类预测")

        if numeric_vars:
            print("\n【回归模型推荐】")
            for num_col in numeric_vars[:3]:
                print(f"  • {num_col} → 数值预测")

        if datetime_vars and numeric_vars:
            print("\n【时间序列分析推荐】")
            print(f"  • 可用日期列: {datetime_vars[0]}")
            print(f"  • 可预测指标: {', '.join(numeric_vars[:3])}")