"""
基础分析模块：类型识别、质量检查、数据清洗
"""

import numpy as np
import pandas as pd
from scipy.stats import shapiro, normaltest
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# ==================== 全局常量定义 ====================

# 变量类型中文描述映射（唯一来源）
TYPE_DESCRIPTION_MAP = {
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

# 相关性阈值
HIGH_CORRELATION_THRESHOLD = 0.7  # 强相关阈值
STATISTICAL_SIGNIFICANCE_THRESHOLD = 0.05  # 统计显著阈值

# 偏态阈值
SKEWNESS_THRESHOLD = 2.0

# 峰度阈值
KURTOSIS_THRESHOLD = 7.0

# 正态性检验样本量限制
NORMALITY_MIN_SAMPLES = 8
NORMALITY_MAX_SAMPLES = 5000


class BaseAnalyzer:
    """基础分析器 - 类型识别、质量检查、数据清洗"""

    def __init__(self, data, variable_types=None, type_reasons=None, quiet=False):
        self.data = data
        self.variable_types = variable_types or {}
        self.type_reasons = type_reasons or {}
        self.quiet = quiet
        self.type_inference_warnings = {}
        self.quality_report = {}
        self.cleaning_suggestions = []

    @staticmethod
    def check_normality(x):
        """
        检查正态性（静态方法，供全局复用）

        参数:
        - x: 一维数组或Series

        返回:
        - is_normal: bool
        - p_value: float
        - stats: dict, 包含 skew, kurtosis
        """
        x = x.dropna()

        # 样本量不足
        if len(x) < NORMALITY_MIN_SAMPLES:
            return False, 1.0, {'skew': 0, 'kurtosis': 0}

        # 样本量过大时采样
        if len(x) > NORMALITY_MAX_SAMPLES:
            x = x.sample(n=NORMALITY_MAX_SAMPLES, random_state=42)

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

        is_normal = (p_value > STATISTICAL_SIGNIFICANCE_THRESHOLD and
                     skewness < SKEWNESS_THRESHOLD and
                     kurtosis < KURTOSIS_THRESHOLD)

        return is_normal, p_value, {'skew': skewness, 'kurtosis': kurtosis}

    @staticmethod
    def get_type_description(var_type):
        """获取类型描述（静态方法）"""
        return TYPE_DESCRIPTION_MAP.get(var_type, var_type)

    @staticmethod
    def get_high_correlations(data, numeric_vars, threshold=HIGH_CORRELATION_THRESHOLD):
        """获取强相关对（处理空数据）"""
        correlations = []
        if len(numeric_vars) >= 2:
            # 过滤掉全空的列
            valid_vars = [col for col in numeric_vars if col in data.columns and data[col].notna().any()]
            if len(valid_vars) >= 2:
                corr_data = data[valid_vars].corr()
                for i in range(len(corr_data.columns)):
                    for j in range(i + 1, len(corr_data.columns)):
                        val = corr_data.iloc[i, j]
                        if not pd.isna(val) and abs(val) >= threshold:
                            correlations.append({
                                'var1': corr_data.columns[i],
                                'var2': corr_data.columns[j],
                                'value': round(val, 3)
                            })
        return sorted(correlations, key=lambda x: abs(x['value']), reverse=True)

    @staticmethod
    def get_skewed_vars(data, variable_types, threshold=SKEWNESS_THRESHOLD):
        """获取偏态变量（处理空数据）"""
        skewed = []
        for col, typ in variable_types.items():
            if typ == 'continuous' and col in data.columns:
                series = data[col].dropna()
                if len(series) > 0:
                    skew = series.skew()
                    if not pd.isna(skew) and abs(skew) >= threshold:
                        skewed.append({'name': col, 'skew': round(skew, 2)})
        return sorted(skewed, key=lambda x: abs(x['skew']), reverse=True)

    @staticmethod
    def get_imbalanced_vars(data, variable_types, threshold=0.8):
        """获取不平衡分类变量（处理空数据）"""
        imbalanced = []
        for col, typ in variable_types.items():
            if typ in ['categorical', 'categorical_numeric', 'ordinal'] and col in data.columns:
                vc = data[col].value_counts(normalize=True, dropna=False)
                if len(vc) > 0:
                    # 排除NaN作为类别
                    vc_clean = vc.dropna()
                    if len(vc_clean) > 0 and vc_clean.max() >= threshold:
                        imbalanced.append({
                            'name': col,
                            'top_category': str(vc_clean.index[0]) if len(vc_clean.index) > 0 else '未知',
                            'top_pct': round(vc_clean.max() * 100, 1)
                        })
        return imbalanced

    def _quick_pre_screen(self):
        """快速初筛（增强空值处理）"""
        self.type_inference_warnings = {}

        for col in self.data.columns:
            # 跳过全空列
            if self.data[col].isna().all():
                if not self.quiet:
                    print(f"  ⚠️ 警告: {col} 全部为空值")
                self.type_inference_warnings[col] = '全为空值'
                continue

            # 处理bytes类型
            if self.data[col].dtype == 'object':
                try:
                    if len(self.data[col]) > 0:
                        sample = self.data[col].iloc[0]
                        if isinstance(sample, bytes):
                            self.data[col] = self.data[col].apply(
                                lambda x: x.decode('gbk', errors='ignore') if isinstance(x, bytes) else x
                            )
                    # 处理空字符串
                    empty_mask = self.data[col].astype(str).str.strip() == ''
                    if empty_mask.any():
                        if not self.quiet:
                            print(f"  ⚠️ 警告: {col} 包含 {empty_mask.sum()} 个空字符串，已替换为NaN")
                        self.data.loc[empty_mask, col] = np.nan
                        self.type_inference_warnings[col] = '包含空字符串'
                except Exception as e:
                    continue

            # 处理数值列中的缺失标记
            if pd.api.types.is_numeric_dtype(self.data[col]):
                missing_flags = [-999, -9999, 999999, 9999999]
                for flag in missing_flags:
                    flag_mask = self.data[col] == flag
                    if flag_mask.any():
                        if not self.quiet:
                            print(f"  ⚠️ 警告: {col} 包含 {flag_mask.sum()} 个缺失标记值 {flag}")
                        self.type_inference_warnings[col] = f'包含缺失标记 {flag}'
                        self.data.loc[flag_mask, col] = np.nan

    def _infer_variable_types(self):
        """自动推断变量类型（增强空值处理）"""
        for col in self.data.columns:
            if col in self.variable_types:
                continue

            values = self.data[col].dropna()

            # 全空列
            if len(values) == 0:
                self.variable_types[col] = 'empty'
                self.type_reasons[col] = '空变量'
                continue

            n_unique = values.nunique()
            unique_ratio = n_unique / len(values) if len(values) > 0 else 0

            # 日期类型检测
            if pd.api.types.is_string_dtype(values) or pd.api.types.is_object_dtype(values):
                try:
                    # 采样检测
                    sample_size = min(100, len(values))
                    converted = pd.to_datetime(values.head(sample_size), errors='coerce')
                    if converted.notna().sum() > sample_size * 0.8:
                        self.variable_types[col] = 'datetime'
                        self.type_reasons[col] = '日期时间类型'
                        self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                        continue
                except:
                    pass

            if pd.api.types.is_datetime64_any_dtype(values):
                self.variable_types[col] = 'datetime'
                self.type_reasons[col] = '日期时间类型'
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
                    is_normal, _, norm_stats = self.check_normality(values)
                    norm_status = "正态" if is_normal else "非正态"
                    self.type_reasons[col] = f'连续变量，{norm_status}分布 (偏度={norm_stats["skew"]:.2f})'
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
        """检查异常值（增强空值处理）"""
        outlier_report = {}
        for col, typ in self.variable_types.items():
            if typ == 'continuous' and col in self.data.columns:
                data = self.data[col].dropna()
                if len(data) > 0:
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR == 0:
                        continue
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
        """检查重复值（增强空值处理）"""
        id_cols = [col for col, typ in self.variable_types.items() if typ == 'identifier' and col in self.data.columns]
        if id_cols:
            # 只使用非空ID列检查重复
            temp_data = self.data[id_cols].dropna()
            if len(temp_data) > 0:
                duplicates = temp_data.duplicated(keep=False)
                return {
                    'count': duplicates.sum(),
                    'percent': duplicates.sum() / len(self.data) * 100 if len(self.data) > 0 else 0,
                    'based_on': id_cols
                }

        # 全列检查
        duplicates = self.data.duplicated(keep=False)
        return {
            'count': duplicates.sum(),
            'percent': duplicates.sum() / len(self.data) * 100 if len(self.data) > 0 else 0,
            'based_on': 'all_columns'
        }

    def _check_type_consistency(self) -> Dict:
        """检查类型一致性（增强空值处理）"""
        inconsistency = {}
        for col in self.data.columns:
            # 排除空值后检查类型
            non_null = self.data[col].dropna()
            if len(non_null) == 0:
                continue
            types = non_null.apply(type).unique()
            if len(types) > 1:
                inconsistency[col] = {'types': [str(t) for t in types]}
        return inconsistency

    def _check_invalid_by_type(self) -> Dict:
        """检查无效值（增强空值处理）"""
        invalid = {}
        for col, typ in self.variable_types.items():
            if col not in self.data.columns:
                continue
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

    def _comprehensive_quality_check(self) -> Dict:
        """综合质量检查"""
        return {
            'missing': self._check_missing_values(),
            'outliers': self._check_outliers_by_type(),
            'duplicates': self._check_duplicates_by_key(),
            'inconsistent_types': self._check_type_consistency(),
            'invalid_values': self._check_invalid_by_type()
        }

    def _generate_cleaning_suggestions(self) -> List[str]:
        """生成清洗建议"""
        suggestions = []
        for item in self.quality_report.get('missing', []):
            col = item['column']
            pct = item['percent']
            if pct > 80:
                suggestions.append(f"⚠️ 严重: {col} 缺失率 {pct:.1f}%，建议删除该列")
            elif pct > 50:
                suggestions.append(f"⚠️ 较高: {col} 缺失率 {pct:.1f}%，建议评估是否保留")
            elif pct > 20:
                suggestions.append(f"📌 {col} 缺失率 {pct:.1f}%，建议填充")
        for col, info in self.quality_report.get('outliers', {}).items():
            if info['percent'] > 10:
                suggestions.append(f"⚠️ {col} 异常值比例 {info['percent']:.1f}%，建议检查")
        dup_info = self.quality_report.get('duplicates', {})
        if dup_info.get('count', 0) > 0:
            suggestions.append(f"📌 发现 {dup_info['count']} 条重复记录 ({dup_info.get('percent', 0):.1f}%)")
        return suggestions

    def _auto_clean(self):
        """自动清洗（增强空值处理）"""
        if not self.quiet:
            print("\n  执行自动清洗...")
        cols_to_drop = []
        for item in self.quality_report.get('missing', []):
            if item['percent'] > 80:
                cols_to_drop.append(item['column'])
        if cols_to_drop:
            self.data = self.data.drop(columns=cols_to_drop)
        dup_info = self.quality_report.get('duplicates', {})
        if dup_info.get('count', 0) > 0:
            if dup_info.get('based_on') != 'all_columns':
                self.data = self.data.drop_duplicates(subset=dup_info['based_on'], keep='first')
            else:
                self.data = self.data.drop_duplicates(keep='first')
        self.quality_report = self._comprehensive_quality_check()

    def _get_variable_summary(self, col):
        """获取变量摘要（增强空值处理）"""
        if col not in self.data.columns:
            return {'name': col, 'type': 'unknown', 'n': 0, 'n_missing': 0, 'missing_pct': 0}

        data = self.data[col].dropna()
        var_type = self.variable_types.get(col, 'unknown')

        summary = {
            'name': col,
            'type': var_type,
            'n': len(data),
            'n_missing': self.data[col].isna().sum(),
            'missing_pct': self.data[col].isna().mean() * 100
        }

        if len(data) == 0:
            return summary

        if var_type in ['categorical', 'categorical_numeric', 'ordinal']:
            value_counts = data.value_counts()
            if len(value_counts) > 0:
                summary.update({
                    'n_unique': data.nunique(),
                    'mode': data.mode().iloc[0] if not data.mode().empty else None,
                    'mode_freq': value_counts.iloc[0] if not value_counts.empty else 0,
                    'mode_pct': (value_counts.iloc[0] / len(data) * 100) if not value_counts.empty else 0,
                    'value_counts': value_counts
                })
            else:
                summary.update({
                    'n_unique': 0,
                    'mode': None,
                    'mode_freq': 0,
                    'mode_pct': 0,
                    'value_counts': pd.Series()
                })
        elif var_type == 'continuous':
            is_normal, p_norm, norm_stats = self.check_normality(data)
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
            # 确保数据是 datetime 类型
            try:
                # 尝试转换为 datetime
                dt_data = pd.to_datetime(data, errors='coerce')
                dt_data = dt_data.dropna()
                if len(dt_data) > 0:
                    min_date = dt_data.min()
                    max_date = dt_data.max()
                    date_range = max_date - min_date
                    summary.update({
                        'min_date': min_date,
                        'max_date': max_date,
                        'date_range': date_range,
                        'n_unique': dt_data.nunique()
                    })
                else:
                    summary.update({
                        'min_date': None,
                        'max_date': None,
                        'date_range': None,
                        'n_unique': data.nunique()
                    })
            except Exception as e:
                # 转换失败，当作分类变量处理
                summary.update({
                    'min_date': None,
                    'max_date': None,
                    'date_range': None,
                    'n_unique': data.nunique()
                })
        else:
            # 其他类型，至少添加 n_unique
            summary.update({
                'n_unique': data.nunique()
            })

        return summary

    def _print_type_summary(self):
        """打印类型汇总"""
        if self.quiet:
            return
        type_counts = {}
        for typ in self.variable_types.values():
            type_counts[typ] = type_counts.get(typ, 0) + 1
        print("\n  类型汇总:")
        for typ, count in type_counts.items():
            print(f"    {self.get_type_description(typ)}: {count}列")

    def _print_quality_summary(self):
        """打印质量摘要"""
        if self.quiet:
            return
        missing_count = len(self.quality_report.get('missing', []))
        outlier_count = len(self.quality_report.get('outliers', {}))
        dup_count = self.quality_report.get('duplicates', {}).get('count', 0)
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