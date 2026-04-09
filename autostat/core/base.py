"""
基础分析模块：类型识别、质量检查、数据清洗
"""

import numpy as np
import pandas as pd
from scipy.stats import shapiro, normaltest
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


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

    def _quick_pre_screen(self):
        """快速初筛"""
        self.type_inference_warnings = {}

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
                        if not self.quiet:
                            print(f"  ⚠️ 警告: {col} 包含 {empty_mask.sum()} 个空字符串，已替换为NaN")
                        self.data.loc[empty_mask, col] = np.nan
                        self.type_inference_warnings[col] = '包含空字符串'
                except Exception as e:
                    continue

            if self.data[col].isna().all():
                if not self.quiet:
                    print(f"  ⚠️ 警告: {col} 全部为空值")
                self.type_inference_warnings[col] = '全为空值'

            if pd.api.types.is_numeric_dtype(self.data[col]):
                missing_flags = [-999, -9999, 999999, 9999999]
                for flag in missing_flags:
                    flag_mask = self.data[col] == flag
                    if flag_mask.any():
                        if not self.quiet:
                            print(f"  ⚠️ 警告: {col} 包含 {flag_mask.sum()} 个缺失标记值 {flag}")
                        self.type_inference_warnings[col] = f'包含缺失标记 {flag}'

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
                    is_normal, _, norm_stats = self._check_normality(values)
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
        """自动清洗"""
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
                'value_counts': value_counts  # 这个字段用于显示类别分布
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