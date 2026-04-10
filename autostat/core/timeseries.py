"""
时间序列分析模块：分组识别、平稳性检验、自相关检验、季节性检测
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Dict, List, Optional, Any
import warnings

from autostat.core.plots import plot_timeseries
from autostat.core.base import BaseAnalyzer

warnings.filterwarnings('ignore')


class TimeSeriesAnalyzer:
    """时间序列分析器"""

    def __init__(self, data, variable_types, date_derived_columns=None, quiet=False):
        self.data = data
        self.variable_types = variable_types
        self.date_derived_columns = date_derived_columns or set()
        self.quiet = quiet
        self.time_series_diagnostics = {}
        self.time_series_grouping = {}

    def _get_type_description(self, var_type):
        """获取类型描述 - 调用 BaseAnalyzer 静态方法"""
        return BaseAnalyzer.get_type_description(var_type)

    def identify_time_series_grouping(self, date_col=None):
        """自动识别时间序列分组字段"""
        print("\n" + "=" * 70)
        print("📅 时间序列分组识别")
        print("=" * 70)

        if date_col is None:
            date_cols = [col for col, typ in self.variable_types.items() if typ == 'datetime']
            if not date_cols:
                print("  ❌ 未发现日期变量，无法进行时间序列分组识别")
                return None
            date_col = date_cols[0]
            print(f"  ✅ 自动选择日期列: {date_col}")
        else:
            if date_col not in self.data.columns:
                print(f"  ❌ 指定的日期列 {date_col} 不存在")
                return None

        candidate_cols = []
        for col in self.data.columns:
            if col == date_col:
                continue
            if col in self.date_derived_columns:
                continue
            if self.variable_types.get(col) == 'datetime':
                continue
            if self.variable_types.get(col) not in ['identifier', 'categorical', 'categorical_numeric', 'ordinal']:
                continue
            candidate_cols.append(col)

        if not candidate_cols:
            print("  ❌ 没有符合条件的候选字段")
            return None

        print(f"\n【候选字段】共 {len(candidate_cols)} 个")
        for col in candidate_cols[:10]:
            type_desc = self._get_type_description(self.variable_types.get(col))
            print(f"  - {col}: {type_desc}")
        if len(candidate_cols) > 10:
            print(f"    ... 还有{len(candidate_cols) - 10}个")

        candidates = []
        excluded_reasons = []

        for col in candidate_cols:
            exclusion_reason = None
            exclude_detail = {}
            n_entities = self.data[col].nunique()

            daily_counts = self.data.groupby([date_col, col]).size()
            if daily_counts.max() > 1:
                max_duplicates = daily_counts.max()
                duplicate_dates = daily_counts[daily_counts > 1].index.get_level_values(0).unique()
                duplicate_dates_count = len(duplicate_dates)

                exclusion_reason = "同一天多条记录"
                exclude_detail = {
                    'col': col,
                    'n_entities': n_entities,
                    'max_duplicates': max_duplicates,
                    'duplicate_dates': duplicate_dates_count,
                    'total_dates': self.data[date_col].nunique()
                }
                excluded_reasons.append({
                    'col': col,
                    'reason': exclusion_reason,
                    'detail': exclude_detail
                })
                continue

            if n_entities > 1:
                entity_records = self.data.groupby(col).size()
                min_records = entity_records.min()
                avg_records = entity_records.mean()

                if min_records < 3:
                    exclusion_reason = "实体记录数不足"
                    exclude_detail = {
                        'col': col,
                        'n_entities': n_entities,
                        'min_records': min_records,
                        'avg_records': avg_records,
                        'max_records': entity_records.max()
                    }
                    excluded_reasons.append({
                        'col': col,
                        'reason': exclusion_reason,
                        'detail': exclude_detail
                    })
                    continue

                total_dates = self.data[date_col].nunique()
                density = avg_records / total_dates if total_dates > 0 else 0

                if density < 0.05:
                    exclusion_reason = "数据密度过低"
                    exclude_detail = {
                        'col': col,
                        'n_entities': n_entities,
                        'density': density,
                        'min_records': min_records,
                        'avg_records': avg_records,
                        'total_dates': total_dates
                    }
                    excluded_reasons.append({
                        'col': col,
                        'reason': exclusion_reason,
                        'detail': exclude_detail
                    })
                    continue

            candidate_info = {
                'col': col,
                'n_entities': n_entities,
                'scenario': 'single' if n_entities == 1 else 'multiple',
                'type': self.variable_types.get(col)
            }

            if n_entities > 1:
                total_dates = self.data[date_col].nunique()
                entity_records = self.data.groupby(col).size()
                density = entity_records.mean() / total_dates if total_dates > 0 else 0

                date_range = (self.data[date_col].max() - self.data[date_col].min()).days
                span_score = min(date_range / 90, 1.0)

                if len(entity_records) > 1:
                    cv = entity_records.std() / entity_records.mean() if entity_records.mean() > 0 else 1
                    stability = 1 / (1 + cv)
                else:
                    stability = 1.0

                total_score = 0.5 * density + 0.25 * span_score + 0.25 * stability

                candidate_info.update({
                    'density': density,
                    'density_pct': f"{density:.1%}",
                    'span_score': span_score,
                    'stability': stability,
                    'total_score': total_score,
                    'total_dates': total_dates,
                    'min_records': min_records,
                    'avg_records': avg_records,
                })

            candidates.append(candidate_info)

        if excluded_reasons:
            print(f"\n【字段排除原因分析】")
            reason_stats = {}
            for item in excluded_reasons:
                reason = item['reason']
                if reason not in reason_stats:
                    reason_stats[reason] = []
                reason_stats[reason].append(item)

            for reason, items in reason_stats.items():
                print(f"\n  📌 {reason}: {len(items)} 个字段")
                for item in items[:5]:
                    detail = item['detail']
                    if reason == "同一天多条记录":
                        duplicate_pct = detail['duplicate_dates'] / detail['total_dates'] * 100
                        print(f"     - {item['col']}: {detail['n_entities']}个实体, "
                              f"最多{detail['max_duplicates']}条/天, "
                              f"涉及{detail['duplicate_dates']}天 ({duplicate_pct:.1f}%)")
                    elif reason == "实体记录数不足":
                        print(f"     - {item['col']}: {detail['n_entities']}个实体, "
                              f"最少{detail['min_records']}条, "
                              f"平均{detail['avg_records']:.1f}条")
                    elif reason == "数据密度过低":
                        print(f"     - {item['col']}: {detail['n_entities']}个实体, "
                              f"密度{detail['density']:.1%}, "
                              f"平均{detail['avg_records']:.1f}条/实体")
                if len(items) > 5:
                    print(f"     ... 还有{len(items) - 5}个")

        if not candidates:
            print("\n  ❌ 没有字段满足时间序列分组条件")
            return {
                'has_group': False,
                'date_col': date_col,
                'group_col': None,
                'scenario': None,
                'candidates': [],
                'excluded_reasons': excluded_reasons,
                'message': '没有字段满足时间序列分组条件'
            }

        single_entity_candidates = [c for c in candidates if c['scenario'] == 'single']
        if single_entity_candidates:
            best = single_entity_candidates[0]
            result = {
                'has_group': True,
                'date_col': date_col,
                'group_col': best['col'],
                'scenario': 'single',
                'n_entities': 1,
                'candidates': candidates,
                'excluded_reasons': excluded_reasons,
                'message': f'检测到单实体时间序列，分组字段: {best["col"]}'
            }
            self._print_grouping_result(result)
            self.time_series_grouping = result
            return result

        multi_candidates = [c for c in candidates if c['scenario'] == 'multiple']
        multi_candidates.sort(key=lambda x: x['total_score'], reverse=True)
        best = multi_candidates[0]

        result = {
            'has_group': True,
            'date_col': date_col,
            'group_col': best['col'],
            'scenario': 'multiple',
            'n_entities': best['n_entities'],
            'best_score': best['total_score'],
            'candidates': multi_candidates,
            'excluded_reasons': excluded_reasons,
            'message': f'推荐分组字段: {best["col"]} (得分: {best["total_score"]:.2f})'
        }

        self._print_grouping_result(result)
        self.time_series_grouping = result
        return result

    def _print_grouping_result(self, result):
        print("\n【分组识别结果】")
        print(f"  {'─' * 50}")

        if not result['has_group']:
            print(f"  ❌ {result['message']}")
            return

        print(f"  ✅ 日期列: {result['date_col']}")
        print(f"  ✅ 分组字段: {result['group_col']}")

        if result['scenario'] == 'single':
            print(f"  📌 场景类型: 单实体时间序列")
            print(f"  📊 实体数量: 1")
            print("\n  💡 分析建议:")
            print("     - 直接进行单变量时间序列分析")
            print("     - 可用模型: ARIMA/SARIMA/Prophet/指数平滑")
        else:
            print(f"  📌 场景类型: 多实体时间序列")
            print(f"  📊 实体数量: {result['n_entities']}")
            print(f"  📊 综合得分: {result['best_score']:.3f}")

            best = [c for c in result['candidates'] if c['col'] == result['group_col']][0]
            print(f"  📊 数据密度: {best.get('density_pct', 'N/A')}")

            print("\n  💡 分析建议:")
            print("     - 按推荐字段分组进行时间序列分析")
            print("     - 可考虑对实体进行聚类后再分析")

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

        group_col = None
        if group_by == 'auto':
            grouping_result = self.identify_time_series_grouping(date_col)
            if grouping_result and grouping_result['has_group']:
                group_col = grouping_result['group_col']
                scenario = grouping_result['scenario']
                print(f"\n  ✅ 采用自动识别的分组字段: {group_col} ({scenario})")
                if scenario == 'single':
                    print(f"  ℹ️ 单实体场景，将按整体进行时间序列分析")
                    group_col = None
            else:
                print(f"\n  ⚠️ 未找到合适的分组字段，进行整体分析")

        if group_col:
            self._grouped_time_series_analysis(date_col, numeric_cols, group_col)
        else:
            self._simple_time_series_analysis(date_col, numeric_cols)

    def _simple_time_series_analysis(self, date_col, numeric_cols):
        print("\n【整体时间序列分析】")
        for num_col in numeric_cols:
            ts_data = self.data.groupby(date_col)[num_col].mean().reset_index()
            ts_data = ts_data.dropna()
            if len(ts_data) < 10:
                print(f"\n  ⚠️ {num_col}: 样本量不足 (n={len(ts_data)}<10)，跳过")
                continue
            print(f"\n📈 分析变量: {num_col}")
            self._analyze_time_series(ts_data, date_col, num_col, entity_name="整体")

    def _grouped_time_series_analysis(self, date_col, numeric_cols, group_col):
        n_entities = self.data[group_col].nunique()
        print(f"\n【分组时间序列分析 - 按 {group_col}】")
        print(f"  实体数量: {n_entities}")

        for num_col in numeric_cols:
            print(f"\n📈 分析变量: {num_col}")
            entities = self.data[group_col].unique()
            for entity in entities[:5]:
                entity_data = self.data[self.data[group_col] == entity]
                ts_data = entity_data.groupby(date_col)[num_col].mean().reset_index()
                ts_data = ts_data.dropna()
                if len(ts_data) >= 10:
                    print(f"\n  ▶ 实体: {entity}")
                    self._analyze_time_series(ts_data, date_col, num_col, entity_name=str(entity), detailed=True)

    def _analyze_time_series(self, ts_data, date_col, value_col, entity_name="", detailed=True):
        ts_data = ts_data.set_index(date_col)
        series = ts_data[value_col]

        if len(series) < 10:
            if detailed:
                print(f"  ⚠️ 样本量不足 (n={len(series)}<10)，跳过详细分析")
            return

        is_stationary = None
        has_autocorrelation = False
        has_seasonality = False

        try:
            adf_result = adfuller(series, autolag='AIC', regression='ct')
            adf_p = adf_result[1]
            is_stationary = adf_p < 0.05
            if detailed:
                print(f"\n  【平稳性检验 (ADF)】p值: {adf_p:.4f} → {'平稳' if is_stationary else '非平稳'}")
        except Exception as e:
            if detailed:
                print(f"  ⚠️ 平稳性检验失败: {e}")

        try:
            max_lag = min(20, len(series) // 5)
            if max_lag >= 2:
                lb_result = acorr_ljungbox(series, lags=[max_lag], return_df=True)
                lb_p = lb_result['lb_pvalue'].iloc[0]
                has_autocorrelation = lb_p < 0.05
                if detailed:
                    print(f"  【自相关性检验】p值: {lb_p:.4f} → {'存在自相关' if has_autocorrelation else '白噪声'}")
        except Exception as e:
            if detailed:
                print(f"  ⚠️ 自相关性检验失败: {e}")

        try:
            if len(series) >= 24:
                corr_12 = series.autocorr(lag=12)
                if corr_12 is not None and abs(corr_12) > 0.3:
                    has_seasonality = True
        except:
            pass

        if detailed and is_stationary is not None and has_autocorrelation is not None:
            print(f"\n  【综合诊断结论】")
            if is_stationary and has_autocorrelation:
                print(f"     ✅ 非常适合时间序列预测 → 推荐 ARIMA/SARIMA")
            elif not is_stationary and has_autocorrelation:
                print(f"     ⚠️ 有自相关但不平稳 → 推荐 差分后ARIMA")
            elif is_stationary and not has_autocorrelation:
                print(f"     ❌ 白噪声序列 → 推荐 均值/中位数")
            else:
                print(f"     ❌ 随机游走序列 → 推荐 简单趋势线")

        key = f"{value_col}_{entity_name}" if entity_name and entity_name != "整体" else value_col
        self.time_series_diagnostics[key] = {
            'is_stationary': is_stationary,
            'has_autocorrelation': has_autocorrelation,
            'has_seasonality': has_seasonality,
            'n_samples': len(series),
            'mean': series.mean(),
            'std': series.std(),
            'last_value': series.iloc[-1]
        }

        # 使用统一模块显示图表
        if detailed:
            plot_timeseries(series, value_col)