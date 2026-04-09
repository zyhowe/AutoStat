"""
场景推荐模块：模型建议（传统/机器学习/深度学习/大模型）
"""

import numpy as np
from typing import List, Dict, Any
import warnings

warnings.filterwarnings('ignore')


class RecommendationAnalyzer:
    """场景推荐器"""

    def __init__(self, data, variable_types, quality_report, time_series_diagnostics=None):
        self.data = data
        self.variable_types = variable_types
        self.quality_report = quality_report
        self.time_series_diagnostics = time_series_diagnostics or {}

    def recommend_scenarios(self):
        """场景推荐"""
        print("\n" + "=" * 80)
        print("💡 智能场景推荐")
        print("=" * 80)

        all_vars = [col for col in self.data.columns
                    if self.variable_types.get(col) not in ['identifier', 'empty']]

        numeric_vars = [col for col in all_vars if self.variable_types.get(col) == 'continuous']
        categorical_vars = [col for col in all_vars
                            if self.variable_types.get(col) in ['categorical', 'categorical_numeric', 'ordinal']]
        datetime_vars = [col for col in all_vars if self.variable_types.get(col) == 'datetime']

        print(f"\n【📊 数据概况】")
        print(f"  • 数值变量: {len(numeric_vars)} 个")
        print(f"  • 分类变量: {len(categorical_vars)} 个")
        print(f"  • 日期变量: {len(datetime_vars)} 个")

        # 获取模型推荐
        model_recs = self._get_model_recommendations(numeric_vars, categorical_vars, datetime_vars)

        print("\n\n【🎯 详细建模建议】")
        print("=" * 80)
        for rec in model_recs:
            print(f"\n【{rec['priority']}优先级】{rec['task_type']}")
            print(f"  传统模型：{rec['traditional']}")
            print(f"  机器学习：{rec['ml']}")
            print(f"  深度学习：{rec['dl']}")
            print(f"  大模型：{rec['llm']}")
            print(f"  📌 建议原因：{rec['reason']}")
            if rec['caution']:
                print(f"  ⚠️ 注意事项：{rec['caution']}")
            print("-" * 60)

    def _get_model_recommendations(self, numeric_vars, categorical_vars, datetime_vars):
        """生成详细的模型建议（基于实际字段）"""
        recommendations = []

        # 1. 分类任务推荐 - 使用实际分类字段
        for cat_col in categorical_vars[:3]:
            n_classes = self.data[cat_col].nunique()
            if n_classes < 2 or n_classes > 50:
                continue

            # 检查是否不平衡
            vc = self.data[cat_col].value_counts(normalize=True)
            is_balanced = vc.max() < 0.7
            top_class = vc.index[0] if len(vc) > 0 else "未知"
            top_pct = round(vc.max() * 100, 1) if len(vc) > 0 else 0

            rec = {
                'priority': '高' if n_classes <= 10 else '中',
                'task_type': f'分类预测 - {cat_col} ({n_classes}类)',
                'target_column': cat_col,
                'traditional': '逻辑回归' if n_classes == 2 else '多项逻辑回归',
                'ml': '随机森林 / XGBoost / LightGBM / CatBoost',
                'dl': 'MLP / TabNet / NODE',
                'llm': 'TabLLM / LLM+Embedding / GPT-4 with few-shot',
                'reason': f'目标字段「{cat_col}」有{n_classes}个类别，' + (
                    '样本分布较均衡' if is_balanced else f'「{top_class}」类占{top_pct}%，存在不平衡'),
                'caution': ''
            }

            if not is_balanced:
                rec['caution'] += f'⚠️ 类别不平衡（{top_class}占{top_pct}%），建议使用SMOTE过采样或Focal Loss；'
            if n_classes > 20:
                rec['caution'] += '⚠️ 类别数较多，考虑分层采样或类别合并；'

            recommendations.append(rec)

        # 2. 回归任务推荐 - 使用实际数值字段
        for num_col in numeric_vars[:3]:
            data = self.data[num_col].dropna()
            skew = abs(data.skew()) if len(data) > 0 else 0
            is_normal = skew < 2

            rec = {
                'priority': '高',
                'task_type': f'数值预测 - {num_col}',
                'target_column': num_col,
                'traditional': '线性回归' if is_normal else '岭回归 / Lasso / 弹性网络',
                'ml': '随机森林 / XGBoost / LightGBM / SVR',
                'dl': 'MLP / DeepAR / TabNet',
                'llm': 'TimeGPT / LLM+时序编码 / 大模型微调',
                'reason': f'目标字段「{num_col}」呈{"正态分布" if is_normal else "偏态分布"}（偏度={skew:.2f}）',
                'caution': ''
            }

            if not is_normal:
                rec['caution'] += '⚠️ 目标变量偏态，建议对数变换或使用Quantile Regression；'

            recommendations.append(rec)

        # 3. 时间序列预测推荐 - 使用实际日期和数值字段
        if datetime_vars and numeric_vars:
            # 检查是否有自相关性
            has_autocorr = False
            for diag in self.time_series_diagnostics.values():
                if diag.get('has_autocorrelation'):
                    has_autocorr = True
                    break

            if has_autocorr:
                rec = {
                    'priority': '高',
                    'task_type': f'时间序列预测',
                    'target_column': f'{numeric_vars[0]} 按 {datetime_vars[0]}',
                    'traditional': 'ARIMA / SARIMA / ETS / 指数平滑',
                    'ml': 'LightGBM with time features / XGBoost with lags / Prophet',
                    'dl': 'LSTM / GRU / Transformer / TCN / N-BEATS / Informer',
                    'llm': 'TimeGPT / LLM-Time / GPT4TS / TimesFM',
                    'reason': f'检测到「{numeric_vars[0]}」在时间维度上有显著自相关，适合时间序列预测',
                    'caution': '⚠️ 需要确保时间顺序正确；⚠️ 注意数据平稳性，可能需要差分'
                }
                recommendations.append(rec)

        return recommendations[:8]