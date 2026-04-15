"""
场景推荐模块：模型建议（传统/机器学习/深度学习/大模型）
"""

import numpy as np
import pandas as pd
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
            print(f"  目标字段: {rec['target_column']}")
            if rec.get('feature_columns'):
                print(f"  推荐特征: {', '.join(rec['feature_columns'][:5])}")
                if len(rec.get('feature_columns', [])) > 5:
                    print(f"            ... 还有 {len(rec['feature_columns']) - 5} 个字段")
            print(f"  传统模型：{rec['traditional']}")
            print(f"  机器学习：{rec['ml']}")
            print(f"  深度学习：{rec['dl']}")
            print(f"  大模型：{rec['llm']}")
            print(f"  📌 建议原因：{rec['reason']}")
            if rec['caution']:
                print(f"  ⚠️ 注意事项：{rec['caution']}")
            print("-" * 60)

    def _get_model_recommendations(self, numeric_vars, categorical_vars, datetime_vars):
        """生成详细的模型建议（基于实际字段，包含推荐特征）"""
        recommendations = []

        # 获取所有可用作特征的字段
        all_vars = numeric_vars + categorical_vars + datetime_vars

        # ==================== 1. 分类任务推荐 ====================
        for cat_col in categorical_vars[:5]:
            n_classes = self.data[cat_col].nunique()
            if n_classes < 2 or n_classes > 50:
                continue

            # 检查是否不平衡
            vc = self.data[cat_col].value_counts(normalize=True)
            is_balanced = vc.max() < 0.7
            top_class = vc.index[0] if len(vc) > 0 else "未知"
            top_pct = round(vc.max() * 100, 1) if len(vc) > 0 else 0
            sample_count = self.data[cat_col].dropna().shape[0]

            # 查找与目标变量相关的特征
            candidate_features = self._find_features_for_target(cat_col, numeric_vars, categorical_vars, datetime_vars)
            top_features = [f[0] for f in candidate_features[:8]] if candidate_features else []

            rec = {
                'priority': '高' if n_classes <= 10 else '中',
                'task_type': f'分类预测 - 预测 {cat_col}',
                'target_column': cat_col,
                'feature_columns': top_features,
                'traditional': '逻辑回归' if n_classes == 2 else '多项逻辑回归',
                'ml': '随机森林 / XGBoost / LightGBM / CatBoost',
                'dl': 'MLP / TabNet / NODE',
                'llm': 'TabLLM / LLM+Embedding / GPT-4 with few-shot',
                'reason': f'目标字段「{cat_col}」有{n_classes}个类别，样本量{sample_count}，' + (
                    '样本分布较均衡' if is_balanced else f'「{top_class}」类占{top_pct}%，存在不平衡'),
                'caution': ''
            }

            if not is_balanced:
                rec['caution'] += f'⚠️ 类别不平衡（{top_class}占{top_pct}%），建议使用SMOTE过采样或Focal Loss；'
            if n_classes > 20:
                rec['caution'] += '⚠️ 类别数较多，考虑分层采样或类别合并；'
            if sample_count < 1000:
                rec['caution'] += '⚠️ 样本量较小，建议使用简单模型或增加数据；'
            if len(top_features) < 2:
                rec['caution'] += '⚠️ 相关特征较少，预测效果可能受限；'

            recommendations.append(rec)

        # ==================== 2. 回归任务推荐 ====================
        for num_col in numeric_vars[:3]:
            data = self.data[num_col].dropna()
            skew = abs(data.skew()) if len(data) > 0 else 0
            is_normal = skew < 2
            sample_count = len(data)

            # 查找与目标变量相关的特征
            candidate_features = self._find_features_for_target(num_col, numeric_vars, categorical_vars, datetime_vars)
            top_features = [f[0] for f in candidate_features[:8]] if candidate_features else []

            rec = {
                'priority': '高',
                'task_type': f'数值预测 - 预测 {num_col}',
                'target_column': num_col,
                'feature_columns': top_features,
                'traditional': '线性回归' if is_normal else '岭回归 / Lasso / 弹性网络',
                'ml': '随机森林 / XGBoost / LightGBM / SVR',
                'dl': 'MLP / DeepAR / TabNet',
                'llm': 'TimeGPT / LLM+时序编码 / 大模型微调',
                'reason': f'目标字段「{num_col}」呈{"正态分布" if is_normal else "偏态分布"}（偏度={skew:.2f}），样本量{sample_count}',
                'caution': ''
            }

            if not is_normal:
                rec['caution'] += '⚠️ 目标变量偏态，建议对数变换或使用Quantile Regression；'
            if sample_count < 500:
                rec['caution'] += '⚠️ 样本量较小，建议使用简单模型；'
            if len(top_features) < 2:
                rec['caution'] += '⚠️ 相关特征较少，预测效果可能受限；'

            recommendations.append(rec)

        # ==================== 3. 时间序列预测推荐 ====================
        if datetime_vars and numeric_vars:
            date_col = datetime_vars[0]
            has_autocorr = False
            forecastable_sequences = []

            for diag in self.time_series_diagnostics.values():
                if diag.get('has_autocorrelation'):
                    has_autocorr = True
                    break

            if has_autocorr:
                for num_col in numeric_vars[:2]:
                    rec = {
                        'priority': '高',
                        'task_type': f'时间序列预测 - 预测 {num_col}',
                        'target_column': num_col,
                        'time_column': date_col,
                        'feature_columns': [date_col, f'{date_col}_month', f'{date_col}_year'],
                        'traditional': 'ARIMA / SARIMA / ETS / 指数平滑',
                        'ml': 'LightGBM with time features / XGBoost with lags / Prophet',
                        'dl': 'LSTM / GRU / Transformer / TCN / N-BEATS / Informer',
                        'llm': 'TimeGPT / LLM-Time / GPT4TS / TimesFM',
                        'reason': f'检测到「{num_col}」在时间维度上有显著自相关，适合时间序列预测',
                        'caution': '⚠️ 需要确保时间顺序正确；⚠️ 注意数据平稳性，可能需要差分'
                    }
                    recommendations.append(rec)

        return recommendations[:8]

    def _find_features_for_target(self, target_col, numeric_vars, categorical_vars, datetime_vars):
        """
        查找与目标变量相关的特征字段

        参数:
        - target_col: 目标变量名
        - numeric_vars: 数值变量列表
        - categorical_vars: 分类变量列表
        - datetime_vars: 日期变量列表

        返回:
        - 排序后的特征列表，每个元素为 (特征名, 相关性得分)
        """
        candidate_features = []

        # 获取所有可用特征（排除目标变量本身）
        all_vars = numeric_vars + categorical_vars + datetime_vars
        feature_vars = [v for v in all_vars if v != target_col]

        for feat in feature_vars:
            if feat not in self.data.columns:
                continue

            type_feat = self.variable_types.get(feat, 'unknown')
            type_target = self.variable_types.get(target_col, 'unknown')

            # 数值-数值相关性
            if type_feat == 'continuous' and type_target == 'continuous':
                try:
                    valid_data = self.data[[target_col, feat]].dropna()
                    if len(valid_data) >= 10:
                        is_norm_target, _, _ = self._check_normality(valid_data[target_col])
                        is_norm_feat, _, _ = self._check_normality(valid_data[feat])
                        if is_norm_target and is_norm_feat:
                            corr, p_value = pearsonr(valid_data[target_col], valid_data[feat])
                        else:
                            corr, p_value = spearmanr(valid_data[target_col], valid_data[feat])
                        if abs(corr) > 0.1 and p_value < 0.05:
                            candidate_features.append((feat, abs(corr), 'numeric'))
                except:
                    pass

            # 分类-分类关联
            elif type_feat in ['categorical', 'categorical_numeric', 'ordinal'] and \
                 type_target in ['categorical', 'categorical_numeric', 'ordinal']:
                try:
                    crosstab = pd.crosstab(self.data[target_col], self.data[feat])
                    if crosstab.size > 0 and crosstab.shape[0] > 1 and crosstab.shape[1] > 1:
                        chi2, p_value, dof, expected = chi2_contingency(crosstab)
                        n = len(self.data)
                        min_dim = min(crosstab.shape) - 1
                        cramer_v = np.sqrt(chi2 / (n * min_dim)) if chi2 > 0 and n > 0 and min_dim > 0 else 0
                        if cramer_v > 0.1 and p_value < 0.05:
                            candidate_features.append((feat, cramer_v, 'categorical'))
                except:
                    pass

            # 数值-分类关联
            elif type_feat == 'continuous' and type_target in ['categorical', 'categorical_numeric', 'ordinal']:
                try:
                    groups = [self.data[self.data[target_col] == name][feat].dropna()
                              for name in self.data[target_col].unique()
                              if len(self.data[self.data[target_col] == name]) > 1]
                    groups = [g for g in groups if len(g) > 1]
                    if len(groups) >= 2:
                        all_values = self.data[feat].dropna()
                        ss_between = sum(len(g) * (g.mean() - all_values.mean()) ** 2 for g in groups)
                        ss_total = sum((all_values - all_values.mean()) ** 2)
                        eta_sq = ss_between / ss_total if ss_total > 0 else 0
                        if eta_sq > 0.05:
                            candidate_features.append((feat, eta_sq, 'numeric-cat'))
                except:
                    pass

            # 分类-数值关联
            elif type_feat in ['categorical', 'categorical_numeric', 'ordinal'] and type_target == 'continuous':
                try:
                    groups = [self.data[self.data[feat] == name][target_col].dropna()
                              for name in self.data[feat].unique()
                              if len(self.data[self.data[feat] == name]) > 1]
                    groups = [g for g in groups if len(g) > 1]
                    if len(groups) >= 2:
                        all_values = self.data[target_col].dropna()
                        ss_between = sum(len(g) * (g.mean() - all_values.mean()) ** 2 for g in groups)
                        ss_total = sum((all_values - all_values.mean()) ** 2)
                        eta_sq = ss_between / ss_total if ss_total > 0 else 0
                        if eta_sq > 0.05:
                            candidate_features.append((feat, eta_sq, 'cat-numeric'))
                except:
                    pass

        # 按相关性得分排序
        candidate_features.sort(key=lambda x: x[1], reverse=True)

        return candidate_features

    def _check_normality(self, x):
        """检查正态性"""
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