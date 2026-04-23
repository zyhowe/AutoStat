"""
场景推荐模块：模型建议（传统/机器学习/深度学习/大模型）
基于数据特征智能推荐分析任务
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Set, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class RecommendationAnalyzer:
    """场景推荐器 - 基于数据特征智能推荐分析任务"""

    def __init__(self, data, variable_types, quality_report, time_series_diagnostics=None):
        self.data = data
        self.variable_types = variable_types
        self.quality_report = quality_report
        self.time_series_diagnostics = time_series_diagnostics or {}

        # 派生列映射（从 analyzer 传入，如果没有则为空）
        self.date_derived_columns = set()
        self.date_column_mapping = {}
        self.date_original_columns = set()

    def set_date_info(self, date_derived_columns: set, date_column_mapping: dict, date_original_columns: set):
        """设置日期派生列信息（由外部调用）"""
        self.date_derived_columns = date_derived_columns
        self.date_column_mapping = date_column_mapping
        self.date_original_columns = date_original_columns

    def _is_derived_column(self, col: str) -> bool:
        """判断是否是派生列"""
        return col in self.date_derived_columns

    def _is_original_datetime(self, col: str) -> bool:
        """判断是否是原始日期列"""
        return col in self.date_original_columns

    def _is_same_source(self, col1: str, col2: str) -> bool:
        """判断两个列是否同源（来自同一个原始日期列）"""
        # 获取原始列
        original1 = self.date_column_mapping.get(col1, col1)
        original2 = self.date_column_mapping.get(col2, col2)

        # 如果相同，则同源
        if original1 == original2:
            return True

        # 如果一个的原始列等于另一个本身
        if original1 == col2 or original2 == col1:
            return True

        return False

    def _should_exclude_target(self, col: str) -> bool:
        """判断是否应该排除该列作为预测目标（派生列不推荐作为目标）"""
        # 派生列不推荐作为预测目标
        if self._is_derived_column(col):
            return True
        return False

    def _should_exclude_pair(self, col1: str, col2: str) -> bool:
        """判断是否应该排除这一对"""
        # 1. 如果两个都是派生列且同源，排除
        if self._is_derived_column(col1) and self._is_derived_column(col2):
            if self._is_same_source(col1, col2):
                return True

        # 2. 如果一个是派生列，一个是其原列，排除
        if self._is_derived_column(col1) and self._is_original_datetime(col2):
            if self.date_column_mapping.get(col1) == col2:
                return True
        if self._is_derived_column(col2) and self._is_original_datetime(col1):
            if self.date_column_mapping.get(col2) == col1:
                return True

        return False

    def _get_all_derived_targets(self) -> Set[str]:
        """获取所有派生列（这些不应该作为预测目标）"""
        return self.date_derived_columns.copy()

    def recommend_scenarios(self):
        """场景推荐 - 输出所有符合条件的分析任务"""
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

        # 获取所有推荐（不限个数）
        recommendations = self._get_all_recommendations(numeric_vars, categorical_vars, datetime_vars)

        if not recommendations:
            print("\n⚠️ 未发现适合的分析任务")
            return

        # 排序
        recommendations = self._sort_recommendations(recommendations)

        print("\n\n【🎯 详细分析建议】")
        print("=" * 80)

        for i, rec in enumerate(recommendations, 1):
            print(f"\n【{i}. {rec['task_type']}】{rec.get('title', '')}")
            if rec.get('description'):
                print(f"  📌 说明：{rec['description']}")
            if rec.get('target'):
                print(f"  🎯 目标：{rec['target']}")
            if rec.get('features'):
                # 全部列出，不截断
                feature_str = ', '.join(rec['features'])
                print(f"  📊 特征：{feature_str}")
            if rec.get('traditional'):
                print(f"  📈 传统模型：{rec['traditional']}")
            if rec.get('ml'):
                print(f"  🤖 机器学习：{rec['ml']}")
            if rec.get('dl'):
                print(f"  🧠 深度学习：{rec['dl']}")
            if rec.get('llm'):
                print(f"  🔮 大模型：{rec['llm']}")
            if rec.get('reason'):
                print(f"  💡 原因：{rec['reason']}")
            if rec.get('caution'):
                print(f"  ⚠️ 注意：{rec['caution']}")
            print("-" * 60)

    def _get_all_recommendations(self, numeric_vars, categorical_vars, datetime_vars) -> List[Dict]:
        """获取所有符合条件的推荐（不限个数）"""
        recommendations = []

        # 1. 时序预测（优先级最高）
        ts_recs = self._get_time_series_recommendations(numeric_vars, datetime_vars)
        recommendations.extend(ts_recs)

        # 2. 回归预测（单特征 + 多特征组合）
        regression_recs = self._get_regression_recommendations(numeric_vars)
        recommendations.extend(regression_recs)

        # 3. 分类预测（数值→分类 + 分类→分类，单特征 + 多特征组合）
        classification_recs = self._get_classification_recommendations(numeric_vars, categorical_vars)
        recommendations.extend(classification_recs)

        # 4. 聚类分析
        clustering_recs = self._get_clustering_recommendations(numeric_vars)
        if clustering_recs:
            recommendations.append(clustering_recs)

        # 5. 关联规则挖掘
        association_recs = self._get_association_recommendations(categorical_vars)
        if association_recs:
            recommendations.append(association_recs)

        # 6. 异常检测
        anomaly_recs = self._get_anomaly_detection_recommendations()
        if anomaly_recs:
            recommendations.append(anomaly_recs)

        return recommendations

    def _sort_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """按优先级排序推荐列表"""
        def get_priority(rec):
            task_type = rec.get('task_type', '')
            # 优先级权重
            if '时间序列预测' in task_type:
                return 1
            elif '回归预测' in task_type and '多特征' in rec.get('title', ''):
                return 2
            elif '回归预测' in task_type:
                # 获取相关系数
                reason = rec.get('reason', '')
                import re
                match = re.search(r'r=([0-9.]+)', reason)
                if match:
                    r = abs(float(match.group(1)))
                    if r >= 0.7:
                        return 3
                    else:
                        return 7
                return 7
            elif '分类预测' in task_type and '多特征' in rec.get('title', ''):
                return 4
            elif '分类预测' in task_type:
                # 获取效应量
                reason = rec.get('reason', '')
                import re
                match = re.search(r'V=([0-9.]+)', reason)
                if not match:
                    match = re.search(r'η²=([0-9.]+)', reason)
                if match:
                    v = abs(float(match.group(1)))
                    if v >= 0.5:
                        return 5
                    else:
                        return 8
                return 8
            elif '聚类分析' in task_type:
                return 6
            elif '关联规则挖掘' in task_type:
                return 9
            elif '异常检测' in task_type:
                return 10
            else:
                return 11

        return sorted(recommendations, key=get_priority)

    # ==================== 1. 时序预测 ====================
    def _get_time_series_recommendations(self, numeric_vars, datetime_vars) -> List[Dict]:
        """获取时序预测推荐（有自相关的数值变量）"""
        recommendations = []

        if not datetime_vars or not numeric_vars:
            return recommendations

        for var_name, diag in self.time_series_diagnostics.items():
            if not diag.get('has_autocorrelation'):
                continue

            # 解析变量名（可能包含分组信息，如 "血糖_女"）
            if '_' in var_name:
                parts = var_name.rsplit('_', 1)
                base_var = parts[0]
                group = parts[1]
                title = f"{base_var}（{group}组）"
                description = f"检测到「{base_var}」在「{group}」组中有显著自相关"
            else:
                base_var = var_name
                group = None
                title = base_var
                description = f"检测到「{base_var}」有显著自相关"

            recommendations.append({
                "task_type": "📅 时间序列预测",
                "title": title,
                "description": description,
                "target": base_var,
                "group": group,
                "features": [base_var],
                "traditional": "ARIMA / SARIMA / ETS / 指数平滑",
                "ml": "Prophet / LightGBM with time features / XGBoost with lags",
                "dl": "LSTM / GRU / Transformer / TCN / N-BEATS",
                "llm": "TimeGPT / TimesFM / LLM-Time",
                "reason": f"自相关检验 p={diag.get('lb_p', 0):.4f}，历史数据可预测未来",
                "caution": "⚠️ 需要确保时间顺序正确；⚠️ 注意数据平稳性，可能需要差分"
            })

        return recommendations

    # ==================== 2. 回归预测 ====================
    def _get_regression_recommendations(self, numeric_vars) -> List[Dict]:
        """获取回归预测推荐（单特征 + 多特征组合）"""
        recommendations = []

        if len(numeric_vars) < 2:
            return recommendations

        # 计算相关系数矩阵
        valid_numeric = [col for col in numeric_vars if col in self.data.columns]
        if len(valid_numeric) < 2:
            return recommendations

        corr_matrix = self.data[valid_numeric].corr()

        # 存储所有显著相关的配对
        significant_pairs = []  # (var1, var2, corr_value, p_value)
        target_to_features = {}  # {target: [(feature, corr), ...]}

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]

                if pd.isna(corr_value):
                    continue

                abs_corr = abs(corr_value)

                # 阈值 0.3
                if abs_corr < 0.3:
                    continue

                # 计算 p 值
                valid_data = self.data[[var1, var2]].dropna()
                if len(valid_data) < 3:
                    continue

                from scipy.stats import pearsonr, spearmanr
                is_norm1, _, _ = self._check_normality(valid_data[var1])
                is_norm2, _, _ = self._check_normality(valid_data[var2])

                if is_norm1 and is_norm2:
                    _, p_value = pearsonr(valid_data[var1], valid_data[var2])
                else:
                    _, p_value = spearmanr(valid_data[var1], valid_data[var2])

                if p_value >= 0.05:
                    continue

                # 确定目标（方差较大的作为目标）
                var1_std = self.data[var1].std()
                var2_std = self.data[var2].std()
                if var1_std > var2_std:
                    target = var1
                    feature = var2
                else:
                    target = var2
                    feature = var1

                significant_pairs.append({
                    'target': target,
                    'feature': feature,
                    'corr': corr_value,
                    'abs_corr': abs_corr,
                    'p_value': p_value
                })

                # 构建 target -> features 映射（用于多特征组合）
                if target not in target_to_features:
                    target_to_features[target] = []
                target_to_features[target].append((feature, abs_corr))

        # 生成单特征推荐
        for pair in significant_pairs:
            strength = "强" if pair['abs_corr'] > 0.7 else "中" if pair['abs_corr'] > 0.5 else "弱"
            direction = "正" if pair['corr'] > 0 else "负"

            recommendations.append({
                "task_type": "📈 回归预测",
                "title": f"{pair['target']} ← {pair['feature']}",
                "description": f"「{pair['target']}」与「{pair['feature']}」呈{strength}{direction}相关（r={pair['corr']:.3f}）",
                "target": pair['target'],
                "features": [pair['feature']],
                "traditional": "线性回归 / 岭回归 / Lasso",
                "ml": "随机森林 / XGBoost / LightGBM / SVR",
                "dl": "MLP / TabNet",
                "llm": "TimeGPT / LLM+时序编码",
                "reason": f"相关系数 r={pair['corr']:.3f}，p={pair['p_value']:.4f}，{pair['feature']} 可解释 {pair['target']} 的 {(pair['abs_corr'] ** 2 * 100):.1f}% 变异",
                "caution": "⚠️ 相关性不代表因果，建议结合业务理解"
            })

        # 生成多特征组合推荐（只针对非派生列目标）
        multi_feature_recs = self._build_multi_feature_regression(target_to_features, numeric_vars)
        recommendations.extend(multi_feature_recs)

        return recommendations

    def _build_multi_feature_regression(self, target_to_features: Dict[str, List], numeric_vars: List) -> List[Dict]:
        """构建回归多特征组合推荐（只针对非派生列目标）"""
        recommendations = []

        for target, features in target_to_features.items():
            # 排除派生列作为目标
            if self._should_exclude_target(target):
                continue

            # 按相关系数排序
            features_sorted = sorted(features, key=lambda x: x[1], reverse=True)

            # 收集所有特征（不限制数量），排除派生列
            all_features = []
            for f, _ in features_sorted:
                if not self._is_derived_column(f):
                    all_features.append(f)

            # 去重去冗余
            all_features = self._deduplicate_features(all_features, target, task_type='regression')

            if len(all_features) < 2:
                continue

            # 生成原因描述（全部列出）
            reason_parts = []
            for f, corr in features_sorted:
                if not self._is_derived_column(f):
                    reason_parts.append(f"{f}(r={corr:.3f})")
            reason_str = "、".join(reason_parts)

            recommendations.append({
                "task_type": "📈 回归预测",
                "title": f"预测 {target}（多特征组合）",
                "description": f"基于 {len(all_features)} 个特征预测「{target}」",
                "target": target,
                "features": all_features,
                "traditional": "线性回归 / 岭回归 / Lasso",
                "ml": "随机森林 / XGBoost / LightGBM / SVR",
                "dl": "MLP / TabNet",
                "llm": "TabLLM / GPT-4 with few-shot",
                "reason": f"相关特征：{reason_str}",
                "caution": "⚠️ 建议先进行特征选择；⚠️ 注意多重共线性"
            })

        return recommendations

    # ==================== 3. 分类预测 ====================
    def _get_classification_recommendations(self, numeric_vars, categorical_vars) -> List[Dict]:
        """获取分类预测推荐（数值→分类 + 分类→分类，单特征 + 多特征组合）"""
        recommendations = []

        if not categorical_vars:
            return recommendations

        # 3.1 数值 → 分类（只推荐非派生列作为目标）
        eta_recs, eta_target_to_features = self._get_numeric_to_categorical_recs(numeric_vars, categorical_vars)
        recommendations.extend(eta_recs)

        # 3.2 分类 → 分类（只推荐非派生列作为目标）
        cramer_recs, cramer_target_to_features = self._get_categorical_to_categorical_recs(categorical_vars)
        recommendations.extend(cramer_recs)

        # 3.3 多特征组合推荐（数值→分类）
        if eta_target_to_features:
            multi_recs = self._build_multi_feature_classification(
                eta_target_to_features,
                numeric_vars,
                categorical_vars,
                task_type='numeric_to_categorical'
            )
            recommendations.extend(multi_recs)

        # 3.4 多特征组合推荐（分类→分类）
        if cramer_target_to_features:
            multi_recs = self._build_multi_feature_classification(
                cramer_target_to_features,
                numeric_vars,
                categorical_vars,
                task_type='categorical_to_categorical'
            )
            recommendations.extend(multi_recs)

        return recommendations

    def _get_numeric_to_categorical_recs(self, numeric_vars, categorical_vars) -> Tuple[List[Dict], Dict[str, List]]:
        """数值 → 分类：基于 Eta-squared >= 0.1，返回推荐列表和 target->features 映射"""
        recommendations = []
        target_to_features = {}  # {categorical_target: [(numeric_feature, eta_sq), ...]}

        if not numeric_vars or not categorical_vars:
            return recommendations, target_to_features

        for num_var in numeric_vars:
            if num_var not in self.data.columns:
                continue

            for cat_var in categorical_vars:
                if cat_var not in self.data.columns:
                    continue

                # 排除派生列作为目标
                if self._should_exclude_target(cat_var):
                    continue

                # 过滤同源派生列配对
                if self._should_exclude_pair(num_var, cat_var):
                    continue

                # 计算 Eta-squared
                groups = [self.data[self.data[cat_var] == name][num_var].dropna()
                          for name in self.data[cat_var].unique()
                          if len(self.data[self.data[cat_var] == name]) > 1]
                groups = [g for g in groups if len(g) > 1]

                if len(groups) < 2:
                    continue

                all_values = self.data[num_var].dropna()
                ss_between = sum(len(g) * (g.mean() - all_values.mean()) ** 2 for g in groups)
                ss_total = sum((all_values - all_values.mean()) ** 2)
                eta_sq = ss_between / ss_total if ss_total > 0 else 0

                if eta_sq < 0.1:  # 阈值 0.1
                    continue

                n_classes = self.data[cat_var].nunique()
                strength = "强" if eta_sq > 0.2 else "中"

                recommendations.append({
                    "task_type": "📊 分类预测",
                    "title": f"预测 {cat_var}（基于 {num_var}）",
                    "description": f"「{cat_var}」的组间差异可解释「{num_var}」{eta_sq * 100:.1f}% 的变异",
                    "target": cat_var,
                    "features": [num_var],
                    "traditional": "逻辑回归" if n_classes == 2 else "多项逻辑回归",
                    "ml": "随机森林 / XGBoost / LightGBM / CatBoost",
                    "dl": "MLP / TabNet",
                    "llm": "TabLLM / GPT-4 with few-shot",
                    "reason": f"Eta-squared = {eta_sq:.3f}（{strength}关联），{num_var} 对 {cat_var} 有显著区分能力",
                    "caution": "⚠️ 可尝试加入更多特征提升效果"
                })

                # 记录用于多特征组合
                if cat_var not in target_to_features:
                    target_to_features[cat_var] = []
                target_to_features[cat_var].append((num_var, eta_sq))

        return recommendations, target_to_features

    def _get_categorical_to_categorical_recs(self, categorical_vars) -> Tuple[List[Dict], Dict[str, List]]:
        """分类 → 分类：基于 Cramer's V >= 0.3 且 p<0.05，返回推荐列表和 target->features 映射"""
        from scipy.stats import chi2_contingency

        recommendations = []
        target_to_features = {}  # {target: [(feature, cramer_v), ...]}

        if len(categorical_vars) < 2:
            return recommendations, target_to_features

        processed_pairs = set()

        for i in range(len(categorical_vars)):
            for j in range(i + 1, len(categorical_vars)):
                var1 = categorical_vars[i]
                var2 = categorical_vars[j]
                pair_key = tuple(sorted([var1, var2]))

                if pair_key in processed_pairs:
                    continue

                # 排除派生列作为目标
                target_candidate1 = None if self._should_exclude_target(var1) else var1
                target_candidate2 = None if self._should_exclude_target(var2) else var2

                # 如果两个都是派生列，跳过
                if target_candidate1 is None and target_candidate2 is None:
                    continue

                # 过滤同源派生列配对
                if self._should_exclude_pair(var1, var2):
                    continue

                if var1 not in self.data.columns or var2 not in self.data.columns:
                    continue

                try:
                    crosstab = pd.crosstab(self.data[var1], self.data[var2])
                    if crosstab.shape[0] <= 1 or crosstab.shape[1] <= 1:
                        continue

                    chi2, p, dof, expected = chi2_contingency(crosstab)

                    if p >= 0.05:
                        continue

                    n = len(self.data)
                    min_dim = min(crosstab.shape) - 1
                    cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

                    if cramer_v < 0.3:  # 阈值 0.3
                        continue

                    processed_pairs.add(pair_key)

                    strength = "强" if cramer_v > 0.5 else "中"
                    n_classes1 = self.data[var1].nunique()
                    n_classes2 = self.data[var2].nunique()

                    # 优先选择非派生列作为目标
                    if target_candidate1 is not None and target_candidate2 is not None:
                        # 两者都是非派生列，选择类别数较少的
                        if n_classes1 <= n_classes2:
                            target = var1
                            feature = var2
                        else:
                            target = var2
                            feature = var1
                    elif target_candidate1 is not None:
                        target = var1
                        feature = var2
                    else:
                        target = var2
                        feature = var1

                    recommendations.append({
                        "task_type": "📊 分类预测",
                        "title": f"预测 {target}（基于 {feature}）",
                        "description": f"「{target}」与「{feature}」呈{strength}关联（Cramer's V={cramer_v:.3f}）",
                        "target": target,
                        "features": [feature],
                        "traditional": "逻辑回归" if n_classes1 == 2 or n_classes2 == 2 else "多项逻辑回归",
                        "ml": "随机森林 / XGBoost / LightGBM / CatBoost",
                        "dl": "MLP / TabNet",
                        "llm": "TabLLM / GPT-4 with few-shot",
                        "reason": f"Cramer's V={cramer_v:.3f}，p={p:.4f}，两变量显著相关",
                        "caution": "⚠️ 可尝试加入更多特征提升效果"
                    })

                    # 记录用于多特征组合（只记录非派生列作为目标）
                    if not self._should_exclude_target(target):
                        if target not in target_to_features:
                            target_to_features[target] = []
                        target_to_features[target].append((feature, cramer_v))

                except Exception:
                    continue

        return recommendations, target_to_features

    def _build_multi_feature_classification(self, target_to_features: Dict[str, List],
                                              numeric_vars: List,
                                              categorical_vars: List,
                                              task_type: str = 'numeric_to_categorical') -> List[Dict]:
        """构建分类多特征组合推荐（只针对非派生列目标）"""
        recommendations = []

        for target, features in target_to_features.items():
            # 排除派生列作为目标
            if self._should_exclude_target(target):
                continue

            # 按效应量排序
            features_sorted = sorted(features, key=lambda x: x[1], reverse=True)

            # 收集所有特征（不限制数量），排除派生列
            all_features = []
            for f, _ in features_sorted:
                if not self._is_derived_column(f):
                    all_features.append(f)

            # 去重去冗余
            all_features = self._deduplicate_features(all_features, target, task_type='classification')

            if len(all_features) < 2:
                continue

            n_classes = self.data[target].nunique() if target in self.data.columns else 2

            # 生成原因描述（全部列出）
            reason_parts = []
            metric_name = "η²" if task_type == 'numeric_to_categorical' else "V"
            for f, val in features_sorted:
                if not self._is_derived_column(f):
                    reason_parts.append(f"{f}({metric_name}={val:.3f})")
            reason_str = "、".join(reason_parts)

            recommendations.append({
                "task_type": "📊 分类预测",
                "title": f"预测 {target}（多特征组合）",
                "description": f"基于 {len(all_features)} 个特征预测「{target}」",
                "target": target,
                "features": all_features,
                "traditional": "逻辑回归" if n_classes == 2 else "多项逻辑回归",
                "ml": "随机森林 / XGBoost / LightGBM / CatBoost",
                "dl": "MLP / TabNet",
                "llm": "TabLLM / GPT-4 with few-shot",
                "reason": f"关联特征：{reason_str}",
                "caution": "⚠️ 建议先进行特征选择；⚠️ 注意类别不平衡问题"
            })

        return recommendations

    def _deduplicate_features(self, features: List[str], target: str, task_type: str = 'regression') -> List[str]:
        """去重去冗余：同源派生列去重，高相关特征去冗余"""
        if len(features) <= 1:
            return features

        # 1. 同源派生列去重
        # 优先级：原列 > 月份 > 季度 > 年 > 周 > 日
        priority = {'': 0, 'month': 1, 'quarter': 2, 'year': 3, 'week': 4, 'day': 5, 'weekday': 6, 'is_weekend': 7}

        source_groups = {}  # {original_col: [(feature, priority)]}

        for f in features:
            original = self.date_column_mapping.get(f, f)
            # 计算优先级
            p = 99
            for suffix, pri in priority.items():
                if suffix and f.endswith(f'_{suffix}'):
                    p = pri
                    break
                elif not suffix and original == f:
                    p = pri
                    break
            if original not in source_groups:
                source_groups[original] = []
            source_groups[original].append((f, p))

        deduped = []
        for original, group in source_groups.items():
            # 按优先级排序，取优先级最高的
            group.sort(key=lambda x: x[1])
            deduped.append(group[0][0])

        if len(deduped) <= 1:
            return deduped

        # 2. 高相关特征去冗余（特征间相关系数 > 0.7，只保留与目标相关性更高的）
        if task_type == 'regression' and target in self.data.columns:
            # 计算每个特征与目标的相关性
            target_corr = {}
            for f in deduped:
                if f in self.data.columns and self.data[f].dtype in ['int64', 'float64']:
                    valid_data = self.data[[target, f]].dropna()
                    if len(valid_data) > 0:
                        target_corr[f] = abs(valid_data.corr().iloc[0, 1])
                    else:
                        target_corr[f] = 0
                else:
                    target_corr[f] = 0

            # 检查特征间相关性
            to_remove = set()
            for i in range(len(deduped)):
                for j in range(i + 1, len(deduped)):
                    f1, f2 = deduped[i], deduped[j]
                    if f1 in to_remove or f2 in to_remove:
                        continue
                    # 计算特征间相关性
                    if f1 in self.data.columns and f2 in self.data.columns:
                        if self.data[f1].dtype in ['int64', 'float64'] and self.data[f2].dtype in ['int64', 'float64']:
                            valid_data = self.data[[f1, f2]].dropna()
                            if len(valid_data) > 0:
                                corr = abs(valid_data.corr().iloc[0, 1])
                                if corr > 0.7:
                                    # 保留与目标相关性更高的
                                    if target_corr.get(f1, 0) < target_corr.get(f2, 0):
                                        to_remove.add(f1)
                                    else:
                                        to_remove.add(f2)

            deduped = [f for f in deduped if f not in to_remove]

        return deduped

    def _check_normality(self, x):
        """检查正态性"""
        from scipy.stats import shapiro, normaltest

        x = x.dropna()
        if len(x) < 8:
            return False, 1.0, {'skew': 0, 'kurtosis': 0}
        if len(x) > 5000:
            x = x.sample(n=5000, random_state=42)

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

        is_normal = (p_value > 0.05 and skewness < 2.0 and kurtosis < 7.0)

        return is_normal, p_value, {'skew': skewness, 'kurtosis': kurtosis}

    # ==================== 4. 聚类分析 ====================
    def _get_clustering_recommendations(self, numeric_vars) -> Optional[Dict]:
        """获取聚类分析推荐（≥3个数值变量）"""
        if len(numeric_vars) < 3:
            return None

        n_samples = len(self.data)
        if n_samples < 100:
            return None

        # 检查数值变量是否有足够变异
        valid_vars = []
        for col in numeric_vars:
            if col in self.data.columns and self.data[col].std() > 0:
                valid_vars.append(col)

        if len(valid_vars) < 3:
            return None

        return {
            "task_type": "🔘 聚类分析",
            "title": f"基于{len(valid_vars)}个数值指标进行分群",
            "description": f"数据包含 {len(valid_vars)} 个数值指标，样本量 {n_samples}，适合进行聚类分析",
            "features": valid_vars,
            "traditional": "K-Means / 层次聚类",
            "ml": "DBSCAN / Gaussian Mixture / BIRCH",
            "dl": "Deep Clustering / DEC",
            "llm": "LLM+Embedding + 聚类",
            "reason": f"{len(valid_vars)}个数值特征，{n_samples}个样本，可识别用户/患者分群",
            "caution": "⚠️ 建议先标准化数据；⚠️ 使用肘部法则确定K值"
        }

    # ==================== 5. 关联规则挖掘 ====================
    def _get_association_recommendations(self, categorical_vars) -> Optional[Dict]:
        """获取关联规则挖掘推荐（≥3个分类变量）"""
        if len(categorical_vars) < 3:
            return None

        # 过滤掉类别数过多的变量
        valid_vars = []
        for col in categorical_vars:
            if col in self.data.columns:
                n_unique = self.data[col].nunique()
                if 2 <= n_unique <= 20:
                    valid_vars.append(col)

        if len(valid_vars) < 3:
            return None

        return {
            "task_type": "🔗 关联规则挖掘",
            "title": f"发现{len(valid_vars)}个分类变量间的关联模式",
            "description": f"数据包含 {len(valid_vars)} 个分类变量，可挖掘频繁项集和关联规则",
            "features": valid_vars,
            "traditional": "Apriori / FP-Growth",
            "ml": "ECLAT / 关联规则 + 聚类",
            "dl": "-",
            "llm": "LLM辅助规则解读",
            "reason": f"{len(valid_vars)}个分类变量，可发现「如果A则B」的关联模式",
            "caution": "⚠️ 建议设置最小支持度≥0.01；⚠️ 注意区分相关性和因果性"
        }

    # ==================== 6. 异常检测 ====================
    def _get_anomaly_detection_recommendations(self) -> Optional[Dict]:
        """获取异常检测推荐（存在异常值）"""
        outliers = self.quality_report.get('outliers', {})

        if not outliers:
            return None

        outlier_cols = list(outliers.keys())
        outlier_info = []
        for col in outlier_cols[:5]:
            info = outliers[col]
            outlier_info.append(f"{col}({info.get('percent', 0):.1f}%)")

        return {
            "task_type": "🚨 异常检测",
            "title": f"检测{len(outlier_cols)}个字段的异常值",
            "description": f"发现异常值字段：{', '.join(outlier_info)}",
            "features": outlier_cols[:10],
            "traditional": "IQR / Z-Score / 箱线图",
            "ml": "Isolation Forest / One-Class SVM / DBSCAN",
            "dl": "Autoencoder / GAN",
            "llm": "LLM辅助异常解释",
            "reason": f"共{len(outlier_cols)}个字段存在异常值，最高比例{max([o.get('percent', 0) for o in outliers.values()]):.1f}%",
            "caution": "⚠️ 需判断异常值是数据错误还是真实异常；⚠️ 建议先验证数据准确性"
        }

    def _get_model_recommendations(self, numeric_vars, categorical_vars, datetime_vars) -> List[Dict]:
        """供 reporter.py 调用的模型推荐（兼容原有格式）"""
        all_recs = self._get_all_recommendations(numeric_vars, categorical_vars, datetime_vars)
        all_recs = self._sort_recommendations(all_recs)

        # 转换为 reporter.py 期望的格式
        formatted = []
        for rec in all_recs:
            formatted.append({
                "priority": self._get_priority(rec.get("task_type", ""), rec),
                "task_type": rec.get("title", rec.get("task_type", "")),
                "target_column": rec.get("target", ""),
                "feature_columns": rec.get("features", []),
                "traditional": rec.get("traditional", ""),
                "ml": rec.get("ml", ""),
                "dl": rec.get("dl", ""),
                "llm": rec.get("llm", ""),
                "reason": rec.get("reason", ""),
                "caution": rec.get("caution", "")
            })

        return formatted

    def _get_priority(self, task_type: str, rec: Dict = None) -> str:
        """根据任务类型返回优先级"""
        if "时间序列预测" in task_type:
            return "高"
        elif "回归预测" in task_type and rec and "多特征" in rec.get('title', ''):
            return "高"
        elif "回归预测" in task_type:
            return "高"
        elif "分类预测" in task_type and rec and "多特征" in rec.get('title', ''):
            return "高"
        elif "分类预测" in task_type:
            return "中"
        elif "聚类分析" in task_type:
            return "中"
        elif "关联规则挖掘" in task_type:
            return "低"
        elif "异常检测" in task_type:
            return "低"
        else:
            return "中"