"""
条件检查器模块
统一管理各分析的适用条件
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, shapiro, levene, mannwhitneyu, kruskal, f_oneway
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

class ConditionChecker:
    """分析条件检查器 - 统一管理各分析的适用条件"""

    def __init__(self, data, variable_types, date_derived_columns=None, date_column_mapping=None):
        self.data = data
        self.variable_types = variable_types
        self.date_derived_columns = date_derived_columns or set()
        self.date_column_mapping = date_column_mapping or {}

    def check_time_series(self, col, date_col=None):
        """检查时间序列分析条件"""
        if self.variable_types.get(col) != 'continuous':
            return {"suitable": False, "reason": "非数值变量", "method": None}

        series = self.data[col].dropna()
        if len(series) < 30:
            return {"suitable": False, "reason": f"样本量不足 (n={len(series)}<30)", "method": "均值/中位数"}

        try:
            # 1. 平稳性检验 (ADF)
            adf_result = adfuller(series, autolag='AIC')
            adf_p = adf_result[1]
            is_stationary = adf_p < 0.05

            # 2. 自相关性检验 (Ljung-Box)
            max_lag = min(10, len(series) // 5)
            if max_lag < 2:
                return {"suitable": False, "reason": "样本量不足做自相关检验", "method": "均值/中位数"}

            lb_result = acorr_ljungbox(series, lags=[max_lag], return_df=True)
            lb_p = lb_result['lb_pvalue'].iloc[0]
            has_autocorrelation = lb_p < 0.05

            # 3. 趋势性判断
            x = np.arange(len(series))
            slope = np.polyfit(x, series, 1)[0] if len(series) > 1 else 0
            has_trend = abs(slope) > 0.01 * series.std() if series.std() > 0 else False

            # 4. 季节性判断
            has_seasonality = False
            if len(series) > 24:
                corr_with_lag12 = series.autocorr(lag=12) if len(series) > 12 else 0
                has_seasonality = abs(corr_with_lag12) > 0.3

            if is_stationary and has_autocorrelation:
                return {
                    "suitable": True,
                    "reason": "平稳且有自相关",
                    "method": "ARIMA/SARIMA",
                    "details": {
                        "adf_p": adf_p,
                        "lb_p": lb_p,
                        "is_stationary": is_stationary,
                        "has_autocorrelation": has_autocorrelation,
                        "has_trend": has_trend,
                        "has_seasonality": has_seasonality
                    }
                }
            elif not is_stationary and has_autocorrelation:
                return {
                    "suitable": True,
                    "reason": "有自相关但不平稳",
                    "method": "差分后ARIMA",
                    "details": {
                        "adf_p": adf_p,
                        "lb_p": lb_p,
                        "is_stationary": is_stationary,
                        "has_autocorrelation": has_autocorrelation,
                        "has_trend": has_trend,
                        "has_seasonality": has_seasonality,
                        "suggestion": "建议先进行一阶差分"
                    }
                }
            elif is_stationary and not has_autocorrelation:
                return {
                    "suitable": False,
                    "reason": "白噪声 (无自相关)",
                    "method": "均值/中位数",
                    "details": {
                        "adf_p": adf_p,
                        "lb_p": lb_p,
                        "is_stationary": is_stationary,
                        "has_autocorrelation": has_autocorrelation
                    }
                }
            else:
                return {
                    "suitable": False,
                    "reason": "随机游走 (非平稳且无自相关)",
                    "method": "简单趋势线",
                    "details": {
                        "adf_p": adf_p,
                        "lb_p": lb_p,
                        "is_stationary": is_stationary,
                        "has_autocorrelation": has_autocorrelation,
                        "has_trend": has_trend
                    }
                }

        except Exception as e:
            return {"suitable": False, "reason": f"检验失败: {str(e)}", "method": "均值/中位数"}

    def check_categorical_relationship(self, col1, col2):
        """检查分类变量关系分析条件"""
        if self.variable_types.get(col1) not in ['categorical', 'categorical_numeric', 'ordinal'] or \
                self.variable_types.get(col2) not in ['categorical', 'categorical_numeric', 'ordinal']:
            return {"suitable": False, "reason": "非分类变量", "method": None}

        if col1 in self.date_derived_columns and col2 in self.date_derived_columns:
            if self.date_column_mapping.get(col1) == self.date_column_mapping.get(col2):
                return {"suitable": False, "reason": "同一日期源的派生列", "method": None}

        data1 = self.data[col1].dropna()
        data2 = self.data[col2].dropna()

        n = len(self.data)
        n1, n2 = data1.nunique(), data2.nunique()

        if n1 > 50 or n2 > 50:
            return {
                "suitable": False,
                "reason": f"类别数过多 ({n1}, {n2})",
                "method": "降维或合并类别"
            }

        crosstab = pd.crosstab(self.data[col1], self.data[col2])

        chi2, p, dof, expected = chi2_contingency(crosstab)
        small_expected_ratio = (expected < 5).sum() / expected.size

        freq1 = self.data[col1].value_counts(normalize=True)
        freq2 = self.data[col2].value_counts(normalize=True)
        imbalance1 = freq1.max() > 0.9
        imbalance2 = freq2.max() > 0.9

        issues = []
        if n < 100:
            issues.append("小样本")
        if small_expected_ratio > 0.2:
            issues.append(f"期望频数<5的单元格占{small_expected_ratio:.1%}")
        if imbalance1 or imbalance2:
            issues.append("严重不平衡")

        if not issues:
            return {
                "suitable": True,
                "reason": "适合卡方检验",
                "method": "卡方检验 + Cramer's V",
                "details": {
                    "n": n,
                    "n_unique": (n1, n2),
                    "small_expected_ratio": small_expected_ratio
                }
            }
        elif "小样本" in issues or "期望频数" in issues:
            return {
                "suitable": True,
                "reason": "建议用Fisher精确检验",
                "method": "Fisher精确检验",
                "details": {
                    "n": n,
                    "issues": issues
                }
            }
        else:
            return {
                "suitable": True,
                "reason": "结果需谨慎解释",
                "method": "卡方检验 + 注意不平衡",
                "details": {
                    "issues": issues
                }
            }

    def check_numerical_categorical(self, num_col, cat_col):
        """检查数值-分类关系分析条件"""
        if self.variable_types.get(num_col) != 'continuous':
            return {"suitable": False, "reason": "第一个变量不是数值型", "method": None}
        if self.variable_types.get(cat_col) not in ['categorical', 'categorical_numeric', 'ordinal']:
            return {"suitable": False, "reason": "第二个变量不是分类型", "method": None}

        groups = []
        group_names = []
        for name in self.data[cat_col].unique():
            group_data = self.data[self.data[cat_col] == name][num_col].dropna()
            if len(group_data) >= 3:
                groups.append(group_data)
                group_names.append(name)

        if len(groups) < 2:
            return {"suitable": False, "reason": "有效分组不足2组", "method": None}

        n_groups = len(groups)
        min_group_size = min(len(g) for g in groups)

        all_normal = True
        normality_results = []
        for i, g in enumerate(groups):
            if len(g) >= 8:
                _, p_norm = shapiro(g)
                is_normal = p_norm > 0.05
                all_normal = all_normal and is_normal
                normality_results.append({
                    'group': group_names[i],
                    'p_value': p_norm,
                    'is_normal': is_normal
                })
            else:
                all_normal = False
                normality_results.append({
                    'group': group_names[i],
                    'p_value': None,
                    'is_normal': False,
                    'reason': '样本量不足'
                })

        if n_groups == 2 and len(groups[0]) > 1 and len(groups[1]) > 1:
            try:
                _, p_var = levene(groups[0], groups[1])
                equal_var = p_var > 0.05
            except:
                equal_var = False
                p_var = None
        else:
            equal_var = None
            p_var = None

        if n_groups == 2:
            if all_normal and equal_var:
                return {
                    "suitable": True,
                    "reason": "满足正态性和方差齐性",
                    "method": "独立样本t检验",
                    "details": {
                        "n_groups": n_groups,
                        "min_group_size": min_group_size,
                        "all_normal": all_normal,
                        "equal_var": equal_var,
                        "normality": normality_results
                    }
                }
            elif all_normal and not equal_var:
                return {
                    "suitable": True,
                    "reason": "满足正态性但方差不齐",
                    "method": "Welch's t检验",
                    "details": {
                        "n_groups": n_groups,
                        "min_group_size": min_group_size,
                        "all_normal": all_normal,
                        "equal_var": equal_var,
                        "normality": normality_results
                    }
                }
            else:
                return {
                    "suitable": True,
                    "reason": "不满足正态性",
                    "method": "Mann-Whitney U检验",
                    "details": {
                        "n_groups": n_groups,
                        "min_group_size": min_group_size,
                        "all_normal": all_normal,
                        "normality": normality_results
                    }
                }
        else:
            if all_normal:
                return {
                    "suitable": True,
                    "reason": "满足正态性",
                    "method": "ANOVA",
                    "details": {
                        "n_groups": n_groups,
                        "min_group_size": min_group_size,
                        "all_normal": all_normal,
                        "normality": normality_results
                    }
                }
            else:
                return {
                    "suitable": True,
                    "reason": "不满足正态性",
                    "method": "Kruskal-Wallis检验",
                    "details": {
                        "n_groups": n_groups,
                        "min_group_size": min_group_size,
                        "all_normal": all_normal,
                        "normality": normality_results
                    }
                }

    def check_clustering(self, cols):
        """检查聚类分析条件"""
        if len(cols) < 2:
            return {"suitable": False, "reason": "至少需要2个变量", "method": None}

        numeric_cols = [col for col in cols if self.variable_types.get(col) == 'continuous']
        if len(numeric_cols) < 2:
            return {"suitable": False, "reason": "至少需要2个数值变量", "method": None}

        n = len(self.data)
        d = len(numeric_cols)

        if n < 10:
            return {"suitable": False, "reason": f"样本量太少 (n={n}<10)", "method": None}

        outlier_scores = []
        for col in numeric_cols:
            Q1, Q3 = self.data[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            if IQR > 0:
                outliers = ((self.data[col] < Q1 - 1.5 * IQR) |
                            (self.data[col] > Q3 + 1.5 * IQR)).mean()
                outlier_scores.append(outliers)

        avg_outlier_score = np.mean(outlier_scores) if outlier_scores else 0

        if d > n / 5:
            return {
                "suitable": True,
                "reason": f"维度过高 (d={d}, n={n})",
                "method": "先PCA降维再聚类",
                "details": {
                    "n": n,
                    "d": d,
                    "ratio": d / n,
                    "avg_outlier_score": avg_outlier_score
                }
            }

        if avg_outlier_score > 0.1:
            return {
                "suitable": True,
                "reason": f"有离群点 (平均{avg_outlier_score:.1%})",
                "method": "DBSCAN",
                "details": {
                    "n": n,
                    "d": d,
                    "avg_outlier_score": avg_outlier_score
                }
            }
        elif n > 10000:
            return {
                "suitable": True,
                "reason": "大样本",
                "method": "Mini-Batch K-Means",
                "details": {
                    "n": n,
                    "d": d,
                    "avg_outlier_score": avg_outlier_score
                }
            }
        elif n > 1000:
            return {
                "suitable": True,
                "reason": "中等样本",
                "method": "K-Means (用肘部法则选K)",
                "details": {
                    "n": n,
                    "d": d,
                    "avg_outlier_score": avg_outlier_score
                }
            }
        else:
            return {
                "suitable": True,
                "reason": "小样本",
                "method": "层次聚类 (可看树状图)",
                "details": {
                    "n": n,
                    "d": d,
                    "avg_outlier_score": avg_outlier_score
                }
            }

    def check_association_rules(self, cols):
        """检查关联规则挖掘条件"""
        categorical_cols = [col for col in cols
                            if self.variable_types.get(col) in ['categorical', 'categorical_numeric', 'ordinal']]

        if len(categorical_cols) < 2:
            return {"suitable": False, "reason": "至少需要2个分类变量", "method": None}

        n = len(self.data)
        total_items = sum(self.data[col].nunique() for col in categorical_cols)

        if n > 100000:
            return {
                "suitable": True,
                "reason": "大数据量",
                "method": "FP-Growth",
                "details": {
                    "n": n,
                    "n_cols": len(categorical_cols),
                    "total_items": total_items
                }
            }
        elif n > 10000:
            return {
                "suitable": True,
                "reason": "中等数据量",
                "method": "FP-Growth 或 Apriori",
                "details": {
                    "n": n,
                    "n_cols": len(categorical_cols),
                    "total_items": total_items
                }
            }
        elif total_items > 500:
            return {
                "suitable": True,
                "reason": "项数过多",
                "method": "先筛选高频项再用Apriori",
                "details": {
                    "n": n,
                    "n_cols": len(categorical_cols),
                    "total_items": total_items,
                    "suggestion": "建议设置最小支持度>1%"
                }
            }
        else:
            return {
                "suitable": True,
                "reason": "适合关联规则挖掘",
                "method": "Apriori",
                "details": {
                    "n": n,
                    "n_cols": len(categorical_cols),
                    "total_items": total_items
                }
            }

    def recommend_queries(self, cols=None):
        """推荐查询场景"""
        if cols is None:
            cols = list(self.data.columns)

        recommendations = []

        datetime_cols = [col for col in cols if self.variable_types.get(col) == 'datetime']
        for dt_col in datetime_cols[:2]:
            min_date = self.data[dt_col].min()
            max_date = self.data[dt_col].max()
            recommendations.append({
                'type': 'time_range',
                'desc': f"按 {dt_col} 时间范围查询",
                'example': f"{dt_col} BETWEEN '{min_date}' AND '{max_date}'",
                'fields': [dt_col]
            })

        categorical_cols = [col for col in cols
                            if self.variable_types.get(col) in ['categorical', 'categorical_numeric', 'ordinal']
                            and self.data[col].nunique() <= 20]
        for cat_col in categorical_cols[:5]:
            top_values = self.data[cat_col].value_counts().head(3).index.tolist()
            recommendations.append({
                'type': 'category_filter',
                'desc': f"按 {cat_col} 筛选",
                'example': f"{cat_col} IN {top_values}",
                'fields': [cat_col]
            })

        numeric_cols = [col for col in cols if self.variable_types.get(col) == 'continuous']
        for num_col in numeric_cols[:5]:
            q1, q3 = self.data[num_col].quantile([0.25, 0.75])
            recommendations.append({
                'type': 'numeric_range',
                'desc': f"按 {num_col} 范围查询",
                'example': f"{num_col} BETWEEN {q1:.2f} AND {q3:.2f}",
                'fields': [num_col]
            })

        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            recommendations.append({
                'type': 'combined',
                'desc': f"组合查询",
                'example': f"{categorical_cols[0]} = '某类别' AND {numeric_cols[0]} > 平均值",
                'fields': [categorical_cols[0], numeric_cols[0]]
            })

        return recommendations