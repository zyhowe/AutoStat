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
    """分析条件检查器"""

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
            adf_result = adfuller(series, autolag='AIC')
            adf_p = adf_result[1]
            is_stationary = adf_p < 0.05

            max_lag = min(10, len(series) // 5)
            if max_lag < 2:
                return {"suitable": False, "reason": "样本量不足做自相关检验", "method": "均值/中位数"}

            lb_result = acorr_ljungbox(series, lags=[max_lag], return_df=True)
            lb_p = lb_result['lb_pvalue'].iloc[0]
            has_autocorrelation = lb_p < 0.05

            x = np.arange(len(series))
            slope = np.polyfit(x, series, 1)[0] if len(series) > 1 else 0
            has_trend = abs(slope) > 0.01 * series.std() if series.std() > 0 else False

            has_seasonality = False
            if len(series) > 24:
                corr_with_lag12 = series.autocorr(lag=12) if len(series) > 12 else 0
                has_seasonality = abs(corr_with_lag12) > 0.3

            if is_stationary and has_autocorrelation:
                return {"suitable": True, "reason": "平稳且有自相关", "method": "ARIMA/SARIMA"}
            elif not is_stationary and has_autocorrelation:
                return {"suitable": True, "reason": "有自相关但不平稳", "method": "差分后ARIMA"}
            elif is_stationary and not has_autocorrelation:
                return {"suitable": False, "reason": "白噪声", "method": "均值/中位数"}
            else:
                return {"suitable": False, "reason": "随机游走", "method": "简单趋势线"}
        except Exception as e:
            return {"suitable": False, "reason": f"检验失败: {str(e)}", "method": "均值/中位数"}

    def check_categorical_relationship(self, col1, col2):
        """检查分类变量关系分析条件"""
        if self.variable_types.get(col1) not in ['categorical', 'categorical_numeric', 'ordinal'] or \
           self.variable_types.get(col2) not in ['categorical', 'categorical_numeric', 'ordinal']:
            return {"suitable": False, "reason": "非分类变量", "method": None}

        n1, n2 = self.data[col1].nunique(), self.data[col2].nunique()
        if n1 > 50 or n2 > 50:
            return {"suitable": False, "reason": f"类别数过多 ({n1}, {n2})", "method": "降维或合并类别"}

        crosstab = pd.crosstab(self.data[col1], self.data[col2])
        chi2, p, dof, expected = chi2_contingency(crosstab)
        small_expected_ratio = (expected < 5).sum() / expected.size

        if small_expected_ratio > 0.2:
            return {"suitable": True, "reason": "建议用Fisher精确检验", "method": "Fisher精确检验"}
        else:
            return {"suitable": True, "reason": "适合卡方检验", "method": "卡方检验 + Cramer's V"}

    def check_numerical_categorical(self, num_col, cat_col):
        """检查数值-分类关系分析条件"""
        if self.variable_types.get(num_col) != 'continuous':
            return {"suitable": False, "reason": "第一个变量不是数值型", "method": None}
        if self.variable_types.get(cat_col) not in ['categorical', 'categorical_numeric', 'ordinal']:
            return {"suitable": False, "reason": "第二个变量不是分类型", "method": None}

        groups = []
        for name in self.data[cat_col].unique():
            group_data = self.data[self.data[cat_col] == name][num_col].dropna()
            if len(group_data) >= 3:
                groups.append(group_data)

        if len(groups) < 2:
            return {"suitable": False, "reason": "有效分组不足2组", "method": None}

        n_groups = len(groups)
        all_normal = True
        for g in groups:
            if len(g) >= 8:
                _, p_norm = shapiro(g)
                if p_norm <= 0.05:
                    all_normal = False

        if n_groups == 2:
            if all_normal:
                return {"suitable": True, "reason": "满足正态性", "method": "独立样本t检验"}
            else:
                return {"suitable": True, "reason": "不满足正态性", "method": "Mann-Whitney U检验"}
        else:
            if all_normal:
                return {"suitable": True, "reason": "满足正态性", "method": "ANOVA"}
            else:
                return {"suitable": True, "reason": "不满足正态性", "method": "Kruskal-Wallis检验"}

    def check_clustering(self, cols):
        """检查聚类分析条件"""
        numeric_cols = [col for col in cols if self.variable_types.get(col) == 'continuous']
        if len(numeric_cols) < 2:
            return {"suitable": False, "reason": "至少需要2个数值变量", "method": None}

        n = len(self.data)
        d = len(numeric_cols)

        if n < 10:
            return {"suitable": False, "reason": f"样本量太少 (n={n}<10)", "method": None}

        if d > n / 5:
            return {"suitable": True, "reason": f"维度过高", "method": "先PCA降维再聚类"}
        elif n > 10000:
            return {"suitable": True, "reason": "大样本", "method": "Mini-Batch K-Means"}
        elif n > 1000:
            return {"suitable": True, "reason": "中等样本", "method": "K-Means"}
        else:
            return {"suitable": True, "reason": "小样本", "method": "层次聚类"}