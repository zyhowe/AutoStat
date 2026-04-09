"""
关系分析模块：相关性矩阵、Cramer's V、Eta-squared、热力图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from typing import Dict, List, Tuple, Optional
import warnings

from autostat.core.plots import plot_correlation

warnings.filterwarnings('ignore')


class RelationshipAnalyzer:
    """关系分析器"""

    def __init__(self, data, variable_types, condition_checker,
                 date_original_columns=None, date_derived_columns=None,
                 date_column_mapping=None, quiet=False):
        self.data = data
        self.variable_types = variable_types
        self.condition_checker = condition_checker
        self.date_original_columns = date_original_columns or set()
        self.date_derived_columns = date_derived_columns or set()
        self.date_column_mapping = date_column_mapping or {}
        self.quiet = quiet

    def auto_analyze_relationships(self):
        """自动关系分析"""
        print("\n" + "=" * 70)
        print("🔗 自动关系分析报告")
        print("=" * 70)

        all_vars = [col for col in self.data.columns
                    if self.variable_types.get(col) not in ['identifier', 'empty']]

        if len(all_vars) < 2:
            print("变量数量不足，无法进行关系分析")
            return

        numeric_vars = [v for v in all_vars if self.variable_types.get(v) == 'continuous']
        categorical_vars = [v for v in all_vars
                            if self.variable_types.get(v) in ['categorical', 'categorical_numeric', 'ordinal']]

        # 排除同源日期派生列的关系
        exclude_pairs = self._build_exclude_pairs(all_vars)
        print(f"\n📌 排除 {len(exclude_pairs)} 对日期相关关系")

        significant_pairs = self._plot_complete_correlation_matrix(all_vars, numeric_vars, categorical_vars, exclude_pairs)

        if significant_pairs:
            self._print_significant_pairs(significant_pairs)
        else:
            print("\n未发现显著的变量关联 (p >= 0.05)")

        return significant_pairs

    def _build_exclude_pairs(self, all_vars):
        """构建需要排除的变量对（同源日期派生列）"""
        exclude_pairs = set()
        for original_col in self.date_original_columns:
            for derived_col in self.date_derived_columns:
                if self.date_column_mapping.get(derived_col) == original_col:
                    if original_col in all_vars and derived_col in all_vars:
                        exclude_pairs.add((original_col, derived_col))
                        exclude_pairs.add((derived_col, original_col))
        for original_col in self.date_original_columns:
            derived_cols = [col for col in self.date_derived_columns
                            if self.date_column_mapping.get(col) == original_col and col in all_vars]
            for i in range(len(derived_cols)):
                for j in range(i + 1, len(derived_cols)):
                    exclude_pairs.add((derived_cols[i], derived_cols[j]))
                    exclude_pairs.add((derived_cols[j], derived_cols[i]))
        return exclude_pairs

    def _plot_complete_correlation_matrix(self, all_vars, numeric_vars, categorical_vars, exclude_pairs):
        """绘制完整相关性矩阵"""
        significant_pairs = []

        # 数值-数值相关性
        if len(numeric_vars) >= 2:
            self._analyze_numeric_numeric(numeric_vars, exclude_pairs, significant_pairs)

        # 分类-分类关联
        if len(categorical_vars) >= 2:
            self._analyze_categorical_categorical(categorical_vars, exclude_pairs, significant_pairs)

        # 数值-分类关联
        if numeric_vars and categorical_vars:
            self._analyze_numeric_categorical(numeric_vars, categorical_vars, exclude_pairs, significant_pairs)

        # 综合热力图 - 使用统一模块
        if len(numeric_vars) >= 2:
            plot_correlation(self.data, numeric_vars)
        else:
            print("\n数值变量不足2个，无法生成相关性热力图")

        return significant_pairs

    def _analyze_numeric_numeric(self, numeric_vars, exclude_pairs, significant_pairs):
        """分析数值-数值相关性"""
        print("\n【数值变量相关系数】")
        valid_numeric = [v for v in numeric_vars if self._has_valid_pair(v, numeric_vars, exclude_pairs)]

        if len(valid_numeric) >= 2:
            corr_data = self.data[valid_numeric].corr()
            print(corr_data.round(4))

            for i in range(len(corr_data.columns)):
                for j in range(i + 1, len(corr_data.columns)):
                    v1, v2 = corr_data.columns[i], corr_data.columns[j]
                    if (v1, v2) in exclude_pairs or (v2, v1) in exclude_pairs:
                        continue
                    corr_value = corr_data.iloc[i, j]
                    if abs(corr_value) > 0.3:
                        valid_data = self.data[[v1, v2]].dropna()
                        if len(valid_data) >= 3:
                            is_norm1, _, _ = self._check_normality(valid_data[v1])
                            is_norm2, _, _ = self._check_normality(valid_data[v2])
                            if is_norm1 and is_norm2:
                                _, p_value = pearsonr(valid_data[v1], valid_data[v2])
                            else:
                                _, p_value = spearmanr(valid_data[v1], valid_data[v2])
                            if p_value < 0.05:
                                significant_pairs.append({
                                    'var1': v1, 'var2': v2, 'type': '数值-数值',
                                    'statistic': corr_value, 'p_value': p_value, 'strength': abs(corr_value)
                                })

    def _analyze_categorical_categorical(self, categorical_vars, exclude_pairs, significant_pairs):
        """分析分类-分类关联"""
        print("\n【分类变量关联矩阵 (Cramer\'s V)】")
        valid_cat = [v for v in categorical_vars if self._has_valid_pair(v, categorical_vars, exclude_pairs)]

        if len(valid_cat) >= 2:
            n_cat = len(valid_cat)
            cramer_matrix = pd.DataFrame(index=valid_cat, columns=valid_cat, dtype=float)

            for i in range(n_cat):
                for j in range(n_cat):
                    if i == j:
                        cramer_matrix.iloc[i, j] = 1.0
                    else:
                        v1, v2 = valid_cat[i], valid_cat[j]
                        if (v1, v2) in exclude_pairs or (v2, v1) in exclude_pairs:
                            cramer_matrix.iloc[i, j] = np.nan
                            continue
                        crosstab = pd.crosstab(self.data[v1], self.data[v2])
                        chi2, p_value, dof, expected = chi2_contingency(crosstab)
                        n = len(self.data)
                        min_dim = min(crosstab.shape) - 1
                        cramer_v = np.sqrt(chi2 / (n * min_dim)) if chi2 > 0 and n > 0 and min_dim > 0 else 0
                        cramer_matrix.iloc[i, j] = cramer_v
                        if p_value < 0.05 and i < j:
                            significant_pairs.append({
                                'var1': v1, 'var2': v2, 'type': '分类-分类',
                                'statistic': cramer_v, 'p_value': p_value, 'strength': cramer_v
                            })

            print(cramer_matrix.round(4))
            plt.figure(figsize=(10, 8))
            sns.heatmap(cramer_matrix, annot=True, cmap='YlOrRd', center=0.5,
                        square=True, linewidths=1, mask=np.triu(np.ones_like(cramer_matrix, dtype=bool), k=1))
            plt.title('分类变量关联矩阵 (Cramer\'s V)')
            plt.tight_layout()
            plt.show()

    def _analyze_numeric_categorical(self, numeric_vars, categorical_vars, exclude_pairs, significant_pairs):
        """分析数值-分类关联"""
        print("\n【数值-分类关联矩阵 (Eta-squared)】")
        eta_matrix = pd.DataFrame(index=numeric_vars, columns=categorical_vars, dtype=float)

        for num_var in numeric_vars:
            for cat_var in categorical_vars:
                if (num_var, cat_var) in exclude_pairs or (cat_var, num_var) in exclude_pairs:
                    eta_matrix.loc[num_var, cat_var] = np.nan
                    continue

                groups = [self.data[self.data[cat_var] == name][num_var].dropna()
                          for name in self.data[cat_var].unique() if len(self.data[self.data[cat_var] == name]) > 1]
                groups = [g for g in groups if len(g) > 1]

                if len(groups) >= 2:
                    all_values = self.data[num_var].dropna()
                    ss_between = sum(len(g) * (g.mean() - all_values.mean()) ** 2 for g in groups)
                    ss_total = sum((all_values - all_values.mean()) ** 2)
                    eta_sq = ss_between / ss_total if ss_total > 0 else 0
                    eta_matrix.loc[num_var, cat_var] = eta_sq
                    if eta_sq > 0.01:
                        significant_pairs.append({
                            'var1': num_var, 'var2': cat_var, 'type': '数值-分类',
                            'statistic': eta_sq, 'p_value': 0.01, 'strength': eta_sq
                        })
                else:
                    eta_matrix.loc[num_var, cat_var] = np.nan

        print(eta_matrix.round(4))
        plt.figure(figsize=(12, 8))
        sns.heatmap(eta_matrix, annot=True, cmap='viridis', center=0.5,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('数值-分类变量关联矩阵 (Eta-squared)')
        plt.tight_layout()
        plt.show()

    def _print_significant_pairs(self, significant_pairs):
        """打印显著关联对"""
        print("\n" + "=" * 80)
        print("📈 发现的显著关联 (按强度排序):")
        print("=" * 80)

        significant_pairs.sort(key=lambda x: x['strength'], reverse=True)
        for i, pair in enumerate(significant_pairs[:20], 1):
            if pair['type'] == '数值-数值':
                strength_desc = "强" if abs(pair['statistic']) > 0.7 else "中" if abs(pair['statistic']) > 0.3 else "弱"
                print(f"\n{i:2d}. {pair['var1']} ↔ {pair['var2']}")
                print(f"     相关系数 r = {pair['statistic']:.4f} ({strength_desc}), p = {pair['p_value']:.4f}")
            else:
                strength_desc = "强" if pair['strength'] > 0.5 else "中" if pair['strength'] > 0.3 else "弱"
                print(f"\n{i:2d}. {pair['var1']} ↔ {pair['var2']}")
                print(f"     效应量 = {pair['statistic']:.4f} ({strength_desc}), p = {pair['p_value']:.4f}")

    def _has_valid_pair(self, col, all_cols, exclude_pairs):
        """检查变量是否有有效配对"""
        for other in all_cols:
            if col != other and (col, other) not in exclude_pairs and (other, col) not in exclude_pairs:
                return True
        return False

    def _check_normality(self, x):
        """检查正态性"""
        from scipy.stats import shapiro
        x = x.dropna()
        if len(x) < 8 or len(x) > 5000:
            return False, 1.0, {}
        try:
            _, p = shapiro(x)
            return p > 0.05, p, {}
        except:
            return False, 0, {}