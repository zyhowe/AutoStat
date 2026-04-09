"""
统一绘图模块 - 所有图表生成逻辑的唯一来源
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
import seaborn as sns
from typing import Optional, Any
from scipy.stats import chi2_contingency


class PlotGenerator:
    """统一图表生成器"""

    @staticmethod
    def categorical(data: pd.Series, col: str, buf: Optional[Any] = None):
        """
        生成分类变量图表（条形图+饼图）

        参数:
        - data: 数据列
        - col: 列名
        - buf: 缓冲区，为None时显示，否则保存到缓冲区
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        value_counts = data.value_counts()
        value_counts.plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
        axes[0].set_title(f'{col} - 条形图')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('频数')
        axes[0].tick_params(axis='x', rotation=45)

        if len(value_counts) > 5:
            plot_data = value_counts.head()
            other_sum = value_counts[5:].sum()
            if other_sum > 0:
                plot_data['其他'] = other_sum
        else:
            plot_data = value_counts

        plot_data.plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
        axes[1].set_title(f'{col} - 饼图')
        axes[1].set_ylabel('')

        plt.tight_layout()

        if buf:
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def continuous(data: pd.Series, col: str, buf: Optional[Any] = None):
        """
        生成连续变量图表（直方图+箱线图）

        参数:
        - data: 数据列
        - col: 列名
        - buf: 缓冲区，为None时显示，否则保存到缓冲区
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(data, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].axvline(data.mean(), color='red', linestyle='--', label=f"均值={data.mean():.2f}")
        axes[0].axvline(data.median(), color='green', linestyle='--', label=f"中位数={data.median():.2f}")
        axes[0].set_title(f'{col} - 直方图 (偏度={data.skew():.2f})')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('频数')
        axes[0].legend()

        axes[1].boxplot(data, vert=True)
        axes[1].set_title(f'{col} - 箱线图')
        axes[1].set_ylabel(col)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if buf:
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def timeseries(series: pd.Series, col: str, buf: Optional[Any] = None):
        """
        生成时间序列图表（4合1：原始序列、滚动统计、自相关图、分布直方图）

        参数:
        - series: 时间序列数据（已按日期索引）
        - col: 列名
        - buf: 缓冲区，为None时显示，否则保存到缓冲区
        """
        if len(series) < 10:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'样本量不足（{len(series)}个时间点）',
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title(f'{col} - 时间序列分析')
            plt.tight_layout()
            if buf:
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{col} - 时间序列分析', fontsize=14, fontweight='bold')

        # 1. 原始时间序列
        axes[0, 0].plot(series.index, series.values, marker='.', linestyle='-',
                       linewidth=1, color='blue', markersize=2)
        axes[0, 0].set_title('原始时间序列')
        axes[0, 0].set_xlabel('日期')
        axes[0, 0].set_ylabel(col)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 滚动统计
        window = min(7, max(2, len(series) // 10))
        if window >= 2:
            rolling_mean = series.rolling(window=window, min_periods=1).mean()
            rolling_std = series.rolling(window=window, min_periods=1).std()
            axes[0, 1].plot(series.index, series.values, alpha=0.5, label='原始',
                           color='blue', linewidth=0.5)
            axes[0, 1].plot(series.index, rolling_mean, 'r-',
                           label=f'{window}期移动平均', linewidth=2)
            axes[0, 1].fill_between(series.index,
                                    rolling_mean - rolling_std,
                                    rolling_mean + rolling_std,
                                    alpha=0.2, color='red', label='±1 标准差')
            axes[0, 1].set_title(f'滚动统计 (窗口={window})')
            axes[0, 1].set_xlabel('日期')
            axes[0, 1].set_ylabel(col)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, '样本量不足', ha='center', va='center',
                           transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('滚动统计 (样本不足)')

        # 3. 自相关图
        if len(series) > 10:
            autocorrelation_plot(series, ax=axes[1, 0])
            axes[1, 0].set_title('自相关图 (ACF)')
            axes[1, 0].set_xlabel('滞后')
            axes[1, 0].set_ylabel('自相关系数')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(y=0, linestyle='--', color='gray', alpha=0.5)
            bound = 1.96 / np.sqrt(len(series))
            axes[1, 0].axhline(y=bound, linestyle='--', color='red', alpha=0.5)
            axes[1, 0].axhline(y=-bound, linestyle='--', color='red', alpha=0.5)
        else:
            axes[1, 0].text(0.5, 0.5, '样本量不足', ha='center', va='center',
                           transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('自相关图 (样本不足)')

        # 4. 分布直方图
        axes[1, 1].hist(series.dropna(), bins=min(30, max(5, len(series) // 5)),
                       edgecolor='black', alpha=0.7, color='skyblue')
        axes[1, 1].axvline(series.mean(), color='red', linestyle='--',
                          label=f'均值={series.mean():.2f}', linewidth=2)
        axes[1, 1].axvline(series.median(), color='green', linestyle='--',
                          label=f'中位数={series.median():.2f}', linewidth=2)
        axes[1, 1].set_title(f'{col} 分布')
        axes[1, 1].set_xlabel(col)
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if buf:
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def correlation(data: pd.DataFrame, numeric_vars: list, buf: Optional[Any] = None):
        """
        生成数值变量相关性热力图

        参数:
        - data: 数据框
        - numeric_vars: 数值变量列表
        - buf: 缓冲区，为None时显示，否则保存到缓冲区
        """
        if len(numeric_vars) < 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, '数值变量不足2个，无法生成相关性热力图',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('数值变量相关性矩阵')
            plt.tight_layout()
            if buf:
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
            return

        corr_data = data[numeric_vars].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('数值变量相关性矩阵')
        plt.tight_layout()

        if buf:
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def categorical_correlation(data: pd.DataFrame, categorical_vars: list, buf: Optional[Any] = None):
        """
        生成分类变量关联热力图 (Cramer's V)

        参数:
        - data: 数据框
        - categorical_vars: 分类变量列表
        - buf: 缓冲区，为None时显示，否则保存到缓冲区
        """
        if len(categorical_vars) < 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, '分类变量不足2个，无法生成关联热力图',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('分类变量关联矩阵 (Cramer\'s V)')
            plt.tight_layout()
            if buf:
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
            return

        n = len(categorical_vars)
        cramer_matrix = pd.DataFrame(index=categorical_vars, columns=categorical_vars, dtype=float)

        for i in range(n):
            for j in range(n):
                if i == j:
                    cramer_matrix.iloc[i, j] = 1.0
                else:
                    crosstab = pd.crosstab(data[categorical_vars[i]], data[categorical_vars[j]])
                    chi2, p, dof, expected = chi2_contingency(crosstab)
                    min_dim = min(crosstab.shape) - 1
                    cramer_v = np.sqrt(chi2 / (len(data) * min_dim)) if min_dim > 0 else 0
                    cramer_matrix.iloc[i, j] = cramer_v

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cramer_matrix, annot=True, cmap='YlOrRd', center=0.5,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('分类变量关联矩阵 (Cramer\'s V)')
        plt.tight_layout()

        if buf:
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def numeric_categorical_eta(data: pd.DataFrame, numeric_vars: list, categorical_vars: list, buf: Optional[Any] = None):
        """
        生成数值-分类变量关联热力图 (Eta-squared)

        参数:
        - data: 数据框
        - numeric_vars: 数值变量列表
        - categorical_vars: 分类变量列表
        - buf: 缓冲区，为None时显示，否则保存到缓冲区
        """
        if not numeric_vars or not categorical_vars:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, '数值变量或分类变量不足，无法生成关联热力图',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('数值-分类变量关联矩阵 (Eta-squared)')
            plt.tight_layout()
            if buf:
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
            return

        eta_matrix = pd.DataFrame(index=numeric_vars, columns=categorical_vars, dtype=float)

        for num_var in numeric_vars:
            for cat_var in categorical_vars:
                groups = [data[data[cat_var] == name][num_var].dropna()
                          for name in data[cat_var].unique() if len(data[data[cat_var] == name]) > 1]
                groups = [g for g in groups if len(g) > 1]

                if len(groups) >= 2:
                    all_values = data[num_var].dropna()
                    ss_between = sum(len(g) * (g.mean() - all_values.mean()) ** 2 for g in groups)
                    ss_total = sum((all_values - all_values.mean()) ** 2)
                    eta_sq = ss_between / ss_total if ss_total > 0 else 0
                    eta_matrix.loc[num_var, cat_var] = eta_sq
                else:
                    eta_matrix.loc[num_var, cat_var] = np.nan

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(eta_matrix, annot=True, cmap='viridis', center=0.5,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('数值-分类变量关联矩阵 (Eta-squared)')
        plt.tight_layout()

        if buf:
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


# 便捷函数
def plot_categorical(data, col, buf=None):
    return PlotGenerator.categorical(data, col, buf)

def plot_continuous(data, col, buf=None):
    return PlotGenerator.continuous(data, col, buf)

def plot_timeseries(series, col, buf=None):
    return PlotGenerator.timeseries(series, col, buf)

def plot_correlation(data, numeric_vars, buf=None):
    return PlotGenerator.correlation(data, numeric_vars, buf)

def plot_categorical_correlation(data, categorical_vars, buf=None):
    return PlotGenerator.categorical_correlation(data, categorical_vars, buf)

def plot_numeric_categorical_eta(data, numeric_vars, categorical_vars, buf=None):
    return PlotGenerator.numeric_categorical_eta(data, numeric_vars, categorical_vars, buf)