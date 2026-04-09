"""
单元测试模块
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autostat.analyzer import AutoStatisticalAnalyzer
from autostat.loader import DataLoader
from autostat.checker import ConditionChecker
from autostat.multi_analyzer import MultiTableStatisticalAnalyzer
from autostat.reporter import Reporter


class TestAutoStatisticalAnalyzer(unittest.TestCase):
    """测试单表分析器"""

    def setUp(self):
        """准备测试数据"""
        np.random.seed(42)
        n = 100

        self.test_data = pd.DataFrame({
            'id': range(1, n + 1),
            'name': [f'user_{i}' for i in range(n)],
            'age': np.random.normal(35, 10, n).astype(int),
            'gender': np.random.choice(['M', 'F'], n, p=[0.5, 0.5]),
            'city': np.random.choice(['北京', '上海', '广州', '深圳'], n),
            'salary': np.random.normal(8000, 2000, n).astype(int),
            'score': np.random.uniform(60, 100, n).round(1),
            'date': pd.date_range('2023-01-01', periods=n, freq='D'),
            'is_active': np.random.choice([True, False], n, p=[0.7, 0.3])
        })

        self.test_data.loc[10:15, 'salary'] = np.nan
        self.test_data.loc[20:25, 'city'] = np.nan

        self.temp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_csv.name, index=False)
        self.temp_csv.close()

    def tearDown(self):
        """清理临时文件"""
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)

    def test_init_from_dataframe(self):
        """测试从DataFrame初始化"""
        analyzer = AutoStatisticalAnalyzer(self.test_data, quiet=True)
        self.assertIsNotNone(analyzer.data)
        self.assertEqual(len(analyzer.data), 100)
        self.assertEqual(len(analyzer.data.columns), 9)

    def test_init_from_file(self):
        """测试从文件初始化"""
        analyzer = AutoStatisticalAnalyzer(self.temp_csv.name, quiet=True)
        self.assertIsNotNone(analyzer.data)
        self.assertEqual(len(analyzer.data), 100)

    def test_variable_types(self):
        """测试变量类型识别"""
        analyzer = AutoStatisticalAnalyzer(self.test_data, quiet=True)

        self.assertEqual(analyzer.variable_types.get('id'), 'identifier')
        self.assertEqual(analyzer.variable_types.get('age'), 'continuous')
        self.assertEqual(analyzer.variable_types.get('gender'), 'categorical')
        self.assertEqual(analyzer.variable_types.get('date'), 'datetime')

    def test_quality_report(self):
        """测试质量报告"""
        analyzer = AutoStatisticalAnalyzer(self.test_data, quiet=True)
        quality_report = analyzer.quality_report

        self.assertIn('missing', quality_report)
        self.assertIn('outliers', quality_report)
        self.assertIn('duplicates', quality_report)

        missing_cols = [m['column'] for m in quality_report['missing']]
        self.assertIn('salary', missing_cols)
        self.assertIn('city', missing_cols)

    def test_cleaning_suggestions(self):
        """测试清洗建议生成"""
        analyzer = AutoStatisticalAnalyzer(self.test_data, quiet=True)
        suggestions = analyzer.cleaning_suggestions
        self.assertIsInstance(suggestions, list)

    def test_auto_clean(self):
        """测试自动清洗"""
        analyzer = AutoStatisticalAnalyzer(self.test_data, auto_clean=True, quiet=True)
        self.assertIsNotNone(analyzer.data)

    def test_get_type_description(self):
        """测试类型描述"""
        analyzer = AutoStatisticalAnalyzer(self.test_data, quiet=True)
        self.assertEqual(analyzer._get_type_description('continuous'), '连续变量')
        self.assertEqual(analyzer._get_type_description('categorical'), '分类变量')
        self.assertEqual(analyzer._get_type_description('datetime'), '日期时间变量')

    def test_date_features_extraction(self):
        """测试日期特征提取"""
        analyzer = AutoStatisticalAnalyzer(
            self.test_data,
            quiet=True,
            date_features_level='basic'
        )
        self.assertIn('date_year', analyzer.data.columns)
        self.assertIn('date_month', analyzer.data.columns)


class TestDataLoader(unittest.TestCase):
    """测试数据加载器"""

    def setUp(self):
        """准备测试数据"""
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03']
        })

        self.temp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        self.test_df.to_csv(self.temp_csv.name, index=False)
        self.temp_csv.close()

        self.temp_json = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.test_df.to_json(self.temp_json.name, orient='records', force_ascii=False)
        self.temp_json.close()

    def tearDown(self):
        """清理临时文件"""
        for f in [self.temp_csv.name, self.temp_json.name]:
            if os.path.exists(f):
                os.unlink(f)

    def test_load_csv(self):
        """测试加载CSV"""
        df = DataLoader.load_csv(self.temp_csv.name, parse_dates=False)
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), ['id', 'name', 'date'])

    def test_load_json(self):
        """测试加载JSON"""
        df = DataLoader.load_json(self.temp_json.name)
        self.assertEqual(len(df), 3)

    def test_load_from_file_csv(self):
        """测试自动识别CSV"""
        df = DataLoader.load_from_file(self.temp_csv.name)
        self.assertEqual(len(df), 3)

    def test_load_from_file_json(self):
        """测试自动识别JSON"""
        df = DataLoader.load_from_file(self.temp_json.name)
        self.assertEqual(len(df), 3)

    def test_unsupported_format(self):
        """测试不支持的文件格式"""
        with self.assertRaises(ValueError):
            DataLoader.load_from_file('test.xyz')


class TestConditionChecker(unittest.TestCase):
    """测试条件检查器"""

    def setUp(self):
        """准备测试数据"""
        np.random.seed(42)
        n = 200

        self.data = pd.DataFrame({
            'continuous_norm': np.random.normal(100, 15, n),
            'continuous_skew': np.random.exponential(10, n),
            'categorical': np.random.choice(['A', 'B', 'C'], n, p=[0.5, 0.3, 0.2]),
            'ordinal': np.random.choice([1, 2, 3, 4, 5], n),
            'date': pd.date_range('2023-01-01', periods=n, freq='D')
        })

        self.variable_types = {
            'continuous_norm': 'continuous',
            'continuous_skew': 'continuous',
            'categorical': 'categorical',
            'ordinal': 'ordinal',
            'date': 'datetime'
        }

        self.checker = ConditionChecker(self.data, self.variable_types)

    def test_check_time_series(self):
        """测试时间序列检查"""
        result = self.checker.check_time_series('continuous_norm')
        self.assertIn('suitable', result)
        self.assertIn('method', result)

    def test_check_categorical_relationship(self):
        """测试分类变量关系检查"""
        result = self.checker.check_categorical_relationship('categorical', 'ordinal')
        self.assertIn('suitable', result)

    def test_check_numerical_categorical(self):
        """测试数值-分类关系检查"""
        result = self.checker.check_numerical_categorical('continuous_norm', 'categorical')
        self.assertIn('suitable', result)
        self.assertIn('method', result)

    def test_check_clustering(self):
        """测试聚类检查"""
        result = self.checker.check_clustering(['continuous_norm', 'continuous_skew'])
        self.assertIn('suitable', result)


class TestReporter(unittest.TestCase):
    """测试报告生成器"""

    def setUp(self):
        """准备测试数据"""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'id': range(1, 101),
            'value': np.random.normal(100, 15, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
        self.analyzer = AutoStatisticalAnalyzer(self.test_data, quiet=True)
        self.reporter = Reporter(self.analyzer)

    def test_to_html(self):
        """测试HTML报告生成"""
        html = self.reporter.to_html()
        self.assertIsInstance(html, str)
        self.assertIn('<!DOCTYPE html>', html)

    def test_to_json(self):
        """测试JSON报告生成"""
        json_str = self.reporter.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn('analysis_time', json_str)

    def test_to_markdown(self):
        """测试Markdown报告生成"""
        md = self.reporter.to_markdown()
        self.assertIsInstance(md, str)
        self.assertIn('# 数据分析报告', md)

    def test_to_html_with_file(self):
        """测试保存HTML文件"""
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            self.reporter.to_html(f.name)
            self.assertTrue(os.path.exists(f.name))
            os.unlink(f.name)


class TestMultiTableAnalyzer(unittest.TestCase):
    """测试多表分析器"""

    def setUp(self):
        """准备测试数据"""
        self.users = pd.DataFrame({
            'user_id': range(1, 51),
            'name': [f'user_{i}' for i in range(1, 51)],
            'city': np.random.choice(['北京', '上海', '广州'], 50),
            'level': np.random.choice(['高', '中', '低'], 50)
        })

        self.orders = pd.DataFrame({
            'order_id': range(1, 201),
            'user_id': np.random.choice(range(1, 51), 200),
            'amount': np.random.normal(500, 100, 200).astype(int),
            'order_date': pd.date_range('2023-01-01', periods=200, freq='D')
        })

        self.tables = {
            'users': self.users,
            'orders': self.orders
        }

    def test_init(self):
        """测试初始化"""
        analyzer = MultiTableStatisticalAnalyzer(self.tables)
        self.assertIsNotNone(analyzer.tables)
        self.assertEqual(len(analyzer.tables), 2)

    def test_discover_relationships(self):
        """测试关系发现"""
        analyzer = MultiTableStatisticalAnalyzer(self.tables)
        relationships = analyzer.discovered_relationships
        self.assertIn('foreign_keys', relationships)


if __name__ == '__main__':
    unittest.main()