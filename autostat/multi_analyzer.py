"""
多表智能统计分析器模块
自动识别表间关系并整合分析
"""

import pandas as pd
import networkx as nx
from typing import Dict, Optional

from autostat.analyzer import AutoStatisticalAnalyzer


class MultiTableStatisticalAnalyzer:
    """多表智能统计分析器"""

    def __init__(self, tables: Dict[str, pd.DataFrame], relationships: Optional[Dict] = None,
                 date_features_level: str = "basic"):
        """
        参数:
        - tables: 表名字典，值为DataFrame
        - relationships: 表间关系定义
        - date_features_level: 日期派生级别
        """
        for name, data in tables.items():
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"表 {name} 的数据必须是 DataFrame")

        self.date_features_level = date_features_level
        self.tables = tables
        self.table_names = list(self.tables.keys())
        self.relationships = relationships or {}

        self.table_variable_types = {}
        self.table_quality_reports = {}

        print("\n" + "=" * 80)
        print("📊 多表智能统计分析系统")
        print("=" * 80)

        self._analyze_all_tables(quiet=True)

        # 自动发现关系
        self.discovered_relationships = self._discover_relationships()
        self.all_relationships = self._merge_relationships()

        self._print_discovered_relationships()

        # 构建关系图
        self.relationship_graph = self._build_relationship_graph()
        self.connected_components = list(nx.connected_components(self.relationship_graph.to_undirected()))
        self.table_groups = self._identify_table_groups()

        print(f"\n✅ 发现 {len([g for g in self.table_groups if g['type'] == 'related'])} 个关联表组")

    def _analyze_all_tables(self, quiet=False):
        """分析所有表"""
        for table_name, df in self.tables.items():
            analyzer = AutoStatisticalAnalyzer(
                df, source_table_name=table_name, auto_clean=False,
                quiet=quiet, date_features_level=self.date_features_level
            )
            self.table_variable_types[table_name] = {
                'types': analyzer.variable_types,
                'reasons': analyzer.type_reasons
            }
            self.table_quality_reports[table_name] = {
                'quality_report': getattr(analyzer, 'quality_report', {}),
                'cleaning_suggestions': getattr(analyzer, 'cleaning_suggestions', [])
            }
            if not quiet:
                missing_count = len(analyzer.quality_report.get('missing', []))
                outlier_count = len(analyzer.quality_report.get('outliers', {}))
                dup_count = analyzer.quality_report.get('duplicates', {}).get('count', 0)
                issues = []
                if missing_count > 0:
                    issues.append(f"缺失{missing_count}列")
                if outlier_count > 0:
                    issues.append(f"异常{outlier_count}列")
                if dup_count > 0:
                    issues.append(f"重复{dup_count}条")
                status = "✅" if not issues else f"⚠️ {', '.join(issues)}"
                print(f"  {table_name}: {status}")

    def _discover_relationships(self) -> Dict:
        """自动发现表间关系"""
        discovered = {'foreign_keys': []}

        # 找出相同列名
        all_columns = {}
        for table_name in self.table_names:
            for col in self.tables[table_name].columns:
                if col not in all_columns:
                    all_columns[col] = []
                all_columns[col].append(table_name)

        common_columns = {col: tables for col, tables in all_columns.items() if len(tables) >= 2}

        for col, tables_list in common_columns.items():
            for i, table1 in enumerate(tables_list):
                for table2 in tables_list[i + 1:]:
                    df1 = self.tables[table1]
                    df2 = self.tables[table2]
                    common_values = set(df1[col].dropna().unique()) & set(df2[col].dropna().unique())
                    if len(common_values) > 0:
                        unique1 = df1[col].nunique()
                        unique2 = df2[col].nunique()
                        total1 = len(df1)
                        total2 = len(df2)
                        if unique1 == total1 and unique2 < total2:
                            rel_type = 'one_to_many'
                        elif unique2 == total2 and unique1 < total1:
                            rel_type = 'many_to_one'
                        else:
                            rel_type = 'many_to_many'
                        discovered['foreign_keys'].append({
                            'from_table': table1,
                            'from_col': col,
                            'to_table': table2,
                            'to_col': col,
                            'type': rel_type,
                            'confidence': len(common_values) / min(unique1, unique2),
                            'auto_discovered': True
                        })
        return discovered

    def _merge_relationships(self) -> Dict:
        """合并用户定义和自动发现的关系"""
        merged = {'foreign_keys': []}
        user_defined_keys = set()

        if self.relationships and 'foreign_keys' in self.relationships:
            for fk in self.relationships['foreign_keys']:
                merged['foreign_keys'].append(fk)
                key = f"{fk['from_table']}-{fk['from_col']}-{fk['to_table']}-{fk['to_col']}"
                user_defined_keys.add(key)

        if self.discovered_relationships:
            for fk in self.discovered_relationships['foreign_keys']:
                key = f"{fk['from_table']}-{fk['from_col']}-{fk['to_table']}-{fk['to_col']}"
                if key not in user_defined_keys:
                    merged['foreign_keys'].append(fk)

        return merged

    def _print_discovered_relationships(self):
        """打印发现的关系"""
        print("\n🔍 发现的表间关系:")
        if self.all_relationships['foreign_keys']:
            for fk in self.all_relationships['foreign_keys']:
                auto_flag = " [自动发现]" if fk.get('auto_discovered', False) else ""
                print(f"  {fk['from_table']}.{fk['from_col']} → {fk['to_table']}.{fk['to_col']}{auto_flag}")
        else:
            print("  未发现表间关系")

    def _build_relationship_graph(self) -> nx.DiGraph:
        """构建关系图"""
        G = nx.DiGraph()
        for table_name in self.table_names:
            G.add_node(table_name)
        for fk in self.all_relationships.get('foreign_keys', []):
            G.add_edge(fk['from_table'], fk['to_table'])
        return G

    def _identify_table_groups(self) -> list:
        """识别表组"""
        groups = []
        for component in self.connected_components:
            if len(component) > 1:
                groups.append({'type': 'related', 'tables': list(component), 'relationships': []})
        all_related = set()
        for g in groups:
            all_related.update(g['tables'])
        for table in set(self.table_names) - all_related:
            groups.append({'type': 'independent', 'tables': [table], 'relationships': []})
        return groups

    def analyze_all_tables(self):
        """分析所有表"""
        print("\n" + "=" * 80)
        print("📊 多表分析报告")
        print("=" * 80)

        for group in self.table_groups:
            if group['type'] == 'independent':
                for table_name in group['tables']:
                    print(f"\n📋 独立表: {table_name}")
                    analyzer = AutoStatisticalAnalyzer(
                        self.tables[table_name], source_table_name=table_name, auto_clean=False, quiet=False
                    )
                    analyzer.generate_full_report()
            else:
                print(f"\n🔗 关联表组: {', '.join(group['tables'])}")
                for table_name in group['tables']:
                    print(f"\n  📋 表: {table_name}")
                    analyzer = AutoStatisticalAnalyzer(
                        self.tables[table_name], source_table_name=table_name, auto_clean=False, quiet=True
                    )
                    analyzer.auto_describe()