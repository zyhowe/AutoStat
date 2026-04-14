"""
多表智能统计分析器模块
自动识别表间关系并整合分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Optional, Any
import warnings

from autostat.analyzer import AutoStatisticalAnalyzer
from autostat.loader import DataLoader
from autostat.core.base import BaseAnalyzer

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class MultiTableStatisticalAnalyzer:
    """多表智能统计分析器 - 自动识别表间关系并整合分析"""

    def __init__(self, tables: Dict[str, pd.DataFrame], relationships: Optional[Dict] = None,
                 date_features_level: str = "basic", predefined_types: Dict[str, Dict[str, str]] = None):
        """
        初始化多表分析器

        参数:
        - tables: 表名字典，值必须是 DataFrame
        - relationships: 用户定义的表间关系（可选）
        - date_features_level: 日期派生列级别
        - predefined_types: 每个表的预定义类型 {表名: {列名: 类型}}
        """
        # 验证输入
        if not tables:
            raise ValueError("tables 不能为空")

        for name, data in tables.items():
            if data is None:
                raise ValueError(f"表 {name} 的数据为 None")
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"表 {name} 的数据必须是 DataFrame，请先使用 DataLoader 加载")
            if data.empty:
                print(f"⚠️ 警告: 表 {name} 为空，将被忽略")
                continue

        # 过滤空表
        self.tables = {name: df for name, df in tables.items() if df is not None and not df.empty}

        if not self.tables:
            raise ValueError("没有有效的表数据")

        self.date_features_level = date_features_level
        self.table_names = list(self.tables.keys())
        self.relationships = relationships or {}
        self.predefined_types = predefined_types or {}  # 新增

        self.table_variable_types = {}
        self.table_quality_reports = {}
        self.table_date_derived_columns = {}
        self.table_date_original_columns = {}
        self.table_date_column_mapping = {}

        print("\n" + "=" * 80)
        print("📊 多表智能统计分析系统 (增强版) - 初始化阶段")
        print("=" * 80)

        self._analyze_all_tables(quiet=True)

        self.discovered_relationships = self._discover_relationships()
        self.all_relationships = self._merge_relationships()
        self._print_discovered_relationships()

        self.relationship_graph = self._build_relationship_graph()
        if len(self.relationship_graph.nodes) > 0:
            self.connected_components = list(nx.connected_components(self.relationship_graph.to_undirected()))
        else:
            self.connected_components = []
        self.table_groups = self._identify_table_groups()

        print("\n✅ 初始化完成，发现 {} 个关联表组，{} 个独立表".format(
            len([g for g in self.table_groups if g['type'] == 'related']),
            len([g for g in self.table_groups if g['type'] == 'independent'])
        ))

    @classmethod
    def from_files(cls, table_files: Dict[str, str], relationships: Optional[Dict] = None, **kwargs):
        """从文件路径加载表并创建分析器"""
        tables = {}
        print("\n📁 从文件加载表...")
        print(f"{'─' * 50}")

        for name, file_path in table_files.items():
            print(f"  加载表 {name}: {file_path}")
            try:
                df = DataLoader.load_from_file(file_path, **kwargs)
                if df is not None and not df.empty:
                    tables[name] = df
                    print(f"    ✅ 成功: {len(df)}行 x {len(df.columns)}列")
                else:
                    print(f"    ⚠️ 文件为空: {file_path}")
            except Exception as e:
                print(f"    ❌ 失败: {e}")

        if not tables:
            raise ValueError("没有成功加载任何表")

        return cls(tables, relationships)

    @classmethod
    def from_json_strings(cls, json_strings: Dict[str, str], relationships: Optional[Dict] = None,
                          parse_dates=True, date_columns=None, **kwargs):
        """从JSON字符串加载表并创建分析器"""
        tables = {}
        print("\n📄 从JSON字符串加载表...")
        print(f"{'─' * 50}")

        for name, json_str in json_strings.items():
            print(f"  加载表 {name}")
            try:
                df = DataLoader.load_json_string(json_str, parse_dates=parse_dates, date_columns=date_columns, **kwargs)
                if df is not None and not df.empty:
                    tables[name] = df
                    print(f"    ✅ 成功: {len(df)}行 x {len(df.columns)}列")
                else:
                    print(f"    ⚠️ JSON为空")
            except Exception as e:
                print(f"    ❌ 失败: {e}")

        if not tables:
            raise ValueError("没有成功加载任何表")

        return cls(tables, relationships)

    def _analyze_all_tables(self, quiet=False):
        """分析所有表：推断类型并进行数据质量体检"""
        for table_name, df in self.tables.items():
            if not quiet:
                print(f"\n【分析表: {table_name}】")
                print(f"{'─' * 50}")

            # 获取该表的预定义类型
            table_predefined_types = self.predefined_types.get(table_name, None)
            # Web模式：如果有预定义类型，跳过自动识别
            skip_auto = table_predefined_types is not None

            analyzer = AutoStatisticalAnalyzer(
                df,
                source_table_name=table_name,
                auto_clean=False,
                quiet=quiet,
                date_features_level=self.date_features_level,
                predefined_types=table_predefined_types,
                skip_auto_inference=skip_auto
            )

            if not quiet:
                print(f"\n📈 对 {table_name} 表进行时间序列分析...")
                analyzer.auto_time_series_analysis(max_numeric=5)

            self.table_variable_types[table_name] = {
                'types': analyzer.variable_types,
                'reasons': analyzer.type_reasons
            }

            if hasattr(analyzer, 'date_derived_columns'):
                self.table_date_derived_columns[table_name] = analyzer.date_derived_columns
            else:
                self.table_date_derived_columns[table_name] = set()

            if hasattr(analyzer, 'date_original_columns'):
                self.table_date_original_columns[table_name] = analyzer.date_original_columns
            else:
                self.table_date_original_columns[table_name] = set()

            if hasattr(analyzer, 'date_column_mapping'):
                self.table_date_column_mapping[table_name] = analyzer.date_column_mapping
            else:
                self.table_date_column_mapping[table_name] = {}

            self.table_quality_reports[table_name] = {
                'quality_report': getattr(analyzer, 'quality_report', {}),
                'cleaning_suggestions': getattr(analyzer, 'cleaning_suggestions', []),
                'type_inference_warnings': getattr(analyzer, 'type_inference_warnings', {}),
                'date_features': getattr(analyzer, 'date_features', {})
            }

            if quiet:
                missing_count = len(analyzer.quality_report.get('missing', []))
                outlier_count = len(analyzer.quality_report.get('outliers', {}))
                dup_count = analyzer.quality_report.get('duplicates', {}).get('count', 0)
                date_count = len([col for col, typ in analyzer.variable_types.items() if typ == 'datetime'])

                issues = []
                if missing_count > 0:
                    issues.append(f"缺失{missing_count}列")
                if outlier_count > 0:
                    issues.append(f"异常{outlier_count}列")
                if dup_count > 0:
                    issues.append(f"重复{dup_count}条")

                status = "✅ 质量良好" if not issues else f"⚠️ {', '.join(issues)}"
                print(f"  {table_name}: {status} (日期列: {date_count}个)")

    def _discover_relationships(self) -> Dict:
        """自动发现表间关系 - 只发现用户未定义的关系"""
        discovered = {'foreign_keys': []}

        user_defined_pairs = set()
        if self.relationships and 'foreign_keys' in self.relationships:
            for fk in self.relationships['foreign_keys']:
                pair = tuple(sorted([fk['from_table'], fk['to_table']]))
                user_defined_pairs.add(pair)

        print("\n🔍 自动发现表间关系...")

        EXCLUDE_AS_FOREIGN_KEY = {'id', '_id', '主键', 'pk'}

        all_columns = {}
        for table_name in self.table_names:
            df = self.tables[table_name]
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in EXCLUDE_AS_FOREIGN_KEY:
                    continue
                if col not in all_columns:
                    all_columns[col] = []
                all_columns[col].append(table_name)

        common_columns = {col: tables for col, tables in all_columns.items() if len(tables) >= 2}

        if common_columns:
            valid_columns = {}
            for col, tables_list in common_columns.items():
                has_new = False
                for i, table1 in enumerate(tables_list):
                    for table2 in tables_list[i + 1:]:
                        pair = tuple(sorted([table1, table2]))
                        if pair not in user_defined_pairs:
                            has_new = True
                            break
                    if has_new:
                        break

                if has_new:
                    valid_columns[col] = tables_list

            if valid_columns:
                print(f"\n  📋 发现共同字段名: {', '.join(list(valid_columns.keys()))}")
            else:
                print(f"\n  ℹ️ 共同字段名均已由用户定义，无新关系")
                return discovered
        else:
            print(f"  📋 未发现共同字段名")
            return discovered

        for col, tables_list in valid_columns.items():
            for i, table1 in enumerate(tables_list):
                for table2 in tables_list[i + 1:]:
                    pair = tuple(sorted([table1, table2]))
                    if pair in user_defined_pairs:
                        continue

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
                        elif unique1 == total1 and unique2 == total2:
                            rel_type = 'one_to_one'
                        else:
                            rel_type = 'many_to_many'

                        total_unique = min(df1[col].nunique(), df2[col].nunique())
                        confidence = len(common_values) / total_unique if total_unique > 0 else 0

                        discovered['foreign_keys'].append({
                            'from_table': table1,
                            'from_col': col,
                            'to_table': table2,
                            'to_col': col,
                            'type': rel_type,
                            'confidence': confidence,
                            'auto_discovered': True
                        })

                        print(f"  🔗 自动发现: {table1}.{col} ↔ {table2}.{col} (置信度: {confidence:.1%})")

        return discovered

    def _merge_relationships(self) -> Dict:
        """合并用户定义和自动发现的关系（用户定义优先）"""
        merged = {'foreign_keys': []}
        user_defined_keys = set()

        if self.relationships and 'foreign_keys' in self.relationships:
            for fk in self.relationships['foreign_keys']:
                merged['foreign_keys'].append(fk)
                key = f"{fk['from_table']}-{fk['from_col']}-{fk['to_table']}-{fk['to_col']}"
                user_defined_keys.add(key)
                key_reverse = f"{fk['to_table']}-{fk['to_col']}-{fk['from_table']}-{fk['from_col']}"
                user_defined_keys.add(key_reverse)
            print(f"\n📌 使用用户定义关系: {len(self.relationships['foreign_keys'])} 个")

        if self.discovered_relationships:
            auto_added = 0
            for fk in self.discovered_relationships['foreign_keys']:
                key = f"{fk['from_table']}-{fk['from_col']}-{fk['to_table']}-{fk['to_col']}"
                if key not in user_defined_keys:
                    merged['foreign_keys'].append(fk)
                    auto_added += 1
            if auto_added > 0:
                print(f"🔍 自动发现补充关系: {auto_added} 个")

        return merged

    def _print_discovered_relationships(self):
        """打印发现的关系"""
        print("\n🔍 发现的表间关系:")
        if self.all_relationships['foreign_keys']:
            print("\n【外键关联】")
            for fk in self.all_relationships['foreign_keys']:
                confidence = fk.get('confidence', 1.0)
                rel_type = fk.get('type', 'unknown')
                auto_flag = " [自动发现]" if fk.get('auto_discovered', False) else " [用户定义]"
                conf_str = f" (置信度: {confidence:.1%})" if confidence < 1.0 else ""
                print(f"  {fk['from_table']}.{fk['from_col']} → {fk['to_table']}.{fk['to_col']} "
                      f"[类型: {rel_type}]{auto_flag}{conf_str}")
        else:
            print("  未发现表间关系")

    def _build_relationship_graph(self) -> nx.DiGraph:
        """构建关系有向图"""
        G = nx.DiGraph()
        for table_name in self.table_names:
            G.add_node(table_name, type='table')
        for fk in self.all_relationships.get('foreign_keys', []):
            from_table = fk.get('from_table')
            to_table = fk.get('to_table')
            if from_table and to_table and from_table in self.table_names and to_table in self.table_names:
                G.add_edge(from_table, to_table,
                           from_col=fk.get('from_col'),
                           to_col=fk.get('to_col'),
                           rel_type=fk.get('type', 'unknown'),
                           confidence=fk.get('confidence', 1.0))
        return G

    def _identify_table_groups(self) -> List[Dict]:
        """识别表组（关联表组和独立表）"""
        groups = []

        if self.connected_components:
            for component in self.connected_components:
                if len(component) > 1:
                    group = {
                        'type': 'related',
                        'tables': list(component),
                        'relationships': []
                    }
                    for fk in self.all_relationships.get('foreign_keys', []):
                        if (fk.get('from_table') in component and
                                fk.get('to_table') in component):
                            group['relationships'].append(fk)
                    groups.append(group)

        all_related_tables = set()
        for group in groups:
            all_related_tables.update(group['tables'])

        independent_tables = set(self.table_names) - all_related_tables
        for table in independent_tables:
            groups.append({
                'type': 'independent',
                'tables': [table],
                'relationships': []
            })

        return groups

    def _merge_related_tables(self, table_group: List[str], relationships: List[Dict]) -> pd.DataFrame:
        """合并关联的表组 - 使用左连接保留所有原始数据"""
        if len(table_group) == 1:
            return self.tables[table_group[0]].copy()

        table_degree = {table: 0 for table in table_group}
        for rel in relationships:
            table_degree[rel['from_table']] += 1
            table_degree[rel['to_table']] += 1

        main_table = max(table_degree.items(), key=lambda x: x[1])[0]
        print(f"\n  主表选择: {main_table}")

        merged_df = self.tables[main_table].copy()
        merged_tables = {main_table}

        while len(merged_tables) < len(table_group):
            merged_before = len(merged_tables)

            for table in table_group:
                if table in merged_tables:
                    continue

                join_info = None
                for rel in relationships:
                    if rel['from_table'] == table and rel['to_table'] in merged_tables:
                        join_info = {
                            'left_table': rel['to_table'],
                            'left_col': rel['to_col'].lower(),
                            'right_table': rel['from_table'],
                            'right_col': rel['from_col'].lower()
                        }
                        break
                    elif rel['to_table'] == table and rel['from_table'] in merged_tables:
                        join_info = {
                            'left_table': rel['from_table'],
                            'left_col': rel['from_col'].lower(),
                            'right_table': rel['to_table'],
                            'right_col': rel['to_col'].lower()
                        }
                        break

                if join_info:
                    df_to_merge = self.tables[table].copy()

                    if join_info['left_col'] not in merged_df.columns:
                        print(f"    ⚠️ 警告: {join_info['left_table']} 表中不存在列 {join_info['left_col']}")
                        continue

                    if join_info['right_col'] not in df_to_merge.columns:
                        print(f"    ⚠️ 警告: {table} 表中不存在列 {join_info['right_col']}")
                        continue

                    left_vals = set(merged_df[join_info['left_col']].dropna().unique())
                    right_vals = set(df_to_merge[join_info['right_col']].dropna().unique())
                    common_vals = left_vals.intersection(right_vals)

                    if len(common_vals) == 0:
                        print(f"    ⚠️ 警告: 无共同值，跳过连接")
                        continue

                    print(f"    共同值数量: {len(common_vals)}")

                    merged_df = pd.merge(
                        merged_df, df_to_merge,
                        left_on=join_info['left_col'],
                        right_on=join_info['right_col'],
                        how='left',
                        suffixes=('', f'_{table}')
                    )
                    print(f"    左连接: {join_info['left_table']}.{join_info['left_col']} ← {table}.{join_info['right_col']}")
                    merged_tables.add(table)

            if len(merged_tables) == merged_before:
                print(f"   ⚠️ 无法合并剩余表: {set(table_group) - merged_tables}")
                break

        return merged_df

    def _filter_metadata_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤掉元数据字段"""
        metadata_cols = [col for col in df.columns if col.startswith('_')]
        if metadata_cols:
            print(f"   🏷️ 自动排除元数据字段: {len(metadata_cols)}个")
        return df.drop(columns=metadata_cols, errors='ignore')

    def _show_relationships(self):
        """显示发现的表间关系"""
        print("\n🔍 发现的表间关系:")
        if self.all_relationships['foreign_keys']:
            print("\n【外键关联】")
            for fk in self.all_relationships['foreign_keys']:
                confidence = fk.get('confidence', 1.0)
                rel_type = fk.get('type', 'unknown')
                auto_flag = " [自动发现]" if fk.get('auto_discovered', False) else " [用户定义]"
                conf_str = f" (置信度: {confidence:.1%})" if confidence < 1.0 else ""
                print(f"  {fk.get('from_table')}.{fk.get('from_col')} → "
                      f"{fk.get('to_table')}.{fk.get('to_col')} "
                      f"[类型: {rel_type}]{auto_flag}{conf_str}")

        self._plot_relationship_graph()
        plt.show(block=False)
        plt.pause(0.5)

    def _plot_relationship_graph(self):
        """绘制表间关系图"""
        if len(self.relationship_graph.nodes) == 0:
            print("  无节点，跳过关系图绘制")
            return

        plt.figure(figsize=(12, 8))
        try:
            pos = nx.spring_layout(self.relationship_graph, k=2, iterations=50)
        except:
            pos = nx.random_layout(self.relationship_graph)

        edges = self.relationship_graph.edges(data=True)
        edge_colors = []
        edge_widths = []

        for u, v, data in edges:
            confidence = data.get('confidence', 0.5)
            edge_colors.append((0, 0, 0, confidence))
            edge_widths.append(1 + confidence * 3)

        nx.draw_networkx_nodes(self.relationship_graph, pos,
                               node_color='lightblue',
                               node_size=3000,
                               node_shape='s')

        nx.draw_networkx_edges(self.relationship_graph, pos,
                               edge_color=edge_colors,
                               width=edge_widths,
                               arrows=True,
                               arrowsize=20)

        nx.draw_networkx_labels(self.relationship_graph, pos,
                                font_size=10,
                                font_weight='bold')

        edge_labels = {}
        for u, v, data in self.relationship_graph.edges(data=True):
            edge_labels[(u, v)] = f"{data.get('from_col', '')}→{data.get('to_col', '')}"

        nx.draw_networkx_edge_labels(self.relationship_graph, pos,
                                     edge_labels=edge_labels,
                                     font_size=8)

        plt.title("表间关系图谱 (颜色越深置信度越高)", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def _get_predefined_types_for_merge(self, table_group: List[str], merged_df: pd.DataFrame) -> Dict:
        """获取合并表所需的预定义类型"""
        predefined_types = {}
        actual_columns = set(merged_df.columns)

        for table_name in table_group:
            if table_name in self.table_variable_types:
                table_types = self.table_variable_types[table_name]['types']
                for col, var_type in table_types.items():
                    if col in actual_columns:
                        if self._is_id_column(col):
                            predefined_types[col] = 'identifier'
                        else:
                            predefined_types[col] = var_type

        return predefined_types

    def _is_id_column(self, col_name: str) -> bool:
        """判断是否是ID列"""
        col_lower = col_name.lower()
        id_keywords = ['id', '_id', '编号', '编码', 'code', 'key']
        return any(kw in col_lower or col_lower.endswith(kw) for kw in id_keywords)

    def _get_date_info_for_merge(self, table_group: List[str]) -> Dict:
        """获取合并表所需的日期派生列信息"""
        date_info = {
            'date_original_columns': set(),
            'date_derived_columns': set(),
            'date_column_mapping': {}
        }

        for table_name in table_group:
            if table_name in self.table_date_original_columns:
                date_info['date_original_columns'].update(self.table_date_original_columns[table_name])
            if table_name in self.table_date_derived_columns:
                date_info['date_derived_columns'].update(self.table_date_derived_columns[table_name])
            if table_name in self.table_date_column_mapping:
                date_info['date_column_mapping'].update(self.table_date_column_mapping[table_name])

        return date_info

    def get_merged_analyzer(self):
        """获取合并后的主表分析器（用于生成完整报告）"""
        # 找出最大的表作为主表
        main_table_name = max(self.tables.keys(), key=lambda x: len(self.tables[x]))

        # 获取与该表关联的其他表
        related_tables = [main_table_name]
        for group in self.table_groups:
            if group['type'] == 'related' and main_table_name in group['tables']:
                related_tables = group['tables']
                break

        # 合并关联表
        if len(related_tables) > 1:
            group_relationships = []
            for group in self.table_groups:
                if group['type'] == 'related' and main_table_name in group['tables']:
                    group_relationships = group['relationships']
                    break
            merged_df = self._merge_related_tables(related_tables, group_relationships)
            merged_df_clean = self._filter_metadata_columns(merged_df)

            predefined_types = self._get_predefined_types_for_merge(related_tables, merged_df_clean)
            date_info = self._get_date_info_for_merge(related_tables)

            analyzer = AutoStatisticalAnalyzer(
                merged_df_clean,
                source_table_name=f"merged_{'_'.join(related_tables)}",
                predefined_types=predefined_types,
                auto_clean=False,
                quiet=True
            )

            if hasattr(analyzer, 'date_derived_columns'):
                analyzer.date_derived_columns = date_info['date_derived_columns']
            if hasattr(analyzer, 'date_original_columns'):
                analyzer.date_original_columns = date_info['date_original_columns']
            if hasattr(analyzer, 'date_column_mapping'):
                analyzer.date_column_mapping = date_info['date_column_mapping']

            # 执行完整分析，填充 time_series_diagnostics 等
            analyzer.generate_full_report()

            return analyzer
        else:
            analyzer = AutoStatisticalAnalyzer(
                self.tables[main_table_name],
                source_table_name=main_table_name,
                auto_clean=False,
                quiet=True
            )

            # 执行完整分析，填充 time_series_diagnostics 等
            analyzer.generate_full_report()

            return analyzer

    def to_html(self, output_file=None, title="多表分析报告"):
        """生成多表分析的HTML报告（使用合并表的完整分析）"""
        merged_analyzer = self.get_merged_analyzer()

        from autostat.reporter import Reporter
        reporter = Reporter(merged_analyzer)

        # 生成合并表的完整报告
        base_html = reporter.to_html(title=title)

        # 添加多表信息
        relationships = self.all_relationships.get('foreign_keys', [])
        table_groups_html = f"""
        <div class="card">
            <h2>🔗 多表关联信息</h2>
            <h3>表间关系</h3>
            <table>
                <thead><tr><th>源表</th><th>源列</th><th>目标表</th><th>目标列</th><th>关系类型</th></tr></thead>
                <tbody>
        """
        for rel in relationships:
            table_groups_html += f"""
                <tr>
                    <td>{rel.get('from_table', '-')}</td>
                    <td>{rel.get('from_col', '-')}</td>
                    <td>{rel.get('to_table', '-')}</td>
                    <td>{rel.get('to_col', '-')}</td>
                    <td>{rel.get('type', 'unknown')}</td>
                </tr>
            """
        table_groups_html += """
                </tbody>
            </table>
            <h3>表组信息</h3>
            <table>
                <thead><tr><th>类型</th><th>包含的表</th><th>关系数量</th></tr></thead>
                <tbody>
        """
        for group in self.table_groups:
            if group['type'] == 'related':
                table_groups_html += f"<tr><td colspan=\"3\">🔗 关联表组: {', '.join(group['tables'])} ({len(group['relationships'])}个关系)</td></tr>"
            else:
                table_groups_html += f"<tr><td colspan=\"3\">📄 独立表: {group['tables'][0]}</td></tr>"
        table_groups_html += """
                </tbody>
            </table>
        </div>
        """

        # 将多表信息插入到报告开头
        final_html = base_html.replace('<div class="summary">', '<div class="summary">' + table_groups_html)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_html)
            print(f"✅ HTML报告已保存到 {output_file}")

        return final_html

    def to_json(self, output_file=None, indent=2, ensure_ascii=False):
        """
        将多表分析结果转换为JSON格式

        返回合并表的完整分析结果（与单表分析一致），同时包含多表关系信息
        """
        import json
        from datetime import datetime

        # 获取合并表的完整分析结果
        merged_analyzer = self.get_merged_analyzer()
        merged_result = json.loads(merged_analyzer.to_json())

        # 添加多表特有的信息
        merged_result['multi_table_info'] = {
            'tables': {},
            'relationships': self.all_relationships.get('foreign_keys', []),
            'table_groups': self.table_groups
        }

        # 添加各表的概要信息
        for table_name, df in self.tables.items():
            # 获取该表的变量类型
            var_types = {}
            if table_name in self.table_variable_types:
                for col, typ in self.table_variable_types[table_name]['types'].items():
                    var_types[col] = {
                        'type': typ,
                        'type_desc': BaseAnalyzer.get_type_description(typ)
                    }

            merged_result['multi_table_info']['tables'][table_name] = {
                'shape': {
                    'rows': len(df),
                    'columns': len(df.columns)
                },
                'column_names': list(df.columns),
                'variable_types': var_types,
                'quality_summary': self.table_quality_reports.get(table_name, {}).get('quality_report', {})
            }

        json_str = json.dumps(merged_result, indent=indent, ensure_ascii=ensure_ascii, default=str)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"✅ 多表分析结果已保存为JSON: {output_file}")
            return output_file
        return json_str

    def analyze_all_tables(self):
        """分析所有表（增强版）"""
        print("\n" + "=" * 80)
        print("📊 多表智能统计分析系统 (增强版) - 分析阶段")
        print("=" * 80)

        self._show_relationships()

        independent_groups = [g for g in self.table_groups if g['type'] == 'independent']
        if independent_groups:
            print("\n" + "=" * 80)
            print("📦 独立表分析")
            print("=" * 80)

            for group in independent_groups:
                for table_name in group['tables']:
                    print(f"\n{'─' * 50}")
                    print(f"📋 独立表: {table_name}")
                    print(f"{'─' * 50}")

                    analyzer = AutoStatisticalAnalyzer(
                        self.tables[table_name],
                        source_table_name=table_name,
                        auto_clean=False,
                        quiet=False
                    )
                    analyzer.generate_full_report()

                    plt.show(block=False)
                    plt.pause(0.5)

        related_groups = [g for g in self.table_groups if g['type'] == 'related']
        if related_groups:
            print("\n" + "=" * 80)
            print("🔗 关联表组分析")
            print("=" * 80)

            for i, group in enumerate(related_groups, 1):
                print(f"\n{'=' * 60}")
                print(f"📦 关联表组 {i}: {', '.join(group['tables'])}")
                print(f"关系数量: {len(group['relationships'])}")
                print(f"{'=' * 60}")

                merged_df = self._merge_related_tables(group['tables'], group['relationships'])
                print(f"\n🔗 合并后数据形状: {merged_df.shape}")

                merged_df_clean = self._filter_metadata_columns(merged_df)
                print(f"过滤元数据后字段数: {len(merged_df_clean.columns)}")

                predefined_types = self._get_predefined_types_for_merge(group['tables'], merged_df_clean)
                date_info = self._get_date_info_for_merge(group['tables'])

                print("\n📋 从源表继承的类型:")
                type_count_display = 0
                for col, var_type in predefined_types.items():
                    if col in merged_df_clean.columns:
                        type_desc = BaseAnalyzer.get_type_description(var_type)
                        if type_count_display < 10:
                            print(f"  {col}: {type_desc}")
                        type_count_display += 1
                if type_count_display > 10:
                    print(f"  ... 还有{type_count_display - 10}列")

                print(f"\n【合并表完整分析】")
                merged_analyzer = AutoStatisticalAnalyzer(
                    merged_df_clean,
                    source_table_name=f"merged_{'_'.join(group['tables'])}",
                    predefined_types=predefined_types,
                    auto_clean=False,
                    quiet=False
                )

                if hasattr(merged_analyzer, 'date_derived_columns'):
                    merged_analyzer.date_derived_columns = date_info['date_derived_columns']
                if hasattr(merged_analyzer, 'date_original_columns'):
                    merged_analyzer.date_original_columns = date_info['date_original_columns']
                if hasattr(merged_analyzer, 'date_column_mapping'):
                    merged_analyzer.date_column_mapping = date_info['date_column_mapping']

                print(f"\n{'=' * 70}")
                print(f"📊 开始合并表完整分析")
                print(f"{'=' * 70}")

                merged_analyzer.generate_full_report(show_outlier_details=True)

                plt.show(block=False)
                plt.pause(0.5)

                print(f"\n{'=' * 70}")
                print(f"【🔗 跨表关联特定推荐】")
                print(f"{'=' * 70}")
                print(f"  此表组包含 {len(group['tables'])} 个关联表，可考虑:")
                print(f"  • 构建星型模型或雪花模型进行数据仓库分析")
                print(f"  • 使用关系图神经网络(R-GCN)进行多表联合建模")
                print(f"  • 进行跨表特征工程，如聚合其他表的统计特征")
                print(f"  • 构建知识图谱进行关系挖掘")
                print()