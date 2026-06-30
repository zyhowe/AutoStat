"""
自然语言转SQL模块

将用户的自然语言查询转换为SQL语句
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

class NL2SQL:
    """
    自然语言转SQL

    使用方式:
        nl2sql = NL2SQL(llm_client)
        sql = nl2sql.convert("上个月销售额最高的产品是什么", schema)
        result = nl2sql.execute(sql, df)
    """

    def __init__(self, llm_client=None):
        """
        初始化

        参数:
        - llm_client: 大模型客户端（用于复杂查询）
        """
        self.llm_client = llm_client

    def convert(
        self,
        question: str,
        schema: Dict[str, Any],
        tables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        将自然语言转换为SQL

        参数:
        - question: 自然语言问题
        - schema: 数据模式（字段名、类型、关系）
        - tables: 多表信息（用于JOIN查询）

        返回: {
            "sql": str,
            "parsed": dict,  # 解析后的意图
            "confidence": float
        }
        """
        # 1. 意图解析
        parsed = self._parse_intent(question, schema)

        # 2. 生成SQL
        if self.llm_client and len(schema.get("fields", [])) > 5:
            sql = self._generate_with_llm(question, schema, tables)
        else:
            sql = self._generate_with_rules(parsed, schema, tables)

        return {
            "sql": sql,
            "parsed": parsed,
            "confidence": self._calculate_confidence(parsed, sql)
        }

    def _parse_intent(self, question: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """解析用户意图"""
        question_lower = question.lower()

        # 识别操作类型
        if any(w in question_lower for w in ["查询", "查看", "显示", "列出"]):
            operation = "select"
        elif any(w in question_lower for w in ["统计", "聚合", "汇总", "求和", "平均", "最大值", "最小值"]):
            operation = "aggregate"
        elif any(w in question_lower for w in ["排名", "排序", "最高", "最低", "前", "top"]):
            operation = "rank"
        elif any(w in question_lower for w in ["占比", "比例", "百分比"]):
            operation = "percentage"
        elif any(w in question_lower for w in ["趋势", "变化", "走势"]):
            operation = "trend"
        else:
            operation = "select"

        # 识别时间条件
        time_condition = self._parse_time(question_lower)

        # 识别指标和维度
        metrics = []
        dimensions = []

        # 从schema中匹配
        fields = schema.get("fields", [])
        for field in fields:
            field_name = field.get("name", "")
            field_lower = field_name.lower()
            field_type = field.get("type", "")

            if field_lower in question_lower:
                if field_type in ["continuous", "numeric"]:
                    metrics.append(field_name)
                else:
                    dimensions.append(field_name)

        # 如果没有匹配到，使用默认
        if not metrics:
            for field in fields:
                if field.get("type") in ["continuous", "numeric"]:
                    metrics.append(field.get("name"))
                    break

        if not dimensions:
            for field in fields:
                if field.get("type") not in ["continuous", "numeric", "identifier"]:
                    dimensions.append(field.get("name"))
                    break

        return {
            "operation": operation,
            "metrics": metrics[:3],
            "dimensions": dimensions[:3],
            "time_condition": time_condition,
            "filter": self._parse_filter(question_lower, fields),
            "limit": self._parse_limit(question_lower),
            "order": self._parse_order(question_lower)
        }

    def _parse_time(self, question: str) -> Dict[str, Any]:
        """解析时间条件"""
        result = {"type": "none"}

        if "上个月" in question or "上月" in question:
            now = datetime.now()
            if now.month == 1:
                year, month = now.year - 1, 12
            else:
                year, month = now.year, now.month - 1
            result = {
                "type": "range",
                "start": f"{year}-{month:02d}-01",
                "end": f"{year}-{month:02d}-31"
            }
        elif "本月" in question or "这个月" in question:
            now = datetime.now()
            result = {
                "type": "range",
                "start": f"{now.year}-{now.month:02d}-01",
                "end": now.strftime("%Y-%m-%d")
            }
        elif "去年" in question or "去年" in question:
            year = datetime.now().year - 1
            result = {
                "type": "range",
                "start": f"{year}-01-01",
                "end": f"{year}-12-31"
            }
        elif "最近" in question:
            # 匹配 "最近N天/周/月"
            match = re.search(r"最近(\d+)(天|周|月)", question)
            if match:
                days = int(match.group(1))
                unit = match.group(2)
                if unit == "周":
                    days *= 7
                elif unit == "月":
                    days *= 30
                start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                result = {
                    "type": "range",
                    "start": start,
                    "end": datetime.now().strftime("%Y-%m-%d")
                }
            else:
                result = {"type": "recent", "days": 30}

        return result

    def _parse_filter(self, question: str, fields: List[Dict]) -> Dict[str, Any]:
        """解析过滤条件"""
        # 简单实现：识别 "大于"、"小于"、"等于"
        patterns = [
            (r"大于(\d+)", "gt"),
            (r"超过(\d+)", "gt"),
            (r"小于(\d+)", "lt"),
            (r"低于(\d+)", "lt"),
            (r"等于(\d+)", "eq"),
            (r"为(\w+)", "eq"),
        ]

        for pattern, op in patterns:
            match = re.search(pattern, question)
            if match:
                value = match.group(1)
                # 尝试匹配字段
                for field in fields:
                    if field.get("name").lower() in question:
                        return {
                            "field": field.get("name"),
                            "operator": op,
                            "value": value
                        }

        return {}

    def _parse_limit(self, question: str) -> Optional[int]:
        """解析限制数量"""
        match = re.search(r"前(\d+)", question)
        if match:
            return int(match.group(1))

        if "top" in question.lower():
            match = re.search(r"top\s*(\d+)", question.lower())
            if match:
                return int(match.group(1))

        return None

    def _parse_order(self, question: str) -> Dict[str, str]:
        """解析排序"""
        if "最高" in question or "最大" in question:
            return {"direction": "desc"}
        elif "最低" in question or "最小" in question:
            return {"direction": "asc"}
        return {}

    def _generate_with_rules(
        self,
        parsed: Dict[str, Any],
        schema: Dict[str, Any],
        tables: Optional[Dict[str, Any]] = None
    ) -> str:
        """基于规则生成SQL"""
        table_name = schema.get("table_name", "data")
        fields = schema.get("fields", [])

        # 构建SELECT子句
        select_parts = []

        if parsed["operation"] == "aggregate":
            for metric in parsed["metrics"]:
                select_parts.append(f"SUM({metric}) as total_{metric}")
            for dim in parsed["dimensions"]:
                select_parts.append(dim)
        elif parsed["operation"] == "percentage":
            for metric in parsed["metrics"]:
                select_parts.append(f"{metric} / SUM({metric}) OVER() * 100 as {metric}_pct")
            for dim in parsed["dimensions"]:
                select_parts.append(dim)
        else:
            select_parts = parsed["metrics"] + parsed["dimensions"]
            if not select_parts:
                select_parts = ["*"]

        # 构建WHERE子句
        where_parts = []
        time_cond = parsed.get("time_condition", {})
        if time_cond.get("type") == "range":
            time_field = schema.get("time_field", "date")
            where_parts.append(
                f"{time_field} BETWEEN '{time_cond['start']}' AND '{time_cond['end']}'"
            )

        filter_cond = parsed.get("filter", {})
        if filter_cond:
            field = filter_cond.get("field")
            op = filter_cond.get("operator")
            value = filter_cond.get("value")
            if field and op and value:
                where_parts.append(f"{field} {op} {value}")

        # 构建GROUP BY子句
        group_by_parts = []
        if parsed["operation"] in ["aggregate", "percentage"]:
            group_by_parts = parsed["dimensions"]

        # 构建ORDER BY子句
        order_parts = []
        if parsed.get("order"):
            order_field = parsed["metrics"][0] if parsed["metrics"] else "1"
            direction = parsed["order"].get("direction", "desc")
            order_parts.append(f"{order_field} {direction.upper()}")

        # 构建LIMIT子句
        limit_clause = ""
        if parsed.get("limit"):
            limit_clause = f"LIMIT {parsed['limit']}"

        # 组装SQL
        sql = f"SELECT {', '.join(select_parts)} FROM {table_name}"

        if where_parts:
            sql += f" WHERE {' AND '.join(where_parts)}"

        if group_by_parts:
            sql += f" GROUP BY {', '.join(group_by_parts)}"

        if order_parts:
            sql += f" ORDER BY {', '.join(order_parts)}"

        if limit_clause:
            sql += f" {limit_clause}"

        return sql

    def _generate_with_llm(
        self,
        question: str,
        schema: Dict[str, Any],
        tables: Optional[Dict[str, Any]] = None
    ) -> str:
        """使用大模型生成SQL"""
        if not self.llm_client:
            return self._generate_with_rules(self._parse_intent(question, schema), schema, tables)

        # 构建Schema描述
        schema_desc = f"表名: {schema.get('table_name', 'data')}\n字段:\n"
        for field in schema.get("fields", []):
            schema_desc += f"  - {field.get('name')}: {field.get('type', 'unknown')}\n"

        if tables:
            schema_desc += "\n表间关系:\n"
            for rel in tables.get("relationships", []):
                schema_desc += f"  - {rel.get('from_table')}.{rel.get('from_col')} = {rel.get('to_table')}.{rel.get('to_col')}\n"

        prompt = f"""请根据以下表结构，将用户的自然语言查询转换为SQL语句。

## 表结构
{schema_desc}

## 用户查询
{question}

## 要求
1. 只输出SQL语句，不要其他文字
2. 使用标准的SQL语法
3. 如果涉及多表，使用正确的JOIN语法
4. 添加必要的注释

SQL:"""

        try:
            response = self.llm_client.chat([{"role": "user", "content": prompt}])
            # 提取SQL
            sql = response.strip()
            if "```sql" in sql:
                sql = sql.split("```sql")[1].split("```")[0]
            elif "```" in sql:
                sql = sql.split("```")[1].split("```")[0]
            return sql.strip()
        except Exception as e:
            print(f"LLM SQL生成失败: {e}")
            return self._generate_with_rules(self._parse_intent(question, schema), schema, tables)

    def _calculate_confidence(self, parsed: Dict[str, Any], sql: str) -> float:
        """计算置信度"""
        confidence = 0.5

        # 如果SQL不为空且包含关键要素
        if sql and len(sql) > 10:
            confidence += 0.2

        # 如果解析出指标
        if parsed.get("metrics"):
            confidence += 0.15

        # 如果解析出维度
        if parsed.get("dimensions"):
            confidence += 0.1

        # 如果解析出时间条件
        if parsed.get("time_condition", {}).get("type") != "none":
            confidence += 0.05

        return min(confidence, 1.0)

    def execute(self, sql: str, df) -> Any:
        """
        执行SQL查询（简化实现）

        参数:
        - sql: SQL语句
        - df: 数据框

        返回: 查询结果
        """
        # 这是一个简化实现，实际应该使用 pandasql 或真正的数据库
        # 这里只做简单的过滤和聚合
        try:
            # 尝试使用 pandasql
            from pandasql import sqldf
            return sqldf(sql, locals())
        except ImportError:
            # 降级方案：只支持简单查询
            return self._simple_execute(sql, df)
        except Exception as e:
            print(f"SQL执行失败: {e}")
            return None

    def _simple_execute(self, sql: str, df) -> Any:
        """简化的SQL执行"""
        # 只支持简单的 SELECT ... WHERE ...
        # 实际使用中应该用 pandasql 或 duckdb
        return df.head(100)