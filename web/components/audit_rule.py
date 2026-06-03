# web/components/audit_rule.py
"""勾稽校验组件 - 解读已有规则 / 校验表结构"""

import streamlit as st
import json
from typing import Dict, Any, Optional

from web.services.session_service import SessionService
from web.services.storage_service import StorageService


def format_audit_rules_for_prompt(audit_rules: Dict[str, Any]) -> str:
    """将 audit_rules 格式化为提示词可读的文本"""
    if not audit_rules:
        return "无勾稽规则"

    parts = []

    arithmetic = audit_rules.get('arithmetic_rules', [])
    if arithmetic:
        parts.append("### 数值关系")
        for rule in arithmetic:
            rule_str = rule.get('rule', '')
            conf = rule.get('confidence', 0)
            parts.append(f"- {rule_str} (置信度: {conf:.1%})")

    fd = audit_rules.get('functional_dependencies', [])
    if fd:
        parts.append("\n### 函数依赖")
        for rule in fd:
            parts.append(f"- {rule.get('rule', '')}")

    temporal = audit_rules.get('temporal_rules', [])
    if temporal:
        parts.append("\n### 时序约束")
        for rule in temporal:
            parts.append(f"- {rule.get('rule', '')}")

    fk = audit_rules.get('foreign_keys', [])
    if fk:
        parts.append("\n### 外键约束")
        for rule in fk:
            parts.append(
                f"- {rule.get('from_table', '')}.{rule.get('from_col', '')} → {rule.get('to_table', '')}.{rule.get('to_col', '')}")

    return "\n".join(parts) if parts else "无勾稽规则"


def render_audit_rule_tab():
    """渲染勾稽校验标签页"""
    st.markdown("### 🔗 勾稽校验")
    st.caption("基于数据一致性规则，自动发现和解读字段间的勾稽关系")

    session_id = SessionService.get_current_session()
    json_data = None
    if session_id:
        json_data = StorageService.load_json("analysis_result", session_id)

    mode = st.radio(
        "选择模式",
        options=["📊 解读已有规则", "📝 校验表结构"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.divider()

    if mode == "📊 解读已有规则":
        st.markdown("#### 📊 解读已有勾稽规则")

        if json_data is None:
            st.warning("请先完成数据分析，生成JSON报告")
            return

        audit_rules = json_data.get('quality_report', {}).get('audit_rules', {})

        arithmetic_count = len(audit_rules.get('arithmetic_rules', []))
        fd_count = len(audit_rules.get('functional_dependencies', []))
        temporal_count = len(audit_rules.get('temporal_rules', []))
        fk_count = len(audit_rules.get('foreign_keys', []))

        total_rules = arithmetic_count + fd_count + temporal_count + fk_count

        if total_rules == 0:
            st.info("当前分析结果中没有发现勾稽规则")
            return

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("数值关系", arithmetic_count)
        with col2:
            st.metric("函数依赖", fd_count)
        with col3:
            st.metric("时序约束", temporal_count)
        with col4:
            st.metric("外键约束", fk_count)

        with st.expander("📋 查看规则详情", expanded=False):
            rules_text = format_audit_rules_for_prompt(audit_rules)
            st.markdown(rules_text)

        if st.button("🤖 AI 解读规则", type="primary", use_container_width=True):
            _interpret_existing_rules(audit_rules, rules_text)

    else:
        st.markdown("#### 📝 校验表结构")
        st.caption("输入表结构描述，与数据分析结果中的勾稽规则进行一致性校验")

        if json_data is None:
            st.warning("请先完成数据分析，生成 JSON 报告")
            return

        table_structure = st.text_area(
            "表结构描述",
            height=250,
            placeholder="""示例：
订单表：订单ID、用户ID、订单金额、折扣、实付金额、订单日期
用户表：用户ID、用户名、注册日期

或者 DDL 格式：
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    user_id INT,
    order_amount DECIMAL,
    discount DECIMAL,
    final_amount DECIMAL,
    order_date DATE
);""",
            key="audit_table_structure"
        )

        if st.button("🔍 开始校验", type="primary", use_container_width=True):
            if not table_structure.strip():
                st.warning("请输入表结构描述")
            else:
                _validate_table_structure(table_structure, session_id, json_data)


def _interpret_existing_rules(audit_rules: Dict[str, Any], rules_text: str):
    llm_client = st.session_state.get("llm_client")
    if llm_client is None:
        st.error("请先在侧边栏配置大模型")
        return

    prompt = """你是数据质量专家。请解读以下勾稽规则：

## 现有勾稽规则
""" + rules_text + """

## 任务
请逐条解释每条规则的业务含义，并说明：
1. 这条规则可能代表什么业务逻辑？
2. 如果违反这条规则，可能是什么原因？
3. 如何处理违反规则的数据？

请用中文回答，结构清晰。"""

    messages = [{"role": "user", "content": prompt}]

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            for chunk in llm_client.chat_stream(messages):
                if chunk:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as e:
            response_placeholder.error(f"调用大模型失败: {str(e)}")


def _validate_table_structure(table_structure: str, session_id: Optional[str], json_data: Optional[Dict]):
    llm_client = st.session_state.get("llm_client")
    if llm_client is None:
        st.error("请先在侧边栏配置大模型")
        return

    if json_data is None:
        st.error("请先完成数据分析，生成 JSON 报告")
        return

    audit_rules = json_data.get('quality_report', {}).get('audit_rules', {})
    variable_types = json_data.get('variable_types', {})

    rules_text = format_audit_rules_for_prompt(audit_rules)

    fields_text = ""
    type_display = {
        'continuous': '连续变量',
        'categorical': '分类变量',
        'categorical_numeric': '数值型分类',
        'ordinal': '有序分类',
        'datetime': '日期时间',
        'identifier': '标识符',
        'text': '文本'
    }
    for col, info in variable_types.items():
        typ = info.get('type', 'unknown')
        type_desc = type_display.get(typ, typ)
        fields_text += f"- {col}: {type_desc}\n"

    with st.spinner("正在校验表结构与数据一致性规则..."):
        prompt = """你是财务数据质量专家，精通固定资产会计处理。请基于业务知识，对 JSON 报告中的规则进行合理性校验。

## 重要：等价规则判断标准
两条规则视为等价，当满足以下任一条件：
1. 移项后左右两边完全一致（如 A+B=C 与 C-A-B=0 等价）
2. 加法顺序交换后相同（如 A+B=C 与 B+A=C 等价）
3. 将子表达式移到另一边后相同（如 A+B=C 与 A=C-B 等价）
4. 规则中的项通过加减运算可互相推导（如 A+B=C 与 C-A=B 等价）

**在输出第4部分之前，请先检查 JSON 中是否存在与用户暗示规则等价的规则。若存在，则注明“已存在等价规则，省略”，不要输出。**

## 第一部分：JSON 报告中的实际规则（已规范化）
""" + rules_text + """

## 第二部分：用户输入的表结构描述（业务定义）
""" + table_structure + """

## 任务
请从业务角度，对 JSON 中的规则进行分类判断。**不要复述用户输入的规则列表和表结构描述，直接输出分析结论。**

### 1. 业务上正确且符合会计准则的规则
（这些规则体现了固定资产的勾稽关系，如：期初 + 本期增加 - 本期减少 = 期末）

### 2. 业务上可能正确但需要验证的规则
（这些规则在数学上成立，但业务含义不明确，或只在特定条件下成立）

### 3. 业务上有问题的规则（请详细说明问题）
- 列出规则表达式
- 说明为什么不符合业务逻辑
- 给出正确的业务公式应该是什么

### 4. JSON 规则未覆盖且无等价形式的用户暗示规则
**严格输出要求：**
- 只输出在 JSON 中找不到任何等价形式的用户暗示规则
- 若存在等价形式（移项、组合等），必须跳过并注明“已存在等价规则，省略”
- 每条规则必须附上判断依据（说明为什么 JSON 中没有等价形式）
- 只输出确实缺失且有业务价值的规则

## 输出格式
- 第1-3条使用表格
- 第4条使用列表，每条格式：`- 规则表达式（判断依据：...）`

请开始输出。"""

        messages = [{"role": "user", "content": prompt}]

        full_response = ""
        response_placeholder = st.empty()

        try:
            for chunk in llm_client.chat_stream(messages):
                if chunk:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as e:
            response_placeholder.error(f"调用大模型失败: {str(e)}")