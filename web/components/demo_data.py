"""
演示数据组件 - 提供内置示例数据
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# 演示数据集配置
DEMO_DATASETS = {
    "sales": {
        "name": "销售分析",
        "icon": "📊",
        "color": "#1f77b4",
        "description": "零售销售数据，可用于销售额预测",
        "rows": 5000,
        "cols": 12,
        "use_case": "预测销售额",
        "features": {
            "numeric": ["销售额", "销量", "单价", "折扣", "成本"],
            "categorical": ["产品类别", "地区", "渠道", "促销活动"],
            "datetime": ["销售日期"]
        },
        "insights": [
            "销售额与促销活动强相关",
            "年底是销售高峰",
            "电子产品利润率最高"
        ]
    },
    "user_behavior": {
        "name": "用户分析",
        "icon": "👥",
        "color": "#2ca02c",
        "description": "用户行为数据，可用于用户分群",
        "rows": 3000,
        "cols": 8,
        "use_case": "用户分群",
        "features": {
            "numeric": ["年龄", "消费金额", "登录次数", "停留时长", "点击次数"],
            "categorical": ["性别", "城市", "会员等级", "设备类型"],
            "datetime": ["注册日期", "最后登录"]
        },
        "insights": [
            "高消费用户集中在25-35岁",
            "iOS用户消费金额更高",
            "周末活跃度降低"
        ]
    },
    "medical": {
        "name": "医疗分析",
        "icon": "🏥",
        "color": "#d62728",
        "description": "患者就诊数据，可用于风险评估",
        "rows": 2000,
        "cols": 15,
        "use_case": "风险评估",
        "features": {
            "numeric": ["年龄", "收缩压", "舒张压", "心率", "血糖", "胆固醇", "住院天数"],
            "categorical": ["性别", "疾病类型", "是否吸烟", "是否饮酒", "家族病史"],
            "datetime": ["就诊日期", "出院日期"]
        },
        "insights": [
            "高血压与年龄正相关",
            "吸烟者心血管风险更高",
            "冬季就诊人数增加"
        ]
    }
}


def generate_demo_data(dataset_key: str) -> pd.DataFrame:
    """生成演示数据"""
    np.random.seed(42)

    if dataset_key == "sales":
        dates = pd.date_range("2023-01-01", periods=5000, freq="D")
        return pd.DataFrame({
            "销售日期": np.random.choice(dates, 5000),
            "产品类别": np.random.choice(["电子产品", "服装", "食品", "家居", "图书"], 5000),
            "地区": np.random.choice(["华东", "华南", "华北", "西南", "西北"], 5000),
            "渠道": np.random.choice(["线上", "线下"], 5000, p=[0.6, 0.4]),
            "促销活动": np.random.choice(["双十一", "618", "平日促销", "无促销"], 5000),
            "销量": np.random.poisson(100, 5000),
            "单价": np.random.uniform(50, 500, 5000).round(2),
            "折扣": np.random.choice([0, 0.05, 0.1, 0.15, 0.2], 5000),
            "成本": np.random.uniform(30, 300, 5000).round(2),
            "利润": np.random.uniform(10, 200, 5000).round(2)
        })

    elif dataset_key == "user_behavior":
        # 生成有季节性的消费金额
        dates = pd.date_range("2020-01-01", periods=3000, freq="D")
        seasonal_effect = [1.2 if d.month in [11, 12] else 0.9 if d.month in [1, 2] else 1.0 for d in dates]

        # 使用列表推导式生成随机天数，避免 numpy 数组类型问题
        random_days = [np.random.randint(1, 365) for _ in range(3000)]

        return pd.DataFrame({
            "用户ID": range(1, 3001),
            "性别": np.random.choice(["男", "女"], 3000, p=[0.48, 0.52]),
            "年龄": np.random.normal(35, 12, 3000).astype(int),
            "城市": np.random.choice(["北京", "上海", "广州", "深圳", "成都", "武汉"], 3000),
            "会员等级": np.random.choice(["普通", "黄金", "铂金", "钻石"], 3000, p=[0.5, 0.3, 0.15, 0.05]),
            "设备类型": np.random.choice(["iOS", "Android", "Web"], 3000, p=[0.35, 0.45, 0.2]),
            "消费金额": (np.random.exponential(500, 3000) * np.array(seasonal_effect)).round(2),
            "登录次数": np.random.poisson(10, 3000),
            "停留时长": np.random.exponential(300, 3000).round(0),
            "点击次数": np.random.poisson(50, 3000),
            "注册日期": dates,
            "最后登录": [dates[i] + pd.Timedelta(days=random_days[i]) for i in range(3000)]
        })

    else:  # medical
        ages = np.random.normal(55, 18, 2000).astype(int)
        ages = np.clip(ages, 18, 100)

        # 年龄与血压相关
        sbp = 110 + (ages - 50) * 0.5 + np.random.normal(0, 15, 2000)
        dbp = 70 + (ages - 50) * 0.3 + np.random.normal(0, 10, 2000)

        # 生成就诊和出院日期
        visit_dates = pd.date_range("2023-01-01", periods=2000, freq="D")
        hospital_days = np.random.poisson(5, 2000)

        return pd.DataFrame({
            "患者ID": range(1, 2001),
            "性别": np.random.choice(["男", "女"], 2000),
            "年龄": ages,
            "疾病类型": np.random.choice(["高血压", "糖尿病", "冠心病", "肺炎", "骨折"], 2000),
            "收缩压": sbp.round(0),
            "舒张压": dbp.round(0),
            "心率": np.random.normal(75, 12, 2000).round(0),
            "血糖": (5.0 + (ages - 50) * 0.02 + np.random.normal(0, 1, 2000)).round(1),
            "胆固醇": (4.5 + (ages - 50) * 0.03 + np.random.normal(0, 1, 2000)).round(1),
            "是否吸烟": np.random.choice(["是", "否"], 2000, p=[0.3, 0.7]),
            "是否饮酒": np.random.choice(["是", "否"], 2000, p=[0.4, 0.6]),
            "家族病史": np.random.choice(["有", "无"], 2000, p=[0.25, 0.75]),
            "住院天数": hospital_days,
            "就诊日期": visit_dates,
            "出院日期": [visit_dates[i] + pd.Timedelta(days=int(hospital_days[i])) for i in range(2000)]
        })


def load_demo_dataset(dataset_key: str) -> Optional[pd.DataFrame]:
    """
    加载演示数据集

    参数:
    - dataset_key: 数据集键名 (sales/user_behavior/medical)

    返回:
    - DataFrame
    """
    return generate_demo_data(dataset_key)


def render_demo_section():
    """渲染演示数据区域"""
    st.markdown("### 🎯 快速体验")
    st.caption("没有数据？试试示例数据，30秒了解AutoStat能做什么")

    cols = st.columns(3)

    for idx, (key, dataset) in enumerate(DEMO_DATASETS.items()):
        with cols[idx]:
            with st.container():
                st.markdown(f"""
                <div style="background: {dataset['color']}10; 
                            border-radius: 12px; 
                            padding: 16px;
                            border: 1px solid {dataset['color']}30;">
                    <div style="font-size: 32px; text-align: center;">{dataset['icon']}</div>
                    <div style="font-weight: bold; text-align: center; margin-top: 8px;">
                        {dataset['name']}
                    </div>
                    <div style="font-size: 12px; color: #666; text-align: center;">
                        {dataset['rows']:,}行 × {dataset['cols']}列
                    </div>
                    <div style="font-size: 12px; color: {dataset['color']}; text-align: center; margin-top: 8px;">
                        🎯 {dataset['use_case']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"使用 {dataset['name']} 数据", key=f"demo_{key}", use_container_width=True):
                    _handle_demo_selection(key, dataset)


def _handle_demo_selection(dataset_key: str, dataset: dict):
    """处理演示数据选择"""
    with st.spinner(f"正在加载 {dataset['name']} 演示数据..."):
        df = load_demo_dataset(dataset_key)

        if df is not None:
            # 存储到 session state
            st.session_state.single_cached_df = df
            st.session_state.single_cached_name = f"{dataset['name']}_demo.csv"
            st.session_state.single_cached_ext = "csv"
            st.session_state.demo_loaded = True
            st.session_state.auto_trigger_analysis = True

            st.success(f"✅ 已加载 {dataset['name']} 数据，共 {len(df)} 行 × {len(df.columns)} 列")

            # 直接触发分析
            from web.services.analysis_service import AnalysisService

            # 获取变量类型（自动推断）
            from web.utils.data_preprocessor import get_default_variable_type
            variable_types = {}
            for col in df.columns:
                var_type = get_default_variable_type(col, df)
                if var_type != "exclude":
                    variable_types[col] = var_type

            # 过滤掉排除的列
            selected_cols = [col for col in df.columns if variable_types.get(col) != "exclude"]
            filtered_df = df[selected_cols].copy()

            # 开始分析
            AnalysisService.analyze_single_file(
                st.session_state.single_cached_name,
                "csv",
                filtered_df,
                variable_types
            )