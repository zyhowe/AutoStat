"""
使用技巧推送 - 根据用户行为推送相关技巧
"""

import streamlit as st
from typing import Dict, Any, List
from datetime import datetime, timedelta


class TipsManager:
    """使用技巧管理器"""

    TIPS = [
        {
            "id": "export",
            "trigger": "view_report",
            "message": "💡 **技巧**：点击右上角下载按钮，可以导出HTML/JSON/Excel格式的报告",
            "action": "点击导出按钮试试看",
            "action_type": "export"
        },
        {
            "id": "multi_table",
            "trigger": "single_analysis",
            "message": "💡 **技巧**：上传多个相关文件，AutoStat会自动发现表间关系并进行关联分析",
            "action": "试试上传订单表和用户表",
            "action_type": "upload_multi"
        },
        {
            "id": "llm",
            "trigger": "no_llm",
            "message": "💡 **技巧**：配置大模型后，可以获得AI智能解读，直接用自然语言提问",
            "action": "在侧边栏配置大模型",
            "action_type": "open_settings"
        },
        {
            "id": "model_training",
            "trigger": "analysis_complete",
            "message": "💡 **技巧**：试试「小模型训练」标签页，基于数据训练预测模型",
            "action": "前往模型训练",
            "action_type": "switch_tab",
            "tab_index": 2
        },
        {
            "id": "large_file",
            "trigger": "large_file",
            "message": "💡 **技巧**：大文件分析较慢，可以使用采样分析或分批处理",
            "action": "在高级设置中调整采样率",
            "action_type": "show_advanced"
        },
        {
            "id": "date_features",
            "trigger": "has_datetime",
            "message": "💡 **技巧**：日期列会自动提取年、月、季度等特征，用于时间序列分析",
            "action": "查看报告中的时间序列分析",
            "action_type": "switch_tab",
            "tab_index": 1
        },
        {
            "id": "export_json",
            "trigger": "view_report",
            "message": "💡 **技巧**：JSON格式的结果可以被其他程序调用，适合集成到工作流中",
            "action": "下载JSON试试",
            "action_type": "export_json"
        },
        {
            "id": "correlation",
            "trigger": "has_correlation",
            "message": "💡 **技巧**：发现强相关的变量时，可以考虑只保留其中一个进行建模",
            "action": "查看相关性分析",
            "action_type": "switch_tab",
            "tab_index": 1
        }
    ]

    def __init__(self):
        self.shown_tips = st.session_state.get("shown_tips", set())
        self.last_tip_time = st.session_state.get("last_tip_time", None)

    def should_show(self, trigger: str) -> bool:
        """判断是否应该显示技巧"""
        # 已显示过的不再显示
        if trigger in self.shown_tips:
            return False

        # 频率限制：每小时最多显示1次
        if self.last_tip_time:
            try:
                last = datetime.fromisoformat(self.last_tip_time)
                if datetime.now() - last < timedelta(hours=1):
                    return False
            except:
                pass

        return True

    def get_tip_for_trigger(self, trigger: str) -> Dict:
        """获取触发对应的技巧"""
        for tip in self.TIPS:
            if tip["trigger"] == trigger:
                return tip
        return None

    def show_tip(self, trigger: str, context: Dict = None):
        """显示技巧"""
        tip = self.get_tip_for_trigger(trigger)
        if not tip or not self.should_show(trigger):
            return

        # 特殊处理：根据上下文选择不同的技巧
        if trigger == "view_report" and context and context.get("has_exported"):
            tip = self.get_tip_for_trigger("export_json")
            if not tip:
                return

        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.info(tip["message"])
            with col2:
                if st.button(tip["action"], key=f"tip_{tip['id']}", use_container_width=True):
                    self._handle_action(tip)
                    st.rerun()

        # 标记为已显示
        self.shown_tips.add(trigger)
        st.session_state.shown_tips = self.shown_tips
        st.session_state.last_tip_time = datetime.now().isoformat()

    def _handle_action(self, tip: Dict):
        """处理技巧中的动作"""
        action_type = tip.get("action_type", "")

        if action_type == "switch_tab":
            st.session_state.current_tab = tip.get("tab_index", 0)
        elif action_type == "open_settings":
            st.session_state.show_settings_dialog = True
        elif action_type == "export":
            st.session_state.current_tab = 1
        elif action_type == "show_advanced":
            st.session_state.show_advanced = True
        # 其他动作可以继续扩展

    def check_and_show(self, context: Dict[str, Any]):
        """根据上下文检查并显示技巧"""
        # 检查各种触发条件
        if context.get("view_report") and self.should_show("view_report"):
            self.show_tip("view_report", context)

        elif context.get("single_analysis") and self.should_show("single_analysis"):
            # 显示次数限制
            count = st.session_state.get("single_analysis_count", 0)
            if count >= 3:
                self.show_tip("multi_table")
                st.session_state.single_analysis_count = 0
            else:
                st.session_state.single_analysis_count = count + 1

        elif context.get("no_llm") and self.should_show("no_llm"):
            self.show_tip("no_llm")

        elif context.get("analysis_complete") and self.should_show("analysis_complete"):
            self.show_tip("model_training")

        elif context.get("large_file") and self.should_show("large_file"):
            self.show_tip("large_file")

        elif context.get("has_datetime") and self.should_show("has_datetime"):
            self.show_tip("date_features")

        elif context.get("has_correlation") and self.should_show("has_correlation"):
            self.show_tip("correlation")


def render_tips_panel():
    """渲染技巧面板（侧边栏底部）"""
    with st.expander("💡 使用技巧", expanded=False):
        st.markdown("""
        - **快速开始**：点击演示数据立即体验
        - **多表分析**：上传多个相关文件，自动发现关系
        - **AI解读**：配置大模型获得智能分析
        - **模型训练**：基于数据训练预测模型
        - **导出报告**：支持HTML/JSON/Excel格式
        """)