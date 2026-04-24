"""
分阶段进度组件 - 显示分析进度的详细阶段
"""

import streamlit as st
import time
from typing import Dict, Optional


class ProgressStage:
    """分阶段进度管理器 - 7阶段详细版本"""

    # 阶段定义
    STAGES = [
        {"id": "loading", "name": "📁 加载数据", "description": "正在读取和解析数据文件..."},
        {"id": "type", "name": "🔍 类型识别", "description": "识别变量类型..."},
        {"id": "feature", "name": "📅 特征提取", "description": "提取日期特征..."},
        {"id": "quality", "name": "📊 质量检查", "description": "检测数据质量..."},
        {"id": "relation", "name": "🔗 关系分析", "description": "分析变量关系..."},
        {"id": "timeseries", "name": "📈 时间序列", "description": "分析时间序列..."},
        {"id": "report", "name": "📝 生成报告", "description": "生成分析报告..."}
    ]

    def __init__(self):
        self.current_stage_index = 0
        self.stage_details = {}
        self.completed_stages = set()
        self.failed_stage = None
        self.failed_reason = ""
        self.start_time = None

    def start(self):
        """开始进度追踪"""
        self.start_time = time.time()
        self.current_stage_index = 0
        self.completed_stages = set()
        self.stage_details = {}
        self.failed_stage = None
        self.failed_reason = ""

    def start_stage(self, stage_id: str):
        """开始一个阶段"""
        for i, stage in enumerate(self.STAGES):
            if stage["id"] == stage_id:
                self.current_stage_index = i
                break

    def complete_stage(self, stage_id: str, detail: str = ""):
        """完成一个阶段"""
        self.completed_stages.add(stage_id)
        if detail:
            self.stage_details[stage_id] = detail

    def fail_stage(self, stage_id: str, reason: str):
        """阶段失败"""
        self.failed_stage = stage_id
        self.failed_reason = reason

    def get_overall_progress(self) -> float:
        """获取整体进度（0-1）"""
        if self.failed_stage:
            return 0
        if len(self.completed_stages) >= len(self.STAGES):
            return 1.0
        return len(self.completed_stages) / len(self.STAGES)

    def get_current_stage_info(self) -> Dict:
        """获取当前阶段信息"""
        if self.current_stage_index < len(self.STAGES):
            return self.STAGES[self.current_stage_index]
        return {"name": "✅ 完成", "description": "分析完成"}

    def get_eta(self) -> Optional[float]:
        """获取预计剩余时间（秒）"""
        if self.start_time is None:
            return None
        elapsed = time.time() - self.start_time
        progress = self.get_overall_progress()
        if progress < 0.01:
            return None
        total_estimated = elapsed / progress
        remaining = total_estimated - elapsed
        return max(0, remaining)

    def render(self, show_detail: bool = True):
        """渲染进度组件"""
        overall = self.get_overall_progress()
        eta = self.get_eta()

        st.progress(overall)

        if self.current_stage_index < len(self.STAGES):
            current_stage = self.STAGES[self.current_stage_index]
            detail = self.stage_details.get(current_stage["id"], "")
            st.info(f"{current_stage['name']}...")
            if detail:
                st.caption(detail)
            else:
                st.caption(current_stage["description"])
        elif self.failed_stage:
            st.error(f"❌ 失败：{self.failed_reason}")
        else:
            st.success("✅ 分析完成！")

        if eta is not None and eta > 0 and self.current_stage_index < len(self.STAGES):
            st.caption(f"预计剩余: {eta:.0f} 秒")

    def complete(self):
        """标记完成"""
        self.current_stage_index = len(self.STAGES)
        self.stage_details = {}


def render_analysis_progress(stage: str, progress: float, eta: float = None, message: str = ""):
    """
    简化版进度渲染函数
    """
    stages_map = {
        "loading": {"name": "📁 加载数据", "desc": "正在读取和解析数据文件..."},
        "analyzing": {"name": "🔍 分析结构", "desc": "识别变量类型和数据质量..."},
        "calculating": {"name": "📊 计算关系", "desc": "分析变量间相关性和关联..."},
        "generating": {"name": "📝 生成报告", "desc": "生成分析报告和可视化图表..."}
    }

    stage_info = stages_map.get(stage, {"name": stage, "desc": message or "处理中..."})

    st.progress(progress)
    st.info(f"{stage_info['name']}... {progress:.0%}")
    if message:
        st.caption(message)
    else:
        st.caption(stage_info["desc"])

    if eta:
        st.caption(f"预计剩余: {eta:.0f} 秒")


def render_simple_progress(stage: str, progress: float, eta: float = None, message: str = ""):
    """
    简化版进度渲染函数（别名，与 render_analysis_progress 相同）
    """
    render_analysis_progress(stage, progress, eta, message)