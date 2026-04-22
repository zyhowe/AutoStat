"""
分阶段进度组件 - 显示分析进度
"""

import streamlit as st
import time
from typing import List, Dict, Any, Optional


class ProgressStage:
    """分阶段进度管理器"""

    STAGES = [
        {"name": "📁 加载数据", "weight": 0.15, "description": "正在读取和解析数据文件..."},
        {"name": "🔍 分析结构", "weight": 0.25, "description": "识别变量类型和数据质量..."},
        {"name": "📊 计算关系", "weight": 0.30, "description": "分析变量间相关性和关联..."},
        {"name": "📝 生成报告", "weight": 0.30, "description": "生成分析报告和可视化图表..."}
    ]

    def __init__(self):
        self.current_stage = 0
        self.stage_progress = 0
        self.start_time = None
        self._message = ""

    def start(self):
        """开始进度追踪"""
        self.start_time = time.time()
        self.current_stage = 0
        self.stage_progress = 0

    def update(self, stage_index: int, stage_progress: float, message: str = ""):
        """更新进度"""
        self.current_stage = stage_index
        self.stage_progress = stage_progress
        if message:
            self._message = message

    def advance_to_next_stage(self):
        """进入下一阶段"""
        if self.current_stage < len(self.STAGES) - 1:
            self.current_stage += 1
            self.stage_progress = 0

    def get_overall_progress(self) -> float:
        """获取整体进度"""
        overall = 0
        for i, stage in enumerate(self.STAGES):
            if i < self.current_stage:
                overall += stage["weight"]
            elif i == self.current_stage:
                overall += stage["weight"] * self.stage_progress
        return min(overall, 0.99)  # 最多99%，留1%给完成

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

    def get_current_stage_info(self) -> Dict:
        """获取当前阶段信息"""
        if self.current_stage < len(self.STAGES):
            return self.STAGES[self.current_stage]
        return {"name": "✅ 完成", "description": "分析完成"}

    def render(self):
        """渲染进度组件"""
        overall = self.get_overall_progress()
        eta = self.get_eta()
        current = self.get_current_stage_info()

        # 进度条
        st.progress(overall)

        # 当前阶段
        st.info(f"{current['name']}... {self.stage_progress:.0%}")
        if self._message:
            st.caption(self._message)
        else:
            st.caption(current["description"])

        # ETA
        if eta is not None and eta > 0:
            st.caption(f"预计剩余: {eta:.0f} 秒")

    def complete(self):
        """标记完成"""
        self.current_stage = len(self.STAGES)
        self.stage_progress = 1.0
        self._message = ""


def render_analysis_progress(stage: str, progress: float, eta: float = None, message: str = ""):
    """简化版进度渲染函数"""
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