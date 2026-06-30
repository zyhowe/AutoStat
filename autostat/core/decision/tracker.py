"""
效果追踪模块

追踪行动建议的执行效果，形成反馈闭环
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import json
from pathlib import Path
import pandas as pd  # 🆕 添加 pandas 导入


@dataclass
class ActionRecord:
    """行动记录"""
    id: str
    suggestion_id: str
    title: str
    status: str  # "pending", "in_progress", "completed", "cancelled", "archived"
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[str] = None
    effect_metrics: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class TrackerResult:
    """追踪结果"""
    action: ActionRecord
    before: Dict[str, Any]
    after: Dict[str, Any]
    effect: str  # "improved", "no_change", "worsened"
    effect_pct: float
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)


class ActionTracker:
    """
    效果追踪器

    使用方式:
        tracker = ActionTracker()
        record = tracker.record_action(suggestion)
        result = tracker.track(record_id, before_data, after_data)
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化追踪器

        参数:
        - storage_path: 存储路径（用于持久化）
        """
        self.records: List[ActionRecord] = []
        self.results: List[TrackerResult] = []
        self.storage_path = storage_path

        if storage_path:
            self._load()

    def record_action(self, suggestion: Dict[str, Any]) -> ActionRecord:
        """
        记录行动

        参数:
        - suggestion: 行动建议

        返回: 行动记录
        """
        record = ActionRecord(
            id=f"action_{int(datetime.now().timestamp())}",
            suggestion_id=suggestion.get("id", ""),
            title=suggestion.get("title", ""),
            status="pending",
            created_at=datetime.now().isoformat(),
            tags=suggestion.get("tags", [])
        )

        self.records.append(record)
        self._save()

        return record

    def record_action_from_suggestion(
        self,
        suggestion: Any,  # ActionSuggestion 对象
        custom_title: Optional[str] = None
    ) -> ActionRecord:
        """
        从 ActionSuggestion 对象记录行动

        参数:
        - suggestion: ActionSuggestion 对象
        - custom_title: 自定义标题

        返回: 行动记录
        """
        if hasattr(suggestion, '__dict__'):
            data = suggestion.__dict__
        else:
            data = suggestion

        return self.record_action({
            "id": data.get("id", ""),
            "title": custom_title or data.get("title", ""),
            "tags": data.get("tags", [])
        })

    def start_action(self, record_id: str) -> bool:
        """
        开始执行行动

        参数:
        - record_id: 行动记录ID

        返回: 是否成功
        """
        record = self._find_record(record_id)
        if record:
            record.status = "in_progress"
            record.started_at = datetime.now().isoformat()
            self._save()
            return True
        return False

    def complete_action(
        self,
        record_id: str,
        result: str,
        effect_metrics: Optional[Dict[str, Any]] = None,
        notes: Optional[List[str]] = None
    ) -> bool:
        """
        完成行动

        参数:
        - record_id: 行动记录ID
        - result: 执行结果描述
        - effect_metrics: 效果指标
        - notes: 备注

        返回: 是否成功
        """
        record = self._find_record(record_id)
        if record:
            record.status = "completed"
            record.completed_at = datetime.now().isoformat()
            record.result = result
            if effect_metrics:
                record.effect_metrics.update(effect_metrics)
            if notes:
                record.notes.extend(notes)
            self._save()
            return True
        return False

    def cancel_action(self, record_id: str, reason: str = "") -> bool:
        """
        取消行动

        参数:
        - record_id: 行动记录ID
        - reason: 取消原因

        返回: 是否成功
        """
        record = self._find_record(record_id)
        if record:
            record.status = "cancelled"
            if reason:
                record.notes.append(f"取消原因: {reason}")
            self._save()
            return True
        return False

    def add_note(self, record_id: str, note: str) -> bool:
        """
        添加备注

        参数:
        - record_id: 行动记录ID
        - note: 备注内容

        返回: 是否成功
        """
        record = self._find_record(record_id)
        if record:
            record.notes.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {note}")
            self._save()
            return True
        return False

    def track(
        self,
        record_id: str,
        before_data: Dict[str, Any],
        after_data: Dict[str, Any],
        metric_keys: Optional[List[str]] = None
    ) -> Optional[TrackerResult]:
        """
        追踪效果

        参数:
        - record_id: 行动记录ID
        - before_data: 执行前数据
        - after_data: 执行后数据
        - metric_keys: 要对比的指标（默认使用 before_data 和 after_data 的所有共同键）

        返回: 追踪结果
        """
        record = self._find_record(record_id)
        if not record:
            return None

        # 确定对比指标
        if metric_keys is None:
            metric_keys = list(set(before_data.keys()) & set(after_data.keys()))

        # 过滤掉非数值指标
        numeric_keys = []
        for key in metric_keys:
            try:
                before_val = before_data.get(key, 0)
                after_val = after_data.get(key, 0)
                float(before_val)
                float(after_val)
                numeric_keys.append(key)
            except (ValueError, TypeError):
                continue

        if not numeric_keys:
            return None

        # 计算效果
        effects = {}
        total_change = 0

        for key in numeric_keys:
            before_val = float(before_data.get(key, 0))
            after_val = float(after_data.get(key, 0))

            if before_val != 0:
                change_pct = (after_val - before_val) / abs(before_val) * 100
            elif after_val != 0:
                change_pct = 100 if after_val > 0 else -100
            else:
                change_pct = 0

            effects[key] = {
                "before": before_val,
                "after": after_val,
                "change": round(change_pct, 1)
            }
            total_change += change_pct

        avg_effect = total_change / len(numeric_keys) if numeric_keys else 0

        # 判断整体效果
        if avg_effect > 5:
            effect = "improved"
        elif avg_effect < -5:
            effect = "worsened"
        else:
            effect = "no_change"

        # 生成摘要
        summary = self._generate_summary(effect, avg_effect, effects, record)

        # 构建结果
        result = TrackerResult(
            action=record,
            before=before_data,
            after=after_data,
            effect=effect,
            effect_pct=avg_effect,
            summary=summary,
            details={
                "metric_effects": effects,
                "numeric_keys": numeric_keys,
                "tracked_at": datetime.now().isoformat()
            }
        )

        self.results.append(result)

        # 更新记录的指标
        record.effect_metrics = {
            "avg_change": avg_effect,
            "effect": effect,
            "tracked_at": datetime.now().isoformat()
        }
        self._save()

        return result

    def track_multiple(
        self,
        record_id: str,
        before_series: List[Dict[str, Any]],
        after_series: List[Dict[str, Any]],
        metric_keys: Optional[List[str]] = None
    ) -> List[TrackerResult]:
        """
        追踪多个时间点的效果

        参数:
        - record_id: 行动记录ID
        - before_series: 执行前多个时间点的数据
        - after_series: 执行后多个时间点的数据
        - metric_keys: 要对比的指标

        返回: 追踪结果列表
        """
        results = []

        # 取平均值作为对比
        if before_series and after_series:
            avg_before = {}
            avg_after = {}

            # 计算平均值
            for key in (metric_keys or []):
                before_vals = [d.get(key, 0) for d in before_series if key in d]
                after_vals = [d.get(key, 0) for d in after_series if key in d]

                if before_vals:
                    avg_before[key] = sum(before_vals) / len(before_vals)
                if after_vals:
                    avg_after[key] = sum(after_vals) / len(after_vals)

            if avg_before and avg_after:
                result = self.track(record_id, avg_before, avg_after, metric_keys)
                if result:
                    results.append(result)

        return results

    def _find_record(self, record_id: str) -> Optional[ActionRecord]:
        """查找记录"""
        for record in self.records:
            if record.id == record_id:
                return record
        return None

    def _generate_summary(
        self,
        effect: str,
        effect_pct: float,
        effects: Dict[str, Any],
        record: ActionRecord
    ) -> str:
        """生成效果摘要"""
        if effect == "improved":
            return f"✅ {record.title}: 效果积极，综合提升 {effect_pct:.1f}%"
        elif effect == "worsened":
            return f"❌ {record.title}: 效果负面，综合下降 {abs(effect_pct):.1f}%"
        else:
            return f"➖ {record.title}: 效果不明显，综合变化 {effect_pct:.1f}%"

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        completed = [r for r in self.records if r.status == "completed"]
        in_progress = [r for r in self.records if r.status == "in_progress"]
        cancelled = [r for r in self.records if r.status == "cancelled"]

        improved = [r for r in self.results if r.effect == "improved"]
        worsened = [r for r in self.results if r.effect == "worsened"]

        # 计算平均效果
        avg_effect = 0
        if self.results:
            avg_effect = sum(r.effect_pct for r in self.results) / len(self.results)

        return {
            "total_actions": len(self.records),
            "completed": len(completed),
            "in_progress": len(in_progress),
            "pending": len([r for r in self.records if r.status == "pending"]),
            "cancelled": len(cancelled),
            "improved": len(improved),
            "worsened": len(worsened),
            "no_change": len([r for r in self.results if r.effect == "no_change"]),
            "avg_effect_pct": round(avg_effect, 1),
            "tracked_count": len(self.results)
        }

    def get_action_by_id(self, record_id: str) -> Optional[ActionRecord]:
        """根据ID获取行动记录"""
        return self._find_record(record_id)

    def get_actions_by_status(self, status: str) -> List[ActionRecord]:
        """根据状态获取行动记录"""
        return [r for r in self.records if r.status == status]

    def get_actions_by_tag(self, tag: str) -> List[ActionRecord]:
        """根据标签获取行动记录"""
        return [r for r in self.records if tag in r.tags]

    def get_active_actions(self) -> List[ActionRecord]:
        """获取活跃的行动（进行中+待处理）"""
        return [r for r in self.records if r.status in ["pending", "in_progress"]]

    def get_completed_summary(self) -> List[Dict[str, Any]]:
        """获取已完成行动的摘要"""
        summary = []
        for record in self.records:
            if record.status != "completed":
                continue

            # 查找对应的追踪结果
            result = None
            for r in self.results:
                if r.action.id == record.id:
                    result = r
                    break

            summary.append({
                "id": record.id,
                "title": record.title,
                "completed_at": record.completed_at,
                "result": record.result,
                "effect": result.effect if result else "unknown",
                "effect_pct": result.effect_pct if result else 0,
                "metrics": record.effect_metrics
            })

        return sorted(summary, key=lambda x: x.get("completed_at", ""), reverse=True)

    def _save(self):
        """保存到文件"""
        if not self.storage_path:
            return

        try:
            path = Path(self.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "records": [
                    {
                        "id": r.id,
                        "suggestion_id": r.suggestion_id,
                        "title": r.title,
                        "status": r.status,
                        "created_at": r.created_at,
                        "started_at": r.started_at,
                        "completed_at": r.completed_at,
                        "result": r.result,
                        "effect_metrics": r.effect_metrics,
                        "notes": r.notes,
                        "tags": r.tags
                    }
                    for r in self.records
                ]
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"保存追踪数据失败: {e}")

    def _load(self):
        """从文件加载"""
        if not self.storage_path:
            return

        try:
            path = Path(self.storage_path)
            if not path.exists():
                return

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.records = []
            for item in data.get("records", []):
                self.records.append(ActionRecord(
                    id=item["id"],
                    suggestion_id=item.get("suggestion_id", ""),
                    title=item.get("title", ""),
                    status=item.get("status", "pending"),
                    created_at=item.get("created_at", datetime.now().isoformat()),
                    started_at=item.get("started_at"),
                    completed_at=item.get("completed_at"),
                    result=item.get("result"),
                    effect_metrics=item.get("effect_metrics", {}),
                    notes=item.get("notes", []),
                    tags=item.get("tags", [])
                ))

        except Exception as e:
            print(f"加载追踪数据失败: {e}")
            self.records = []

    def export_to_dataframe(self) -> pd.DataFrame:
        """
        导出为DataFrame

        返回: pandas DataFrame
        """
        rows = []
        for record in self.records:
            # 查找对应的追踪结果
            result = None
            for r in self.results:
                if r.action.id == record.id:
                    result = r
                    break

            rows.append({
                "id": record.id,
                "title": record.title,
                "status": record.status,
                "created_at": record.created_at,
                "started_at": record.started_at,
                "completed_at": record.completed_at,
                "result": record.result,
                "effect": result.effect if result else "unknown",
                "effect_pct": result.effect_pct if result else 0,
                "tags": ", ".join(record.tags)
            })

        return pd.DataFrame(rows)


def track_action(
    suggestion: Dict[str, Any],
    before_data: Dict[str, Any],
    after_data: Dict[str, Any],
    metric_keys: Optional[List[str]] = None,
    **kwargs
) -> Optional[TrackerResult]:
    """
    便捷函数：记录并追踪一个行动

    参数:
    - suggestion: 行动建议
    - before_data: 执行前数据
    - after_data: 执行后数据
    - metric_keys: 要对比的指标

    返回: 追踪结果
    """
    tracker = ActionTracker(**kwargs)
    record = tracker.record_action(suggestion)
    return tracker.track(record.id, before_data, after_data, metric_keys)