"""
质量告警引擎

基于质量评分和规则配置触发告警
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import requests


class AlertLevel(Enum):
    """告警级别"""
    INFO = ("info", "ℹ️")
    WARNING = ("warning", "⚠️")
    ERROR = ("error", "🚨")
    CRITICAL = ("critical", "🔴")

    def __init__(self, label: str, icon: str):
        self.label = label
        self.icon = icon


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    condition: Callable[[Dict], bool]
    level: AlertLevel
    message_template: str
    enabled: bool = True


@dataclass
class AlertEvent:
    """告警事件"""
    id: str
    timestamp: str
    level: AlertLevel
    title: str
    message: str
    data: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False


class QualityAlert:
    """
    质量告警引擎

    使用方式:
        alert = QualityAlert()
        alert.add_rule(rule)
        events = alert.check(score_result)
        alert.notify(events)
    """

    def __init__(self):
        self.rules: List[AlertRule] = []
        self.events: List[AlertEvent] = []
        self.notifiers: List[Callable] = []

        # 注册默认规则
        self._register_default_rules()

    def _register_default_rules(self):
        """注册默认规则"""
        # 规则1: 综合评分低于60
        self.add_rule(AlertRule(
            name="综合评分过低",
            condition=lambda d: d.get("overall_score", 100) < 60,
            level=AlertLevel.CRITICAL,
            message_template="综合评分 {overall_score:.1f} 低于60分，请立即处理"
        ))

        # 规则2: 综合评分低于70
        self.add_rule(AlertRule(
            name="综合评分偏低",
            condition=lambda d: 60 <= d.get("overall_score", 100) < 70,
            level=AlertLevel.ERROR,
            message_template="综合评分 {overall_score:.1f} 低于70分，建议尽快处理"
        ))

        # 规则3: 完整性低于60%
        self.add_rule(AlertRule(
            name="完整性不足",
            condition=lambda d: d.get("dimensions", {}).get("completeness", 100) < 60,
            level=AlertLevel.ERROR,
            message_template="完整性得分 {completeness:.1f}%，存在大量缺失值"
        ))

        # 规则4: 准确性低于60%
        self.add_rule(AlertRule(
            name="准确性不足",
            condition=lambda d: d.get("dimensions", {}).get("accuracy", 100) < 60,
            level=AlertLevel.ERROR,
            message_template="准确性得分 {accuracy:.1f}%，存在较多异常值"
        ))

        # 规则5: 一致性低于60%
        self.add_rule(AlertRule(
            name="一致性不足",
            condition=lambda d: d.get("dimensions", {}).get("consistency", 100) < 60,
            level=AlertLevel.WARNING,
            message_template="一致性得分 {consistency:.1f}%，存在勾稽规则违反"
        ))

        # 规则6: 唯一性低于80%
        self.add_rule(AlertRule(
            name="唯一性不足",
            condition=lambda d: d.get("dimensions", {}).get("uniqueness", 100) < 80,
            level=AlertLevel.WARNING,
            message_template="唯一性得分 {uniqueness:.1f}%，存在较多重复记录"
        ))

        # 规则7: 综合评分持续下降（需要外部数据，使用特殊标记）
        # 此规则在 check 中通过额外参数处理

    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.rules.append(rule)

    def add_notifier(self, notifier: Callable[[List[AlertEvent]], None]):
        """添加通知器"""
        self.notifiers.append(notifier)

    def check(self, score_data: Dict[str, Any], context: Optional[Dict] = None) -> List[AlertEvent]:
        """
        检查告警

        参数:
        - score_data: 评分数据（来自 QualityScore 或字典）
        - context: 额外上下文（如历史趋势）

        返回: 告警事件列表
        """
        events = []

        # 构建上下文
        data = {
            "overall_score": score_data.get("overall_score", 0),
            "dimensions": score_data.get("dimensions", {}),
            "field_scores": score_data.get("field_scores", {}),
            "alerts": score_data.get("alerts", []),
            **(context or {})
        }

        # 检查每个规则
        for rule in self.rules:
            if not rule.enabled:
                continue
            try:
                if rule.condition(data):
                    event = self._create_event(rule, data)
                    events.append(event)
            except Exception as e:
                print(f"告警规则执行失败: {rule.name} - {e}")

        # 额外检查：趋势持续下降
        if context and context.get("trend"):
            trend = context.get("trend")
            if trend.get("is_anomaly") and trend.get("anomaly_type") == "continuous_drop":
                events.append(AlertEvent(
                    id=f"trend_{datetime.now().timestamp()}",
                    timestamp=datetime.now().isoformat(),
                    level=AlertLevel.ERROR,
                    title="质量持续下降",
                    message=f"质量评分连续下降 {trend.get('change_pct', 0):.1f}%",
                    data={"trend": trend}
                ))

        # 保存事件
        self.events.extend(events)

        # 触发通知
        if events:
            self._notify(events)

        return events

    def _create_event(self, rule: AlertRule, data: Dict) -> AlertEvent:
        """创建告警事件"""
        # 格式化消息
        message = rule.message_template.format(**data)

        return AlertEvent(
            id=f"{rule.name}_{datetime.now().timestamp()}",
            timestamp=datetime.now().isoformat(),
            level=rule.level,
            title=rule.name,
            message=message,
            data=data
        )

    def _notify(self, events: List[AlertEvent]):
        """发送通知"""
        for notifier in self.notifiers:
            try:
                notifier(events)
            except Exception as e:
                print(f"通知发送失败: {e}")

    def get_unresolved_events(self) -> List[AlertEvent]:
        """获取未解决的事件"""
        return [e for e in self.events if not e.resolved]

    def resolve_event(self, event_id: str):
        """标记事件已解决"""
        for event in self.events:
            if event.id == event_id:
                event.resolved = True
                event.resolved_at = datetime.now().isoformat()
                break

    def acknowledge_event(self, event_id: str):
        """确认事件"""
        for event in self.events:
            if event.id == event_id:
                event.acknowledged = True
                break


# ==================== 内置通知器 ====================

def dingtalk_notifier(webhook_url: str):
    """创建钉钉通知器"""
    def notify(events: List[AlertEvent]):
        if not events:
            return

        # 只发送 ERROR 和 CRITICAL 级别
        critical_events = [e for e in events if e.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]]
        if not critical_events:
            return

        msg = "**【数据质量告警】**\n\n"
        for e in critical_events[:5]:
            msg += f"{e.level.icon} **{e.title}**\n"
            msg += f"  {e.message}\n"
            msg += f"  时间: {e.timestamp[:19]}\n\n"

        if len(critical_events) > 5:
            msg += f"... 还有 {len(critical_events) - 5} 条告警"

        try:
            requests.post(webhook_url, json={
                "msgtype": "text",
                "text": {"content": msg}
            }, timeout=5)
        except Exception as e:
            print(f"钉钉通知失败: {e}")

    return notify


def wechat_notifier(webhook_url: str):
    """创建企业微信通知器"""
    def notify(events: List[AlertEvent]):
        if not events:
            return

        critical_events = [e for e in events if e.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]]
        if not critical_events:
            return

        msg = "**【数据质量告警】**\n\n"
        for e in critical_events[:5]:
            msg += f"{e.level.icon} **{e.title}**\n"
            msg += f"  {e.message}\n\n"

        try:
            requests.post(webhook_url, json={
                "msgtype": "text",
                "text": {"content": msg}
            }, timeout=5)
        except Exception as e:
            print(f"企业微信通知失败: {e}")

    return notify


def email_notifier(smtp_config: Dict):
    """创建邮件通知器"""
    def notify(events: List[AlertEvent]):
        # 简化实现，实际使用时需要配置 SMTP
        print(f"📧 邮件通知: {len(events)} 条告警")
        # 实际邮件发送逻辑省略

    return notify


def console_notifier():
    """控制台通知器（用于调试）"""
    def notify(events: List[AlertEvent]):
        if not events:
            return
        print("\n" + "=" * 60)
        print("📢 质量告警")
        print("=" * 60)
        for e in events[:5]:
            print(f"{e.level.icon} {e.title}: {e.message}")
        print("=" * 60 + "\n")

    return notify