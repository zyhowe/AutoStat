"""
预警规则引擎

基于规则配置触发预警
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass


class AlertLevel(Enum):
    """预警级别"""
    INFO = ("info", "ℹ️")
    WARNING = ("warning", "⚠️")
    ERROR = ("error", "🚨")
    CRITICAL = ("critical", "🔴")

    def __init__(self, label: str, icon: str):
        self.label = label
        self.icon = icon


@dataclass
class AlertRule:
    """预警规则"""
    id: str
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    level: AlertLevel
    message_template: str
    enabled: bool = True
    cooldown: int = 3600  # 冷却时间（秒）


@dataclass
class AlertEvent:
    """预警事件"""
    id: str
    rule_id: str
    level: AlertLevel
    title: str
    message: str
    data: Dict[str, Any]
    triggered_at: str
    acknowledged: bool = False
    resolved: bool = False


class AlertEngine:
    """
    预警规则引擎

    使用方式:
        engine = AlertEngine()
        engine.add_rule(rule)
        events = engine.check(data)
    """

    def __init__(self):
        self.rules: List[AlertRule] = []
        self.events: List[AlertEvent] = []
        self.notifiers: List[Callable] = []
        self._last_triggered: Dict[str, float] = {}

        # 注册默认规则
        self._register_default_rules()

    def _register_default_rules(self):
        """注册默认规则"""
        self.add_rule(AlertRule(
            id="value_below_threshold",
            name="数值低于阈值",
            condition=lambda d: d.get("value", 0) < d.get("threshold", 0),
            level=AlertLevel.WARNING,
            message_template="{target} 低于阈值: {value:.2f} < {threshold:.2f}"
        ))

        self.add_rule(AlertRule(
            id="value_above_threshold",
            name="数值高于阈值",
            condition=lambda d: d.get("value", 0) > d.get("threshold", 0),
            level=AlertLevel.WARNING,
            message_template="{target} 高于阈值: {value:.2f} > {threshold:.2f}"
        ))

        self.add_rule(AlertRule(
            id="continuous_decline",
            name="持续下降",
            condition=lambda d: d.get("decline_periods", 0) >= 3,
            level=AlertLevel.ERROR,
            message_template="{target} 连续下降 {decline_periods} 期"
        ))

        self.add_rule(AlertRule(
            id="continuous_increase",
            name="持续上升",
            condition=lambda d: d.get("increase_periods", 0) >= 5,
            level=AlertLevel.INFO,
            message_template="{target} 连续上升 {increase_periods} 期"
        ))

        self.add_rule(AlertRule(
            id="out_of_prediction_range",
            name="超出预测范围",
            condition=lambda d: d.get("actual", 0) < d.get("lower_bound", 0) or
                             d.get("actual", 0) > d.get("upper_bound", 0),
            level=AlertLevel.ERROR,
            message_template="{target} 超出预测区间: {actual:.2f} 不在 [{lower_bound:.2f}, {upper_bound:.2f}]"
        ))

        self.add_rule(AlertRule(
            id="high_anomaly_rate",
            name="异常率过高",
            condition=lambda d: d.get("anomaly_rate", 0) > 0.1,
            level=AlertLevel.WARNING,
            message_template="{target} 异常率过高: {anomaly_rate:.1%}"
        ))

    def add_rule(self, rule: AlertRule):
        """添加规则"""
        self.rules.append(rule)

    def add_notifier(self, notifier: Callable[[List[AlertEvent]], None]):
        """添加通知器"""
        self.notifiers.append(notifier)

    def check(self, data: Dict[str, Any]) -> List[AlertEvent]:
        """
        检查规则

        参数:
        - data: 检查数据

        返回: 触发的事件列表
        """
        events = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            # 检查冷却
            now = datetime.now().timestamp()
            last = self._last_triggered.get(rule.id, 0)
            if now - last < rule.cooldown:
                continue

            try:
                if rule.condition(data):
                    event = self._create_event(rule, data)
                    events.append(event)
                    self._last_triggered[rule.id] = now
            except Exception as e:
                print(f"规则检查失败: {rule.name} - {e}")

        # 保存事件
        self.events.extend(events)

        # 发送通知
        if events:
            self._notify(events)

        return events

    def check_batch(self, data_list: List[Dict[str, Any]]) -> List[AlertEvent]:
        """批量检查"""
        all_events = []
        for data in data_list:
            events = self.check(data)
            all_events.extend(events)
        return all_events

    def _create_event(self, rule: AlertRule, data: Dict[str, Any]) -> AlertEvent:
        """创建事件"""
        # 格式化消息
        try:
            message = rule.message_template.format(**data)
        except KeyError:
            message = rule.message_template

        return AlertEvent(
            id=f"{rule.id}_{datetime.now().timestamp()}",
            rule_id=rule.id,
            level=rule.level,
            title=rule.name,
            message=message,
            data=data,
            triggered_at=datetime.now().isoformat()
        )

    def _notify(self, events: List[AlertEvent]):
        """发送通知"""
        for notifier in self.notifiers:
            try:
                notifier(events)
            except Exception as e:
                print(f"通知发送失败: {e}")

    def get_unresolved(self) -> List[AlertEvent]:
        """获取未解决的事件"""
        return [e for e in self.events if not e.resolved]

    def resolve(self, event_id: str):
        """标记事件已解决"""
        for event in self.events:
            if event.id == event_id:
                event.resolved = True
                break

    def acknowledge(self, event_id: str):
        """确认事件"""
        for event in self.events:
            if event.id == event_id:
                event.acknowledged = True
                break


# ==================== 内置通知器 ====================

def dingtalk_alert_notifier(webhook_url: str):
    """钉钉通知器"""
    import requests

    def notify(events: List[AlertEvent]):
        if not events:
            return

        critical = [e for e in events if e.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]]
        if not critical:
            return

        msg = "**【数据预警】**\n\n"
        for e in critical[:5]:
            msg += f"{e.level.icon} **{e.title}**\n"
            msg += f"  {e.message}\n\n"

        try:
            requests.post(webhook_url, json={
                "msgtype": "text",
                "text": {"content": msg}
            }, timeout=5)
        except Exception as e:
            print(f"钉钉通知失败: {e}")

    return notify


def console_alert_notifier():
    """控制台通知器"""
    def notify(events: List[AlertEvent]):
        if not events:
            return
        print("\n" + "=" * 60)
        print("🚨 数据预警")
        print("=" * 60)
        for e in events[:5]:
            print(f"{e.level.icon} {e.title}: {e.message}")
        print("=" * 60 + "\n")

    return notify


def create_alert_rule(
    name: str,
    condition: Callable,
    level: AlertLevel = AlertLevel.WARNING,
    message_template: str = "",
    **kwargs
) -> AlertRule:
    """创建预警规则"""
    return AlertRule(
        id=f"rule_{datetime.now().timestamp()}",
        name=name,
        condition=condition,
        level=level,
        message_template=message_template or f"{{target}}: {name}",
        **kwargs
    )