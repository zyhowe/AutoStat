"""
事件抽取模块 - 基于 DuEE（百度事件抽取模型）
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


class EventExtractor:
    """事件抽取器 - 基于 DuEE 模型"""

    # 金融领域事件类型（DuEE 支持的65种事件类型）
    EVENT_TYPES = [
        "发布财报", "业绩预告", "业绩快报", "业绩修正",
        "上市", "增发", "收购", "减持", "增持", "质押", "解禁", "回购",
        "中标", "签约", "投资", "产能扩张", "项目投产",
        "高管任命", "高管离职", "股权激励",
        "分红", "送转",
        "停牌", "复牌", "ST", "摘帽",
        "诉讼", "违规", "监管", "立案",
        "合作", "战略合作", "框架协议",
        "研发进展", "产品获批", "临床试验",
        "涨价", "降价", "成本变化"
    ]

    # 事件触发词映射
    TRIGGER_MAP = {
        "发布财报": ["发布", "公布", "披露"],
        "业绩预告": ["预告", "预计"],
        "收购": ["收购", "并购", "要约收购"],
        "减持": ["减持", "卖出"],
        "增持": ["增持", "买入"],
        "上市": ["上市", "IPO", "发行"],
        "中标": ["中标", "获得", "取得"],
        "签约": ["签约", "签署", "签订"],
        "投资": ["投资", "出资", "设立"],
        "高管任命": ["任命", "聘任", "委任"],
        "高管离职": ["辞职", "离职", "卸任"],
        "停牌": ["停牌"],
        "复牌": ["复牌"],
        "合作": ["合作", "携手", "联合"],
        "立案": ["立案调查", "立案"],
        "监管": ["监管", "问询", "警示"]
    }

    def __init__(self, use_model: bool = True):
        """
        初始化事件抽取器

        参数:
        - use_model: 是否使用 DuEE 模型（False 时使用规则匹配）
        """
        self.use_model = use_model
        self.model = None

        if use_model:
            self._load_model()
        else:
            print("  ⚠️ 使用规则匹配模式（事件抽取效果有限）")

    def _load_model(self):
        """加载 DuEE 模型"""
        try:
            # 尝试加载 PaddleNLP 的 DuEE 模型
            from paddlenlp import Taskflow
            # 使用通用信息抽取（UIE）作为 DuEE 的替代
            self.model = Taskflow(
                'information_extraction',
                schema=self.EVENT_TYPES,
                task_path='uie-base-zh'
            )
            print("  ✅ DuEE 模型加载完成（PaddleNLP）")
        except ImportError:
            print("  ⚠️ PaddleNLP 未安装，使用规则匹配模式")
            self.use_model = False
        except Exception as e:
            print(f"  ⚠️ DuEE 模型加载失败: {e}，使用规则匹配模式")
            self.use_model = False

    def extract(self, texts: List[str]) -> List[List[Dict]]:
        """
        批量抽取事件

        参数:
        - texts: 文本列表

        返回: [
            [  # 第 i 条文本的事件列表
                {
                    "event_type": "发布财报",
                    "trigger": "发布",
                    "args": {
                        "公司": "宁德时代",
                        "时间": "2024-01-15",
                        "营收": "847亿元"
                    },
                    "confidence": 0.95
                },
                ...
            ],
            ...
        ]
        """
        results = []
        for i, text in enumerate(texts):
            if self.use_model and self.model:
                events = self._extract_with_model(text)
            else:
                events = self._extract_with_rules(text)
            results.append(events)
        return results

    def _extract_with_model(self, text: str) -> List[Dict]:
        """使用模型抽取事件"""
        try:
            # UIE 模型返回格式
            result = self.model(text)
            events = self._parse_model_result(result)
            return events
        except Exception as e:
            print(f"  ⚠️ 模型抽取失败: {e}，使用规则匹配")
            return self._extract_with_rules(text)

    def _parse_model_result(self, result: Any) -> List[Dict]:
        """解析模型输出"""
        events = []

        if not result or len(result) == 0:
            return events

        # 处理不同格式的返回
        if isinstance(result, list):
            for item in result:
                event = self._parse_event_item(item)
                if event:
                    events.append(event)
        elif isinstance(result, dict):
            event = self._parse_event_item(result)
            if event:
                events.append(event)

        return events

    def _parse_event_item(self, item: Dict) -> Optional[Dict]:
        """解析单个事件项"""
        try:
            event_type = list(item.keys())[0] if item else None
            if not event_type:
                return None

            event_data = item[event_type]

            # 提取论元
            args = {}
            trigger = ""

            if isinstance(event_data, dict):
                for arg_type, arg_value in event_data.items():
                    if arg_type == "触发词":
                        trigger = arg_value
                    elif isinstance(arg_value, str):
                        args[arg_type] = arg_value
                    elif isinstance(arg_value, list) and len(arg_value) > 0:
                        if isinstance(arg_value[0], str):
                            args[arg_type] = arg_value[0]

            return {
                "event_type": event_type,
                "trigger": trigger,
                "args": args,
                "confidence": 0.8
            }
        except Exception:
            return None

    def _extract_with_rules(self, text: str) -> List[Dict]:
        """使用规则匹配抽取事件"""
        events = []

        for event_type, triggers in self.TRIGGER_MAP.items():
            for trigger in triggers:
                if trigger in text:
                    # 提取上下文
                    context = self._extract_context(text, trigger)
                    events.append({
                        "event_type": event_type,
                        "trigger": trigger,
                        "args": context,
                        "confidence": 0.5
                    })
                    break  # 每类事件只匹配一次

        return events

    def _extract_context(self, text: str, trigger: str) -> Dict:
        """提取事件上下文（实体和时间）"""
        args = {}

        # 提取日期
        date_patterns = [
            r'(\d{4}年\d{1,2}月\d{1,2}日)',
            r'(\d{4}-\d{1,2}-\d{1,2})',
            r'(\d{1,2}月\d{1,2}日)',
            r'(昨日|今日|明日|本周|本月|今年)'
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                args["时间"] = match.group(1)
                break

        # 提取金额
        amount_patterns = [
            r'(\d+(?:\.\d+)?\s*亿(?:元|美元)?)',
            r'(\d+(?:\.\d+)?\s*万(?:元|美元)?)',
            r'(\d+(?:\.\d+)?\s*%)'
        ]

        for pattern in amount_patterns:
            matches = re.findall(pattern, text)
            if matches:
                args["金额"] = matches[0]
                break

        return args

    def get_event_stats(self, events_results: List[List[Dict]]) -> Dict:
        """获取事件统计"""
        stats = {}
        event_counter = Counter()
        event_by_type = Counter()

        for events in events_results:
            for event in events:
                event_type = event.get("event_type", "未知")
                event_counter[event_type] += 1
                event_by_type[event_type] += 1

        stats["total_events"] = sum(event_counter.values())
        stats["event_types"] = dict(event_by_type)
        stats["top_events"] = event_by_type.most_common(20)

        return stats