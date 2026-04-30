"""
事件抽取模块 - 优先使用 DuEE（百度事件抽取），降级使用规则匹配
"""

import re
from typing import List, Dict, Any, Optional
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


class EventExtractor:
    """事件抽取器 - 优先 DuEE，降级规则匹配"""

    # 规则匹配的事件类型（DuEE 不可用时使用）
    RULE_EVENT_TYPES = {
        "发布财报": ["发布", "公布", "披露", "财报", "年报", "季报", "业绩"],
        "业绩预告": ["预告", "预计", "业绩预告"],
        "收购": ["收购", "并购", "要约收购"],
        "减持": ["减持", "套现"],
        "增持": ["增持"],
        "上市": ["上市", "IPO", "发行", "登陆"],
        "中标": ["中标", "获得", "取得"],
        "签约": ["签约", "签署", "签订"],
        "投资": ["投资", "出资", "设立", "建厂"],
        "高管任命": ["任命", "聘任", "委任", "担任"],
        "高管离职": ["辞职", "离职", "卸任", "辞任"],
        "合作": ["合作", "携手", "联合"],
        "涨价": ["涨价", "上涨", "上调"],
        "降价": ["降价", "下跌", "下调"],
        "创新高": ["创新高", "历史新高", "突破"],
        "融资": ["融资", "募资"],
        "分红": ["分红", "派息"],
        "回购": ["回购"]
    }

    COMPANY_PATTERN = r'([\u4e00-\u9fff]{2,10}(?:公司|集团|股份|有限|银行|证券|基金|保险))'
    PERSON_PATTERN = r'([\u4e00-\u9fff]{2,4}(?:先生|女士|总(?:裁|经理)|董事长|总经理|CEO))'
    MONEY_PATTERN = r'(\d+(?:\.\d+)?\s*(?:亿|万|千)\s*(?:元|美元|港元)?)'
    PERCENT_PATTERN = r'(\d+(?:\.\d+)?\s*%)'
    DATE_PATTERN = r'(\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{1,2}-\d{1,2})'

    def __init__(self, use_model: bool = True):
        """
        初始化事件抽取器

        参数:
        - use_model: 是否使用 DuEE 模型（True: 使用 DuEE, False: 使用规则匹配）
        """
        self.use_model = use_model
        self.model = None

        if use_model:
            self._load_duee()

    def _load_duee(self):
        """加载 DuEE 模型"""
        try:
            from paddlenlp import Taskflow

            # DuEE 事件抽取模型
            # 金融领域事件类型：收购、减持、上市、财报、中标、签约、投资、高管任免等
            self.model = Taskflow(
                'information_extraction',
                schema=[
                    "收购", "减持", "增持", "上市", "中标", "签约",
                    "投资", "高管任命", "高管离职", "发布财报", "业绩预告",
                    "合作", "涨价", "降价", "融资", "分红", "回购"
                ],
                model='uie-base-zh'
            )
            print("  ✅ DuEE 模型加载完成")
        except ImportError:
            print("  ⚠️ PaddleNLP 未安装，使用规则匹配模式")
            self.use_model = False
        except Exception as e:
            print(f"  ⚠️ DuEE 模型加载失败: {e}，使用规则匹配模式")
            self.use_model = False

    def extract(self, texts: List[str]) -> List[List[Dict]]:
        """批量抽取事件"""
        results = []
        for idx, text in enumerate(texts):
            if self.use_model and self.model:
                events = self._extract_with_duee(text, idx)
            else:
                events = self._extract_with_rule(text, idx)
            results.append(events)
        return results

    def _extract_with_duee(self, text: str, text_index: int) -> List[Dict]:
        """使用 DuEE 抽取事件"""
        events = []
        try:
            result = self.model(text)

            if not result:
                return events

            # 解析 DuEE 返回结果
            for item in result:
                if isinstance(item, dict):
                    for event_type, event_data in item.items():
                        # 提取论元
                        args = {}
                        description = event_type

                        if isinstance(event_data, dict):
                            for arg_type, arg_value in event_data.items():
                                if arg_type == "触发词":
                                    description = f"{event_type}({arg_value})"
                                elif isinstance(arg_value, str):
                                    args[arg_type] = arg_value
                                elif isinstance(arg_value, list) and len(arg_value) > 0:
                                    if isinstance(arg_value[0], str):
                                        args[arg_type] = arg_value[0]

                            # 构建完整描述
                            if "公司" in args:
                                description = f"{args['公司']}{event_type}"
                            if "金额" in args:
                                description = f"{description}，金额{args['金额']}"
                            if "比例" in args:
                                description = f"{description}，{args['比例']}"

                        events.append({
                            "event_id": f"e_{text_index}_{len(events)}",
                            "event_type": event_type,
                            "trigger": event_data.get("触发词", "") if isinstance(event_data, dict) else "",
                            "args": args,
                            "description": description,
                            "timestamp": self._extract_date(text),
                            "text_index": text_index,
                            "confidence": 0.85
                        })

        except Exception as e:
            print(f"  ⚠️ DuEE 抽取失败: {e}，使用规则匹配")
            return self._extract_with_rule(text, text_index)

        return events

    def _extract_with_rule(self, text: str, text_index: int) -> List[Dict]:
        """使用规则匹配抽取事件（降级方案）"""
        events = []
        for event_type, triggers in self.RULE_EVENT_TYPES.items():
            for trigger in triggers:
                if trigger in text:
                    args = self._extract_args_by_rule(text, event_type, trigger)
                    description = self._build_description_by_rule(event_type, args)
                    events.append({
                        "event_id": f"e_{text_index}_{len(events)}",
                        "event_type": event_type,
                        "trigger": trigger,
                        "args": args,
                        "description": description,
                        "timestamp": self._extract_date(text),
                        "text_index": text_index,
                        "confidence": 0.6
                    })
                    break
        return events

    def _extract_args_by_rule(self, text: str, event_type: str, trigger: str) -> Dict:
        """规则提取论元"""
        args = {}
        trigger_pos = text.find(trigger)
        if trigger_pos < 0:
            return args

        context = text[max(0, trigger_pos - 80):min(len(text), trigger_pos + 150)]

        company_match = re.search(self.COMPANY_PATTERN, context)
        if company_match:
            company = company_match.group(1)
            if len(company) >= 4:
                args["公司"] = company

        person_match = re.search(self.PERSON_PATTERN, context)
        if person_match:
            person = person_match.group(1)
            if len(person) >= 2:
                args["人物"] = person

        money_match = re.search(self.MONEY_PATTERN, context)
        if money_match:
            args["金额"] = money_match.group(1)

        percent_match = re.search(self.PERCENT_PATTERN, context)
        if percent_match:
            args["比例"] = percent_match.group(1)

        return args

    def _build_description_by_rule(self, event_type: str, args: Dict) -> str:
        """构建事件描述"""
        company = args.get("公司", "")
        money = args.get("金额", "")
        percent = args.get("比例", "")
        person = args.get("人物", "")

        if event_type == "发布财报":
            if company and money:
                return f"{company}发布财报，{money}"
            elif company and percent:
                return f"{company}发布财报，同比增长{percent}"
            elif company:
                return f"{company}发布财报"
        elif event_type == "收购":
            if company and money:
                return f"{company}收购，交易金额{money}"
            elif company:
                return f"{company}收购"
        elif event_type == "投资":
            if company and money:
                return f"{company}投资{money}"
            elif company:
                return f"{company}投资"
        elif event_type == "高管任命":
            if company and person:
                return f"{company}任命{person}"
            elif company:
                return f"{company}高管任命"
        elif event_type == "高管离职":
            if company and person:
                return f"{company}{person}离职"
            elif company:
                return f"{company}高管离职"
        elif event_type in ["涨价", "降价"]:
            if company and percent:
                return f"{company}{event_type}{percent}"
            elif company:
                return f"{company}{event_type}"
        elif company:
            return f"{company}{event_type}"

        return event_type

    def _extract_date(self, text: str) -> str:
        date_match = re.search(self.DATE_PATTERN, text)
        if date_match:
            date_str = date_match.group(1)
            try:
                if '年' in date_str:
                    year = int(re.search(r'(\d{4})', date_str).group(1))
                    if 2000 <= year <= 2030:
                        return date_str
            except:
                pass
        return ""

    def get_event_stats(self, events_results: List[List[Dict]]) -> Dict:
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