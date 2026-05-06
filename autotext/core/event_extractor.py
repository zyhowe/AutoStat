"""
事件抽取模块 - UIE 原生抽取
速度优化：合并 schema + 批量推理
描述优化：直接使用 UIE 抽取的事件描述
"""

from typing import List, Dict, Any, Optional
from collections import Counter
import hashlib
import warnings

warnings.filterwarnings('ignore')


class UIEExtractor:
    """UIE 底层抽取器 - 批量推理"""

    def __init__(self, batch_size: int = 8, max_seq_len: int = 256):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _load_model(self):
        if self._loaded:
            return

        try:
            from paddlenlp.transformers import UIE, AutoTokenizer
            import paddle

            paddle.set_device('cpu')
            self.model = UIE.from_pretrained("uie-base")
            self.tokenizer = AutoTokenizer.from_pretrained("uie-base")
            self.model.eval()
            self._loaded = True
            print("  ✅ UIE 模型加载完成")

        except Exception as e:
            print(f"  ⚠️ UIE 模型加载失败: {e}")
            raise

    def _convert_examples(self, texts: List[str], schema: str):
        inputs = self.tokenizer(
            text=[schema] * len(texts),
            text_pair=texts,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pd"
        )
        return inputs

    def extract_batch(self, texts: List[str], schema: str, threshold: float = 0.51) -> List[List[str]]:
        if not texts:
            return []

        self._load_model()

        try:
            import paddle

            inputs = self._convert_examples(texts, schema)
            with paddle.no_grad():
                start_logits, end_logits = self.model(**inputs)

            start_probs = paddle.sigmoid(start_logits).numpy()
            end_probs = paddle.sigmoid(end_logits).numpy()

            all_tokens = [
                self.tokenizer.convert_ids_to_tokens(ids)
                for ids in inputs["input_ids"].numpy()
            ]

            results = []
            for i in range(len(texts)):
                ents = self._decode(all_tokens[i], start_probs[i], end_probs[i], threshold)
                results.append(ents)

            return results

        except Exception as e:
            return [[] for _ in texts]

    def _decode(self, tokens, start_prob, end_prob, threshold=0.51) -> List[str]:
        ents = []
        seq_len = len(tokens)

        candidates = []
        for i in range(seq_len):
            if start_prob[i] < threshold:
                continue
            for j in range(i, min(i + 30, seq_len)):
                if end_prob[j] >= threshold:
                    conf = min(start_prob[i], end_prob[j])
                    candidates.append((i, j, conf))

        candidates.sort(key=lambda x: x[2], reverse=True)

        used_spans = set()
        for i, j, conf in candidates:
            overlap = False
            for ui, uj in used_spans:
                if not (j < ui or i > uj):
                    overlap = True
                    break

            if not overlap:
                ent = self._reconstruct(tokens[i:j + 1])
                if ent:
                    ents.append(ent)
                    used_spans.add((i, j))

        return ents

    def _reconstruct(self, tokens):
        text = ""
        for t in tokens:
            if t in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            if t.startswith("##"):
                text += t[2:]
            else:
                text += t
        text = text.replace(" ", "")
        return text.strip()


class EventExtractor:
    """事件抽取器 - 完整事件类型"""

    # 完整的事件类型映射表
    EVENT_TYPE_MAP = {
        # 资本市场
        "上市": "上市", "IPO": "上市", "发行": "上市", "登陆": "上市",
        "增发": "增发", "配股": "配股", "可转债": "可转债发行",

        # 股权变动
        "收购": "收购", "并购": "收购", "要约收购": "收购",
        "增持": "增持", "买入": "增持",
        "减持": "减持", "卖出": "减持", "套现": "减持",
        "转让": "股权转让", "股权转让": "股权转让",
        "回购": "股份回购", "回购股份": "股份回购",

        # 投融资
        "投资": "投资", "出资": "投资", "设立": "投资设立",
        "融资": "融资", "募资": "融资", "募集": "融资",

        # 经营业绩
        "发布": "财报发布", "公布": "财报发布", "披露": "财报发布",
        "业绩": "业绩公告", "预告": "业绩预告",
        "分红": "分红派息", "派息": "分红派息", "送股": "送转",

        # 人事变动
        "任命": "高管任命", "聘任": "高管任命", "委任": "高管任命",
        "辞职": "高管离职", "辞任": "高管离职", "离职": "高管离职",
        "卸任": "高管离职", "免去": "高管免职",

        # 业务合作
        "合作": "战略合作", "签约": "签约", "签署": "签约",
        "中标": "中标", "获得": "中标",

        # 市场表现
        "上涨": "股价上涨", "涨停": "涨停", "创新高": "创新高",
        "下跌": "股价下跌", "跌停": "跌停", "创新低": "创新低",
        "涨价": "产品涨价", "提价": "产品涨价",
        "降价": "产品降价", "下调": "产品降价",

        # 监管与法律
        "立案": "立案调查", "调查": "立案调查",
        "处罚": "行政处罚", "罚款": "行政处罚",
        "诉讼": "诉讼", "起诉": "诉讼", "判决": "判决",

        # 调研与会议
        "调研": "机构调研", "接待": "机构调研",
        "会议": "股东大会", "董事会": "董事会会议",

        # 地缘政治
        "停火": "停火", "谈判": "谈判", "协议": "协议签署",
        "制裁": "制裁", "封锁": "封锁",

        # 其他
        "澄清": "澄清公告", "更名": "更名", "停牌": "停牌", "复牌": "复牌"
    }

    def __init__(self, use_model: bool = True, batch_size: int = 8):
        self.use_model = use_model
        self.batch_size = batch_size
        self._extractor = None
        self._cache = {}

        if use_model:
            self._init_extractor()

    def _init_extractor(self):
        try:
            self._extractor = UIEExtractor(batch_size=self.batch_size)
            print("  ✅ UIE 模型加载完成")
        except Exception as e:
            print(f"  ⚠️ UIE 模型加载失败: {e}")
            self.use_model = False

    def extract(self, texts: List[str]) -> List[List[Dict]]:
        """批量抽取事件"""
        if not self.use_model or self._extractor is None:
            return [[] for _ in texts]

        # 过滤空文本
        valid_indices = [i for i, t in enumerate(texts) if t and len(t) > 20]
        valid_texts = [texts[i] for i in valid_indices]

        if not valid_texts:
            return [[] for _ in texts]

        # 检查缓存
        uncached_indices = []
        uncached_texts = []
        results = [[] for _ in texts]

        for orig_idx, text in zip(valid_indices, valid_texts):
            cache_key = hashlib.md5(text[:200].encode()).hexdigest()
            if cache_key in self._cache:
                results[orig_idx] = self._cache[cache_key]
            else:
                uncached_indices.append(orig_idx)
                uncached_texts.append(text)

        if not uncached_texts:
            return results

        # 合并 schema - 增加事件描述
        schema = "触发词、公司、人物、时间、金额、事件描述"

        # 分批处理
        for batch_start in range(0, len(uncached_texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(uncached_texts))
            batch_texts = uncached_texts[batch_start:batch_end]
            batch_indices = uncached_indices[batch_start:batch_end]

            batch_results = self._extractor.extract_batch(batch_texts, schema, threshold=0.51)

            for orig_idx, text, entities in zip(batch_indices, batch_texts, batch_results):
                events = self._parse_entities_to_events(entities, text, orig_idx)
                cache_key = hashlib.md5(text[:200].encode()).hexdigest()
                self._cache[cache_key] = events
                results[orig_idx] = events

        return results

    def _parse_entities_to_events(self, entities: List[str], text: str, text_index: int) -> List[Dict]:
        """将抽取的实体转换为事件 - 单文本内去重"""
        if not entities:
            return []

        # 分类实体
        triggers = []
        companies = []
        persons = []
        times = []
        moneys = []
        descriptions = []

        for ent in entities:
            ent_clean = ent.replace(" ", "")
            if len(ent_clean) < 2:
                continue

            # 跳过垃圾词
            if ent_clean in ["触发词", "事件", "触发"]:
                continue

            # 事件描述：通常是较长的文本（>10字符）或包含"描述"
            if len(ent_clean) > 10 or "描述" in ent_clean:
                descriptions.append(ent_clean)
            # 公司识别
            elif any(ent_clean.endswith(suffix) for suffix in ["公司", "集团", "股份", "有限", "银行", "证券", "基金", "保险"]):
                companies.append(ent_clean)
            # 人名识别
            elif any(ent_clean.endswith(suffix) for suffix in ["总", "长", "理", "裁", "董", "先生", "女士", "董事长", "总经理", "CEO"]):
                persons.append(ent_clean)
            # 时间识别
            elif any(c in ent_clean for c in ["年", "月", "日", "周", "季度", "季", "时", "分"]):
                times.append(ent_clean)
            # 金额识别
            elif any(c in ent_clean for c in ["亿", "万", "元", "%", "美元", "港元", "欧元", "英镑"]):
                moneys.append(ent_clean)
            else:
                # 可能是触发词
                triggers.append(ent_clean)

        # 如果没有触发词，从文本提取
        if not triggers:
            triggers = self._extract_event_words(text)

        # 构建事件 - 单文本内去重
        events = []
        seen_in_text = set()

        for trigger in triggers[:5]:
            event_type = self._get_event_type(trigger)

            # 构造 args
            args = {}
            if companies:
                args["公司"] = companies[0]
            if persons:
                args["人物"] = persons[0]
            if times:
                args["时间"] = times[0]
            if moneys:
                args["金额"] = moneys[0]

            # 单文本内去重
            key = f"{event_type}|{args.get('公司', '')}|{args.get('人物', '')}"
            if key in seen_in_text:
                continue
            seen_in_text.add(key)

            # 描述：优先使用 UIE 抽取的事件描述，没有则用触发词
            description = descriptions[0] if descriptions else trigger

            events.append({
                "event_id": f"e_{text_index}_{abs(hash(trigger)) % 10000}",
                "event_type": event_type,
                "trigger": trigger,
                "args": args,
                "description": description,
                "timestamp": args.get("时间", ""),
                "text_index": text_index,
                "confidence": 0.85
            })

        return events

    def _extract_event_words(self, text: str) -> List[str]:
        """从文本中提取候选事件词"""
        event_words = [
            "收购", "投资", "上市", "增持", "减持", "回购", "分红", "融资",
            "中标", "签约", "合作", "任命", "辞职", "辞任", "离职", "聘任",
            "发布", "公布", "披露", "预告", "涨停", "跌停", "上涨", "下跌",
            "涨价", "降价", "转让", "立案", "调研", "入主", "谈判", "停火"
        ]

        found = []
        for word in event_words:
            if word in text and word not in found:
                found.append(word)
                if len(found) >= 3:
                    break
        return found

    def _get_event_type(self, trigger: str) -> str:
        """获取事件类型"""
        for key, value in self.EVENT_TYPE_MAP.items():
            if key in trigger:
                return value
        return trigger[:8] if len(trigger) > 8 else trigger

    def get_event_stats(self, events_results: List[List[Dict]]) -> Dict:
        """获取事件统计"""
        event_by_type = Counter()
        for events in events_results:
            for event in events:
                event_by_type[event.get("event_type", "未知")] += 1

        return {
            "total_events": sum(event_by_type.values()),
            "event_types": dict(event_by_type),
            "top_events": event_by_type.most_common(20)
        }

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()