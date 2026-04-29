"""
信息抽取模块 - 实体-属性-关系抽取（基于 spaCy）
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


class InfoExtractor:
    """信息抽取器 - 实体属性、关系、事件归属"""

    def __init__(self):
        self.nlp = None
        self._load_model()

    def _load_model(self):
        """加载 spaCy 模型"""
        try:
            import spacy
            # 尝试加载中文模型
            try:
                self.nlp = spacy.load("zh_core_web_trf")
            except OSError:
                try:
                    self.nlp = spacy.load("zh_core_web_sm")
                except OSError:
                    print("⚠️ spaCy 中文模型未安装，请运行: python -m spacy download zh_core_web_trf")
                    self.nlp = None
        except ImportError:
            print("⚠️ spaCy 未安装，信息抽取功能不可用")
            self.nlp = None

    def is_available(self) -> bool:
        return self.nlp is not None

    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        从单条文本中抽取信息

        返回: {
            "entities": [{"text": "宁德时代", "type": "ORG", "start": 0, "end": 4}],
            "attributes": [{"entity": "宁德时代", "attr": "营收", "value": "847亿元"}],
            "relations": [{"subject": "宁德时代", "predicate": "发布", "object": "财报"}],
            "event": "财报发布"
        }
        """
        if not self.is_available() or not text:
            return {"entities": [], "attributes": [], "relations": [], "event": ""}

        doc = self.nlp(text[:2000])  # 限制长度

        # 1. 实体抽取
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

        # 2. 属性抽取（数值修饰关系）
        attributes = []
        for token in doc:
            # 数值修饰（nummod）: 金额、数量等
            if token.dep_ == "nummod" and token.head.pos_ == "NOUN":
                attr_name = token.head.text
                attr_value = token.text
                # 找到所属实体
                entity = self._find_entity_for_token(token.head, entities)
                if entity:
                    attributes.append({
                        "entity": entity["text"],
                        "attr": attr_name,
                        "value": attr_value
                    })

        # 3. 关系抽取（主语-谓语-宾语）
        relations = []
        for token in doc:
            if token.dep_ == "nsubj":  # 名词性主语
                subject = token.text
                # 找谓语和宾语
                for child in token.head.children:
                    if child.dep_ == "dobj":  # 直接宾语
                        predicate = token.head.text
                        obj = child.text
                        relations.append({
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj
                        })
                    elif child.dep_ == "attr":  # 表语
                        predicate = "是"
                        obj = child.text
                        relations.append({
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj
                        })

        # 4. 事件识别（基于动词和实体组合）
        event = self._identify_event(doc, entities)

        return {
            "entities": entities,
            "attributes": attributes,
            "relations": relations,
            "event": event
        }

    def _find_entity_for_token(self, token, entities: List[Dict]) -> Optional[Dict]:
        """找到 token 所属的实体"""
        for ent in entities:
            if ent["start"] <= token.idx < ent["end"]:
                return ent
        return None

    def _identify_event(self, doc, entities: List[Dict]) -> str:
        """识别事件类型"""
        # 常见事件关键词
        event_keywords = {
            "发布": "发布",
            "宣布": "宣布",
            "收购": "收购",
            "投资": "投资",
            "合作": "合作",
            "签约": "签约",
            "上市": "上市",
            "增持": "增持",
            "减持": "减持",
            "财报": "财报发布",
            "业绩": "财报发布",
            "涨停": "涨停",
            "跌停": "跌停",
            "停牌": "停牌",
            "复牌": "复牌"
        }

        for token in doc:
            if token.text in event_keywords:
                return event_keywords[token.text]

        # 没有匹配的关键词，返回空
        return ""

    def extract_batch(self, texts: List[str]) -> List[Dict]:
        """批量抽取"""
        results = []
        for text in texts:
            results.append(self.extract_from_text(text))
        return results

    def aggregate_by_topic(self, extraction_results: List[Dict],
                            cluster_labels: List[int]) -> Dict[int, List[Dict]]:
        """按主题/聚类聚合抽取结果"""
        topic_events = defaultdict(list)

        for idx, result in enumerate(extraction_results):
            if idx < len(cluster_labels) and cluster_labels[idx] != -1:
                topic_id = cluster_labels[idx]
                if result["event"]:
                    topic_events[topic_id].append({
                        "event": result["event"],
                        "entities": result["entities"],
                        "attributes": result["attributes"],
                        "relations": result["relations"]
                    })

        return dict(topic_events)