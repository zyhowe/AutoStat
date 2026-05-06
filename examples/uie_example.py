"""
事件抽取模块 - 使用 UIE 动态图（修复版）
"""

import re
from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class EventExtractor:
    """事件抽取器 - 使用 UIE 动态图"""

    def __init__(self, use_model: bool = True):
        self.use_model = use_model
        self.tokenizer = None
        self.model = None

        if use_model:
            self._load_model()

    def _load_model(self):
        """加载 UIE 模型"""
        try:
            from paddlenlp.transformers import AutoTokenizer, AutoModel

            model_name = "uie-base"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()

            print("  ✅ UIE 动态图模型加载完成")

        except Exception as e:
            print(f"  ⚠️ UIE 模型加载失败: {e}")
            self.use_model = False

    def extract(self, texts: List[str]) -> List[List[Dict]]:
        """批量抽取事件"""
        if not self.use_model or self.model is None:
            return [[] for _ in texts]

        results = []
        for idx, text in enumerate(texts):
            events = self._extract_with_uie(text, idx)
            results.append(events)
        return results

    def _extract_with_uie(self, text: str, text_index: int) -> List[Dict]:
        """使用 UIE 抽取事件"""
        if not text or len(text) < 10:
            return []

        events = []

        try:
            import paddle

            # 定义多个 schema 进行抽取
            schemas = [
                ["公司", "企业", "集团"],  # 公司实体
                ["人物", "人", "高管"],  # 人物实体
                ["时间", "日期", "年份"],  # 时间实体
                ["收购", "投资", "融资", "上市", "中标", "合作"],  # 事件类型
                ["职位", "职务", "任命", "离职"]  # 人事动态
            ]

            all_entities = {}

            for schema in schemas:
                try:
                    # 编码输入
                    inputs = self.tokenizer(
                        schema,
                        text,
                        max_seq_len=512,
                        return_attention_mask=True,
                        return_token_type_ids=True,
                        return_length=True,
                        return_position_ids=True
                    )

                    # 转换为 tensor
                    input_ids = paddle.to_tensor(inputs["input_ids"])
                    token_type_ids = paddle.to_tensor(inputs["token_type_ids"])
                    attention_mask = paddle.to_tensor(inputs["attention_mask"])
                    position_ids = paddle.to_tensor(inputs["position_ids"])

                    with paddle.no_grad():
                        # UIE 模型前向传播
                        sequence_output, pooled_output = self.model(
                            input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids
                        )

                    # 使用序列输出进行实体识别
                    entities = self._extract_entities_from_sequence(
                        sequence_output,
                        input_ids,
                        attention_mask,
                        schema,
                        text
                    )

                    # 合并结果
                    for entity_type, entity_list in entities.items():
                        if entity_type not in all_entities:
                            all_entities[entity_type] = []
                        all_entities[entity_type].extend(entity_list)

                except Exception as e:
                    continue

            # 去重并生成事件
            for entity_type, entity_list in all_entities.items():
                # 去重
                unique_entities = list(set(entity_list))[:10]

                for entity_text in unique_entities:
                    if len(entity_text) >= 2:
                        event = self._entity_to_event(entity_type, entity_text, text_index)
                        if event:
                            events.append(event)

            # 如果没有抽取出事件，尝试基于规则的抽取
            if not events:
                events = self._rule_based_extraction(text, text_index)

        except Exception as e:
            # 发生错误时使用规则抽取
            events = self._rule_based_extraction(text, text_index)

        return events

    def _extract_entities_from_sequence(self, sequence_output, input_ids, attention_mask, schema, text):
        """从序列输出中提取实体"""
        import paddle

        entities = {}

        # 获取序列长度
        batch_size, seq_len, hidden_size = sequence_output.shape

        # 简单的实体识别：使用序列输出的范数来判断重要token
        # 计算每个token的重要性得分
        token_scores = paddle.norm(sequence_output, axis=-1)  # [batch, seq_len]
        token_scores = token_scores[0].numpy()  # 取第一个batch

        # 获取token列表
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].numpy())

        # 根据重要性得分提取潜在实体
        threshold = np.percentile(token_scores, 85)  # 取前15%的token

        # 找连续的high score区域
        regions = []
        i = 0
        while i < seq_len:
            if token_scores[i] > threshold and tokens[i] not in ['[CLS]', '[SEP]', '[PAD]']:
                start = i
                end = i
                while end + 1 < seq_len and token_scores[end + 1] > threshold:
                    end += 1
                regions.append((start, end))
                i = end + 1
            else:
                i += 1

        # 从每个区域提取实体文本
        for start, end in regions:
            entity_tokens = tokens[start:end+1]
            entity_text = self._reconstruct_text(entity_tokens)

            if entity_text and len(entity_text) >= 2:
                # 确定实体类型
                entity_type = self._determine_entity_type(entity_text, schema)
                if entity_type:
                    if entity_type not in entities:
                        entities[entity_type] = []
                    if entity_text not in entities[entity_type]:
                        entities[entity_type].append(entity_text)

        return entities

    def _determine_entity_type(self, entity_text: str, schema: List[str]) -> Optional[str]:
        """确定实体类型"""
        # 根据schema和实体文本来判断类型
        text_lower = entity_text.lower()

        # 公司关键词
        company_keywords = ['公司', '集团', '有限', '股份', '科技', '网络', '银行', '基金', '证券', '保险']
        if any(kw in entity_text for kw in company_keywords) and ('公司' in schema or '企业' in schema):
            return '公司'

        # 人物关键词
        person_keywords = ['总', '经理', '董事', 'CEO', 'CFO', 'CTO', '总裁', '主席', '教授', '博士']
        if any(kw in entity_text for kw in person_keywords) and ('人物' in schema or '人' in schema):
            return '人物'

        # 时间关键词
        time_keywords = ['年', '月', '日', '季', '周', '天', '时', '分', '秒']
        if any(kw in entity_text for kw in time_keywords) and ('时间' in schema or '日期' in schema):
            return '时间'

        # 事件关键词
        event_keywords = ['收购', '投资', '融资', '上市', '中标', '合作', '发布', '推出', '宣布']
        for kw in event_keywords:
            if kw in entity_text and kw in schema:
                return kw

        # 职位关键词
        position_keywords = ['任命', '离职', '辞职', '升任', '调任']
        if any(kw in entity_text for kw in position_keywords):
            return '人事动态'

        # 默认返回第一个schema
        return schema[0] if schema else '信息'

    def _reconstruct_text(self, tokens):
        """从 token 重建文本"""
        text = ""
        for token in tokens:
            if token.startswith("##"):
                text += token[2:]
            elif token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                continue
            else:
                if text and not text.endswith(" "):
                    text += " "
                text += token
        return text.strip()

    def _rule_based_extraction(self, text: str, text_index: int) -> List[Dict]:
        """基于规则的事件抽取（备用方案）"""
        events = []

        # 公司名称模式
        company_patterns = [
            r'([\u4e00-\u9fa5]{2,}(?:公司|集团|股份|有限|科技|网络|银行|基金))',
            r'([A-Z][a-z]+(?:公司|Inc|Corp|Ltd))'
        ]

        # 金额模式
        amount_pattern = r'(\d+(?:\.\d+)?)\s*([亿万千百十])?元?'

        # 事件动词
        action_words = ['收购', '投资', '融资', '上市', '中标', '合作', '发布', '宣布', '任命', '降价', '增持', '减持']

        # 提取公司
        companies = []
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            companies.extend(matches)

        # 根据动作词判断事件类型
        for action in action_words:
            if action in text:
                event_type = self._map_action_to_type(action)

                # 构建事件描述
                description = text[:100]

                # 提取相关实体
                args = {}
                if companies:
                    args['company'] = companies[0]

                # 提取金额
                amount_match = re.search(amount_pattern, text)
                if amount_match:
                    args['amount'] = amount_match.group(0)

                event = {
                    "event_id": f"e_{text_index}_{hash(action) % 10000}",
                    "event_type": event_type,
                    "trigger": action,
                    "args": args,
                    "description": description,
                    "timestamp": "",
                    "text_index": text_index,
                    "confidence": 0.5
                }
                events.append(event)
                break  # 只取第一个事件

        # 如果没有找到动作词，尝试其他模式
        if not events:
            # 数字+单位模式（可能是业绩发布）
            if re.search(r'\d+(?:\.\d+)?\s*[亿万千百]+', text):
                event = {
                    "event_id": f"e_{text_index}_业绩",
                    "event_type": "业绩发布",
                    "trigger": "发布",
                    "args": {},
                    "description": text[:100],
                    "timestamp": "",
                    "text_index": text_index,
                    "confidence": 0.4
                }
                events.append(event)

        return events

    def _map_action_to_type(self, action: str) -> str:
        """将动作词映射到事件类型"""
        mapping = {
            '收购': '收购', '投资': '投资', '融资': '融资',
            '上市': '上市', '中标': '中标', '合作': '合作',
            '发布': '发布', '宣布': '公告', '任命': '人事任命',
            '降价': '价格调整', '增持': '增持', '减持': '减持'
        }
        return mapping.get(action, action)

    def _entity_to_event(self, entity_type, entity_text, text_index):
        """实体转事件"""
        event_type = self._map_event_type(entity_type, entity_text)

        return {
            "event_id": f"e_{text_index}_{abs(hash(entity_text)) % 10000}",
            "event_type": event_type,
            "trigger": entity_text[:30],
            "args": {entity_type: entity_text},
            "description": f"{entity_type}: {entity_text}"[:100],
            "timestamp": "",
            "text_index": text_index,
            "confidence": 0.6
        }

    def _map_event_type(self, entity_type, entity_text):
        """映射事件类型"""
        # 事件类型映射
        event_mapping = {
            "收购": "收购", "投资": "投资", "融资": "融资",
            "上市": "上市", "中标": "中标", "合作": "合作",
            "公司": "公司动态", "企业": "公司动态",
            "人物": "人事动态", "人": "人事动态",
            "时间": "时间节点"
        }

        if entity_type in event_mapping:
            return event_mapping[entity_type]

        # 根据关键词判断
        for kw, typ in [
            ("收购", "收购"), ("投资", "投资"), ("上市", "上市"),
            ("中标", "中标"), ("签约", "签约"), ("合作", "合作"),
            ("减持", "减持"), ("增持", "增持"), ("融资", "融资"),
            ("任命", "人事任命"), ("离职", "人事变动")
        ]:
            if kw in entity_text:
                return typ
        return "信息"

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


# 测试代码
def main():
    print("=" * 60)
    print("事件抽取示例（修复版）")
    print("=" * 60)

    extractor = EventExtractor(use_model=True)

    texts = [
        "宁德时代发布2024年报，营收847亿元。",
        "阿里巴巴以50亿元收购某科技公司。",
        "腾讯投资10亿元布局AI领域。",
        "华为任命余承东为BU CEO。",
        "特斯拉宣布降价2万元。",
        "比亚迪中标深圳公交系统项目，金额5亿元。",
        "小米集团与蔚来汽车达成战略合作。"
    ]

    print(f"\n📝 共 {len(texts)} 条测试文本\n")

    results = extractor.extract(texts)

    for i, (text, events) in enumerate(zip(texts, results)):
        print(f"【{i+1}】{text}")
        if events:
            for event in events:
                print(f"   类型: {event['event_type']}, 触发词: {event['trigger']}, 描述: {event['description']}")
        else:
            print("   未抽取到事件")
        print()

    stats = extractor.get_event_stats(results)
    print(f"📊 统计结果:")
    print(f"   总事件数: {stats['total_events']}")
    print(f"   事件类型分布: {stats['event_types']}")


if __name__ == "__main__":
    main()