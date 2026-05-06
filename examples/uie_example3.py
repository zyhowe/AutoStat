"""
✅ UIE 长文章事件抽取
支持多事件、跨句子、滑动窗口处理
"""
from paddlenlp.transformers import UIE, AutoTokenizer
import paddle
import re


# ====================== 底层工具 ======================
def convert_example(text, schema, tokenizer, max_seq_len=512):
    inputs = tokenizer(
        text=schema,
        text_pair=text,
        max_length=max_seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pd"
    )
    return inputs


class UIEExtractor:
    def __init__(self):
        self.model = UIE.from_pretrained("uie-base")
        self.tokenizer = AutoTokenizer.from_pretrained("uie-base")
        self.model.eval()
        self.max_seq_len = 512

    def extract(self, text, schema, threshold=0.51):
        """单段文本抽取"""
        inputs = convert_example(text, schema, self.tokenizer, self.max_seq_len)
        with paddle.no_grad():
            start_logits, end_logits = self.model(**inputs)
        start = paddle.sigmoid(start_logits)[0].numpy()
        end = paddle.sigmoid(end_logits)[0].numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].numpy())
        return self._decode(tokens, start, end, threshold)

    def _decode(self, tokens, start_prob, end_prob, threshold=0.51):
        """解码实体"""
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
            # 检查重叠
            overlap = False
            for ui, uj in used_spans:
                if not (j < ui or i > uj):
                    overlap = True
                    break

            if not overlap:
                ent = self._reconstruct(tokens[i:j + 1])
                if ent and len(ent) >= 1:
                    ents.append(ent)
                    used_spans.add((i, j))

        return ents

    def _reconstruct(self, tokens):
        """还原 text"""
        text = ""
        for t in tokens:
            if t in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            if t.startswith("##"):
                text += t[2:]
            else:
                if text and not text.endswith(" "):
                    text += " "
                text += t
        return text.strip()


# ====================== 长文章事件抽取器 ======================
class LongDocumentEventExtractor:
    def __init__(self, window_size=400, overlap=100):
        """
        window_size: 滑动窗口大小（字符数）
        overlap: 窗口重叠大小
        """
        self.uie = UIEExtractor()
        self.window_size = window_size
        self.overlap = overlap

    def split_text(self, text):
        """将长文本切分为多个窗口"""
        windows = []
        step = self.window_size - self.overlap

        for i in range(0, len(text), step):
            window = text[i:i + self.window_size]
            if len(window) < 50:  # 太短就不切了
                continue
            windows.append({
                "text": window,
                "start_pos": i,
                "end_pos": i + len(window)
            })

        # 如果文本本身不长，直接返回原文本
        if not windows:
            windows = [{"text": text, "start_pos": 0, "end_pos": len(text)}]

        return windows

    def merge_entities(self, all_entities, window_info):
        """
        合并重叠的实体
        all_entities: [(entity_text, window_index, entity_start, entity_end), ...]
        """
        # 简单去重
        unique_entities = {}
        for entity in all_entities:
            entity_text = entity[0]
            if entity_text not in unique_entities:
                unique_entities[entity_text] = entity

        return list(unique_entities.values())

    def extract_events_from_long_text(self, long_text, schemas):
        """
        从长文章中抽取所有事件

        schemas: {
            "事件类型": ["触发词"],
            "公司": ["公司"],
            "人物": ["人物"],
            ...
        }
        或者直接传列表: ["触发词", "公司", "人物", "时间", "金额"]
        """
        # 支持两种输入格式
        if isinstance(schemas, list):
            schemas = {s: [s] for s in schemas}

        # 切分文本
        windows = self.split_text(long_text)

        # 存储所有结果
        all_results = {
            "事件类型": [],
            "公司": [],
            "人物": [],
            "时间": [],
            "金额": []
        }

        # 记录每个实体出现的窗口和位置，用于去重
        all_entities = {k: [] for k in all_results.keys()}

        # 逐窗口抽取
        for idx, window in enumerate(windows):
            window_text = window["text"]

            for schema_name, schema_list in schemas.items():
                # 对每个 schema 进行抽取
                if schema_name not in all_entities:
                    continue

                for schema in schema_list:
                    entities = self.uie.extract(window_text, schema)
                    for ent in entities:
                        all_entities[schema_name].append({
                            "entity": ent,
                            "window_idx": idx,
                            "window_start": window["start_pos"],
                            "window_end": window["end_pos"],
                            "schema": schema
                        })

        # 去重合并
        for schema_name in all_entities:
            seen = set()
            for item in all_entities[schema_name]:
                if item["entity"] not in seen:
                    seen.add(item["entity"])
                    all_results[schema_name].append(item["entity"])

        return all_results

    def extract_events_with_context(self, long_text, schemas):
        """
        带上下文的事件抽取（保留事件-论元关联）
        """
        windows = self.split_text(long_text)

        # 存储事件结构
        events = []

        for idx, window in enumerate(windows):
            window_text = window["text"]

            # 1. 抽取触发词（事件类型）
            triggers = self.uie.extract(window_text, "触发词")

            if not triggers:
                continue

            # 2. 抽取该窗口内的所有论元
            arguments = {
                "公司": self.uie.extract(window_text, "公司"),
                "人物": self.uie.extract(window_text, "人物"),
                "时间": self.uie.extract(window_text, "时间"),
                "金额": self.uie.extract(window_text, "金额")
            }

            # 3. 为每个触发词创建事件（简单启发式：同窗口内的论元归为该事件）
            for trigger in triggers:
                event = {
                    "event_type": trigger,
                    "trigger": trigger,
                    "arguments": {},
                    "position": {
                        "window": idx,
                        "text_start": window["start_pos"],
                        "text_end": window["end_pos"]
                    }
                }

                # 添加论元
                for arg_type, arg_values in arguments.items():
                    if arg_values:
                        event["arguments"][arg_type] = arg_values

                events.append(event)

        # 跨窗口合并相同事件（基于触发词和位置相近）
        merged_events = self._merge_cross_window_events(events)

        return merged_events

    def _merge_cross_window_events(self, events):
        """合并跨窗口的相同事件"""
        merged = {}

        for event in events:
            key = f"{event['event_type']}_{event['position']['window'] // 2}"

            if key not in merged:
                merged[key] = event
            else:
                # 合并论元
                for arg_type, arg_values in event["arguments"].items():
                    if arg_type not in merged[key]["arguments"]:
                        merged[key]["arguments"][arg_type] = []
                    for val in arg_values:
                        if val not in merged[key]["arguments"][arg_type]:
                            merged[key]["arguments"][arg_type].append(val)

        return list(merged.values())


# ====================== 使用示例 ======================
def main():
    # 长文章示例
    long_article = """
    2024年3月15日，阿里巴巴集团宣布以50亿元人民币收购饿了么的全部股份。
    此次收购将加强阿里巴巴在本地生活服务领域的布局。

    与此同时，腾讯公司也在同一天宣布投资10亿美元给美团点评，用于发展AI技术。
    腾讯总裁刘炽平表示，这笔投资将帮助美团提升用户体验。

    另外，字节跳动在2024年1月完成了对游戏公司沐瞳科技的收购，交易金额为40亿美元。
    字节跳动CEO梁汝波表示，这将加速公司在游戏领域的扩张。

    华为技术有限公司于2024年2月任命余承东为智能汽车解决方案BU CEO。
    余承东将直接向华为轮值董事长徐直军汇报工作。

    特斯拉中国在2024年3月1日宣布Model 3降价2万元人民币，Model Y降价3万元。
    这是特斯拉今年以来的第二次降价。

    宁德时代新能源科技股份有限公司发布2023年年报，全年营收达到847亿元人民币，
    同比增长22%，净利润为128亿元。

    小米集团2023年第四季度营收732亿元，同比增长10.9%。
    小米创始人雷军表示，2024年将重点投入汽车业务。
    """

    extractor = LongDocumentEventExtractor(window_size=400, overlap=100)

    print("=" * 80)
    print("📰 长文章事件抽取结果")
    print("=" * 80)

    # 方式1：简单抽取（只抽取实体，不区分事件关联）
    print("\n【方式1：简单实体抽取】")
    results = extractor.extract_events_from_long_text(
        long_article,
        ["触发词", "公司", "人物", "时间", "金额"]
    )

    for schema_name, entities in results.items():
        if entities:
            print(f"\n{schema_name}:")
            for ent in entities:
                print(f"  - {ent}")

    # 方式2：带上下文的事件抽取（保留事件-论元关联）
    print("\n" + "=" * 80)
    print("【方式2：结构化事件抽取】")
    print("=" * 80)

    events = extractor.extract_events_with_context(
        long_article,
        ["触发词", "公司", "人物", "时间", "金额"]
    )

    for i, event in enumerate(events, 1):
        print(f"\n📌 事件 {i}: {event['event_type']}")
        print(f"   触发词: {event['trigger']}")
        if event["arguments"]:
            print("   相关论元:")
            for arg_type, arg_values in event["arguments"].items():
                print(f"     • {arg_type}: {', '.join(arg_values)}")
        print(f"   位置: 窗口 {event['position']['window']}")


if __name__ == "__main__":
    main()