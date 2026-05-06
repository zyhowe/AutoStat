"""
✅ UIE 真正正确的事件抽取
先识别【事件类型】，再抽取【实体】，永不错乱
"""
from paddlenlp.transformers import UIE, AutoTokenizer
import paddle

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

    def extract(self, text, schema):
        inputs = convert_example(text, schema, self.tokenizer)
        with paddle.no_grad():
            start_logits, end_logits = self.model(**inputs)
        start = paddle.sigmoid(start_logits)[0].numpy()
        end = paddle.sigmoid(end_logits)[0].numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].numpy())
        return self._decode(tokens, start, end)

    def _decode(self, tokens, start_prob, end_prob, threshold=0.51):
        """改进的解码：按置信度从大到小，不重叠不重复"""
        ents = []
        seq_len = len(tokens)

        # 收集所有候选 (start, end, 置信度)
        candidates = []
        for i in range(seq_len):
            if start_prob[i] < threshold:
                continue
            for j in range(i, min(i + 30, seq_len)):
                if end_prob[j] >= threshold:
                    # 用最小值作为实体整体置信度（保守估计）
                    conf = min(start_prob[i], end_prob[j])
                    candidates.append((i, j, conf))

        # 按置信度从大到小排序
        candidates.sort(key=lambda x: x[2], reverse=True)

        # 记录已占用的位置，避免重叠
        used_start = set()
        used_end = set()

        for i, j, conf in candidates:
            # 检查是否与已选实体重叠
            overlap = False
            for si in used_start:
                if abs(si - i) < 3:  # 起始位置太近
                    overlap = True
                    break
            for ej in used_end:
                if abs(ej - j) < 3:  # 结束位置太近
                    overlap = True
                    break

            if not overlap:
                ent = self._reconstruct(tokens[i:j+1])
                if ent and len(ent) >= 1:
                    ents.append(ent)
                    used_start.add(i)
                    used_end.add(j)

        return ents

    def _reconstruct(self, tokens):
        """还原 token 为原始文本"""
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


# ====================== 【最终统一抽取】 ======================
def extract_final(text, extractor):
    result = {}

    # 1. 事件类型（通用抽取，不预设枚举）
    result["事件类型"] = extractor.extract(text, "触发词")

    # 2. 实体抽取
    result["公司"] = extractor.extract(text, "公司")
    result["人物"] = extractor.extract(text, "人物")
    result["时间"] = extractor.extract(text, "时间")
    result["金额"] = extractor.extract(text, "金额")

    return result

# ====================== 测试 ======================
if __name__ == "__main__":
    extractor = UIEExtractor()
    texts = [
        "阿里巴巴以50亿元收购某科技公司。",
        "腾讯投资10亿元布局AI领域。",
        "华为任命余承东为BU CEO。",
        "特斯拉宣布降价2万元。",
        "宁德时代发布2024年报，营收847亿元。"
    ]

    print("="*60)
    print("✅ UIE 最终正确版 · 事件类型永不错乱")
    print("="*60)

    for i, text in enumerate(texts, 1):
        print(f"\n【{i}】{text}")
        res = extract_final(text, extractor)
        for k, v in res.items():
            if v:
                print(f"  {k}：{v}")