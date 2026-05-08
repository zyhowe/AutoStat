"""
事件抽取模块 - 基于 UIE，只抽取事件描述
"""

from typing import List, Dict, Any
from collections import Counter
import hashlib
import warnings

warnings.filterwarnings('ignore')


class UIEExtractor:
    """UIE 底层抽取器"""

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

    def extract_batch(self, texts: List[str], schema: str, threshold: float = 0.5) -> List[List[str]]:
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
            print(f"  ⚠️ UIE 批量抽取失败: {e}")
            return [[] for _ in texts]

    def _decode(self, tokens, start_prob, end_prob, threshold=0.5) -> List[str]:
        ents = []
        seq_len = len(tokens)

        start_positions = [i for i, p in enumerate(start_prob) if p > threshold]

        for start in start_positions:
            for end in range(start, min(start + 30, seq_len)):
                if end_prob[end] > threshold:
                    ent = self._reconstruct(tokens[start:end+1])
                    if ent and len(ent) >= 2 and ent not in ents:
                        ents.append(ent)
                    break

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
    """事件抽取器 - 只保留事件描述"""

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
        """批量抽取事件，返回事件描述"""
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

        # 只抽取"事件" schema
        schema = "事件"

        # 分批处理
        for batch_start in range(0, len(uncached_texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(uncached_texts))
            batch_texts = uncached_texts[batch_start:batch_end]
            batch_indices = uncached_indices[batch_start:batch_end]

            batch_results = self._extractor.extract_batch(batch_texts, schema, threshold=0.5)

            for orig_idx, text, events_desc in zip(batch_indices, batch_texts, batch_results):
                events = []
                for desc in events_desc:
                    if desc and len(desc) > 5:
                        events.append({
                            "event_id": f"e_{orig_idx}_{abs(hash(desc)) % 10000}",
                            "description": desc,
                            "text_index": orig_idx
                        })

                cache_key = hashlib.md5(text[:200].encode()).hexdigest()
                self._cache[cache_key] = events
                results[orig_idx] = events

        return results

    def get_event_stats(self, events_results: List[List[Dict]]) -> Dict:
        """获取事件统计"""
        event_by_desc = Counter()
        for events in events_results:
            for event in events:
                desc = event.get("description", "")[:20]
                if desc:
                    event_by_desc[desc] += 1

        return {
            "total_events": sum(event_by_desc.values()),
            "top_events": event_by_desc.most_common(20)
        }

    def clear_cache(self):
        self._cache.clear()