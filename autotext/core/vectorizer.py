# autotext/core/vectorizer.py

import os
import numpy as np
import torch
from typing import List, Optional, Dict, Any
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 设置国内镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class BertVectorizer:
    """BERT向量化器"""

    def __init__(self, model_name: str = "bert-base-chinese", device: str = "cpu",
                 cache_dir: Optional[str] = None, offline: bool = False):
        self.model_name = model_name
        self.device = device
        self.offline = offline

        if offline:
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            print("  📴 离线模式已启用")

        print(f"  🔄 加载BERT模型: {model_name}...")

        try:
            # 关键：直接导入，不要用 try-except 包裹
            from transformers import AutoTokenizer, AutoModel

            if cache_dir:
                self.cache_dir = Path(cache_dir)
            else:
                self.cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

            self.cache_dir.mkdir(parents=True, exist_ok=True)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                local_files_only=offline
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                local_files_only=offline
            )
            self.model.to(device)
            self.model.eval()

            print(f"  ✅ BERT模型加载完成")

        except Exception as e:
            print(f"  ❌ 模型加载失败: {e}")
            print(f"  💡 提示: 请先运行 download_bert_model.py 下载模型")
            raise

        from .cache import CacheManager
        self.cache = CacheManager()
        self._embedding_index, self._cached_embeddings = self.cache.load_embeddings_cache()
        self._next_index = len(self._embedding_index) if self._embedding_index else 0



    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """获取单条文本的BERT向量"""
        # 检查缓存
        text_hash = self.cache._get_text_hash(text)
        if self._embedding_index and text_hash in self._embedding_index:
            idx = self._embedding_index[text_hash]
            if self._cached_embeddings is not None and idx < len(self._cached_embeddings):
                return self._cached_embeddings[idx]

        # 计算向量
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用 [CLS] token 向量
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

        return embedding

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量获取文本向量（带缓存）"""
        embeddings = []

        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []

            for text in batch:
                # 检查缓存
                text_hash = self.cache._get_text_hash(text)
                if self._embedding_index and text_hash in self._embedding_index:
                    idx = self._embedding_index[text_hash]
                    if self._cached_embeddings is not None and idx < len(self._cached_embeddings):
                        batch_embeddings.append(self._cached_embeddings[idx])
                        continue

                # 计算新向量
                emb = self._get_bert_embedding(text)
                batch_embeddings.append(emb)

                # 更新缓存索引
                if self._embedding_index is None:
                    self._embedding_index = {}
                self._embedding_index[text_hash] = self._next_index
                self._next_index += 1

            embeddings.extend(batch_embeddings)

        # 合并所有向量
        embeddings = np.array(embeddings)

        # 保存缓存
        if self._cached_embeddings is not None:
            all_embeddings = np.vstack([self._cached_embeddings, embeddings])
        else:
            all_embeddings = embeddings

        if self._embedding_index:
            self.cache.save_embeddings_cache(self._embedding_index, all_embeddings)
        self._cached_embeddings = all_embeddings

        return embeddings

    def is_model_cached(self) -> bool:
        """检查模型是否已缓存"""
        cache_path = self.cache_dir / f"models--{self.model_name.replace('/', '--')}"
        return cache_path.exists()