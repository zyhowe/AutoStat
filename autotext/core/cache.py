"""
缓存管理模块 - 向量和中间结果持久化
"""

import hashlib
import json
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


class CacheManager:
    """统一缓存管理器"""

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = Path.home() / ".autotext" / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 缓存文件
        self.embeddings_file = self.cache_dir / "embeddings.npz"
        self.embeddings_index_file = self.cache_dir / "embeddings_index.json"
        self.bert_outputs_file = self.cache_dir / "bert_outputs.pkl"

    def _get_text_hash(self, text: str) -> str:
        """计算文本哈希"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    # ==================== 向量缓存 ====================

    def load_embeddings_cache(self) -> Tuple[Dict[str, int], Optional[np.ndarray]]:
        """加载向量缓存"""
        if self.embeddings_index_file.exists() and self.embeddings_file.exists():
            with open(self.embeddings_index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            embeddings = np.load(self.embeddings_file)['arr_0']
            return index, embeddings
        return {}, None

    def save_embeddings_cache(self, index: Dict[str, int], embeddings: np.ndarray):
        """保存向量缓存"""
        with open(self.embeddings_index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        np.savez_compressed(self.embeddings_file, embeddings)

    def get_cached_embedding(self, text: str, index: Dict[str, int]) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """获取单条文本的缓存向量"""
        text_hash = self._get_text_hash(text)
        if text_hash in index:
            return index[text_hash], None
        return None, None

    # ==================== BERT输出缓存 ====================

    def load_bert_outputs_cache(self) -> Dict[str, Any]:
        """加载BERT输出缓存"""
        if self.bert_outputs_file.exists():
            with open(self.bert_outputs_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_bert_outputs_cache(self, outputs: Dict[str, Any]):
        """保存BERT输出缓存"""
        # 限制缓存大小，只保留最近1000条
        if len(outputs) > 1000:
            # 保留最新的500条
            keys_to_keep = list(outputs.keys())[-500:]
            outputs = {k: outputs[k] for k in keys_to_keep}
        with open(self.bert_outputs_file, 'wb') as f:
            pickle.dump(outputs, f)

    # ==================== 聚类结果缓存 ====================

    def load_cluster_cache(self, cache_key: str) -> Optional[Dict]:
        """加载聚类结果缓存"""
        cache_file = self.cache_dir / f"cluster_{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def save_cluster_cache(self, cache_key: str, result: Dict):
        """保存聚类结果缓存"""
        cache_file = self.cache_dir / f"cluster_{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

    # ==================== 清理 ====================

    def clear_cache(self):
        """清空所有缓存"""
        for f in self.cache_dir.glob("*"):
            f.unlink()
        print("✅ 缓存已清空")

    def get_cache_size(self) -> int:
        """获取缓存大小（字节）"""
        total = 0
        for f in self.cache_dir.glob("*"):
            total += f.stat().st_size
        return total