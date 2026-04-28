"""
实体识别模块 - 使用预训练的 NER 模型（优先本地模型）
"""

import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import re
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# 设置国内镜像源（用于下载）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class EntityRecognizer:
    """实体识别器 - 基于预训练 NER 模型"""

    # 模型名称
    MODEL_NAME = "shibing624/bert4ner-base-chinese"

    # 本地模型路径（按优先级查找）
    LOCAL_MODEL_PATHS = [
        Path("./models/bert4ner"),                    # 当前目录下的 models/bert4ner
        Path("../models/bert4ner"),                   # 上级目录
        Path.home() / ".cache" / "bert4ner",          # 用户缓存目录
        Path.home() / ".autotext" / "models" / "bert4ner",  # 应用目录
    ]

    def __init__(self, model_name: str = None, device: str = "cpu", offline: bool = False):
        """
        初始化实体识别器

        参数:
        - model_name: 模型名称或路径（可选）
        - device: 设备 ('cpu' 或 'cuda')
        - offline: 是否离线模式
        """
        self.device = device
        self.offline = offline

        # 优先使用本地模型路径
        self.model_path = self._find_local_model()

        if self.model_path:
            print(f"  🔄 加载本地NER模型: {self.model_path}")
        else:
            self.model_path = model_name or self.MODEL_NAME
            print(f"  🔄 加载在线NER模型: {self.model_path}")
            if not offline:
                print(f"  📦 首次使用会下载模型（约420MB），请耐心等待...")

        # 设置离线模式
        if offline and not self._is_local_model():
            raise FileNotFoundError(f"离线模式下找不到本地模型: {self.LOCAL_MODEL_PATHS}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=offline and self._is_local_model()
            )
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=offline and self._is_local_model()
            )
            self.model.to(device)
            self.model.eval()

            # 获取标签映射
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id

            # 过滤需要的实体类型
            self.keep_types = {'PER', 'ORG', 'LOC', 'COMPANY', 'PRODUCT', 'TIME', 'NUMBER'}

            print(f"  ✅ NER模型加载完成")
            print(f"     支持实体类型: {len(self.id2label)} 种")
            print(f"     实体类型示例: {list(self.id2label.values())[:10]}")

        except Exception as e:
            print(f"  ❌ 模型加载失败: {e}")
            raise

    def _find_local_model(self) -> Optional[Path]:
        """查找本地模型"""
        for path in self.LOCAL_MODEL_PATHS:
            if path.exists() and (path / "config.json").exists():
                # 检查是否有模型权重文件
                if (path / "pytorch_model.bin").exists() or \
                   (path / "model.safetensors").exists():
                    return path
        return None

    def _is_local_model(self) -> bool:
        """检查当前使用的是否是本地模型"""
        return isinstance(self.model_path, Path)

    def _predict_batch(self, texts: List[str]) -> List[List[Tuple[str, int, int, str]]]:
        """批量预测实体"""
        all_entities = []

        for text in texts:
            if not text or len(text) < 10:
                all_entities.append([])
                continue

            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                    max_length=512, return_offsets_mapping=True)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            offset_mapping = inputs['offset_mapping'][0].numpy()

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()

            # 解析实体（BIO 格式）
            entities = self._parse_entities(predictions, offset_mapping, text)
            all_entities.append(entities)

        return all_entities

    def _parse_entities(self, predictions: np.ndarray, offset_mapping: np.ndarray,
                        text: str) -> List[Tuple[str, int, int, str]]:
        """解析 BIO 标签序列为实体列表"""
        entities = []
        current_entity = None
        current_start = None
        current_type = None

        for i, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            if start == 0 and end == 0:
                continue

            label = self.id2label.get(pred, 'O')

            if label.startswith('B-'):
                # 开始新实体
                if current_entity:
                    entities.append((current_entity, current_start, end, current_type))
                entity_type = label[2:]
                current_entity = text[start:end]
                current_start = start
                current_type = entity_type

            elif label.startswith('I-') and current_entity is not None:
                # 继续当前实体
                entity_type = label[2:]
                if current_type == entity_type:
                    current_entity = text[current_start:end]
                else:
                    entities.append((current_entity, current_start, end, current_type))
                    current_entity = None
                    current_start = None
                    current_type = None

            else:
                # O 标签，结束当前实体
                if current_entity:
                    entities.append((current_entity, current_start, end, current_type))
                    current_entity = None
                    current_start = None
                    current_type = None

        if current_entity:
            entities.append((current_entity, current_start, current_start + len(current_entity), current_type))

        return entities

    def _filter_entities(self, entities: List[Tuple[str, int, int, str]]) -> List[Tuple[str, int, int, str]]:
        """过滤无效实体"""
        # 停用词表
        stopwords = {'的', '了', '是', '在', '和', '与', '或', '也', '都', '还',
                     '这', '那', '有', '为', '对', '而', '并', '且', '但', '就',
                     '到', '从', '由', '于', '之', '将', '会', '能', '可', '以',
                     '年', '月', '日', '时', '分', '秒', '上', '下', '中', '内',
                     '外', '前', '后', '左', '右', '高', '低', '大', '小', '多',
                     '少', '新', '旧', '好', '坏', '正', '负', '涨', '跌', '不',
                     '没', '无', '非', '莫', '勿', '别', '未', '过', '很', '太',
                     '同比', '环比', '增长', '下降', '上升', '回落', '稳定'}

        filtered = []
        for entity, start, end, entity_type in entities:
            # 只保留需要的实体类型
            if entity_type not in self.keep_types:
                continue

            # 长度过滤
            if len(entity) < 2:
                continue

            # 停用词过滤
            if entity in stopwords:
                continue

            # 纯数字过滤
            if entity.isdigit():
                continue

            # 股票代码/期货代码过滤（如 A2607C5400）
            if re.match(r'^[A-Z0-9]{6,}$', entity):
                continue
            if re.match(r'^[A-Z]{2,}[0-9]{4,}[A-Z]?$', entity):
                continue
            if re.match(r'^[A-Z]{1,2}[0-9]{4}$', entity):
                continue

            # 单字英文过滤
            if re.match(r'^[A-Za-z]$', entity):
                continue

            # 过滤包含特殊符号的短实体
            if re.search(r'[^\u4e00-\u9fffA-Za-z0-9]', entity) and len(entity) < 4:
                continue

            filtered.append((entity, start, end, entity_type))

        return filtered

    def recognize(self, texts: List[str]) -> List[Dict[str, List[Tuple[str, int, int]]]]:
        """识别实体

        返回: [{
            'PER': [('张三', 0, 2), ...],
            'ORG': [('腾讯', 3, 5), ...],
            'LOC': [('北京', 6, 8), ...],
            ...
        }, ...]
        """
        batch_results = self._predict_batch(texts)

        results = []
        for entities in batch_results:
            filtered = self._filter_entities(entities)
            result = {}
            for entity_text, start, end, entity_type in filtered:
                if entity_type not in result:
                    result[entity_type] = []
                result[entity_type].append((entity_text, start, end))
            results.append(result)

        return results

    def get_entity_stats(self, results: List[Dict[str, List[Tuple[str, int, int]]]]) -> Dict[str, Any]:
        """获取实体统计（已过滤）"""
        stats = {}
        all_entities = {}

        for result in results:
            for entity_type, entities in result.items():
                if entity_type not in all_entities:
                    all_entities[entity_type] = []
                for entity_text, _, _ in entities:
                    all_entities[entity_type].append(entity_text)

        for entity_type, entities in all_entities.items():
            counter = Counter(entities)
            # 过滤低频实体（出现次数 < 2）
            top_items = [(name, count) for name, count in counter.most_common(30)
                        if count >= 2 and len(name) >= 2]
            stats[entity_type.lower()] = {
                'total': len(entities),
                'unique': len(counter),
                'top': top_items[:20]
            }

        return stats

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_path': str(self.model_path),
            'is_local': self._is_local_model(),
            'num_labels': len(self.id2label),
            'labels': list(self.id2label.values())
        }