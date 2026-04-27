"""
实体识别模块 - 人名、地名、组织名
"""

import re
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter


class EntityRecognizer:
    """实体识别器 - 基于规则和词典"""

    # 中文姓氏（常见）
    CHINESE_SURNAMES = {
        '王', '李', '张', '刘', '陈', '杨', '黄', '赵', '吴', '周', '徐', '孙', '马', '朱',
        '林', '郭', '何', '高', '郑', '罗', '梁', '谢', '宋', '唐', '许', '邓', '肖', '冯',
        '韩', '曹', '彭', '曾', '肖', '田', '董', '潘', '袁', '于', '蒋', '蔡', '余', '杜',
        '苏', '吕', '丁', '沈', '任', '姚', '卢', '傅', '钟', '姜', '崔', '谭', '廖', '范'
    }

    # 常见地名后缀
    LOCATION_SUFFIXES = {
        '省', '市', '区', '县', '镇', '乡', '村', '街', '路', '道', '号', '大厦',
        '广场', '中心', '酒店', '饭店', '餐厅', '商场', '超市', '公园', '景区',
        '北京', '上海', '广州', '深圳', '杭州', '南京', '武汉', '成都', '重庆'
    }

    # 组织名称后缀
    ORG_SUFFIXES = {
        '公司', '集团', '有限', '股份', '科技', '网络', '技术', '软件', '数据',
        '银行', '保险', '证券', '基金', '大学', '学院', '医院', '研究院',
        '政府', '局', '部', '委', '办', '处', '科', '中心', '协会', '学会',
        '商会', '联盟', '组织', '机构', '平台', '媒体', '报社', '出版社'
    }

    def __init__(self, language: str = "auto"):
        """
        初始化实体识别器

        参数:
        - language: 语言 ('auto', 'zh', 'en')
        """
        self.language = language

    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        if self.language != "auto":
            return self.language

        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        if chinese_chars / max(len(text), 1) > 0.2:
            return "zh"
        return "en"

    def _extract_chinese_entities(self, text: str) -> Dict[str, List[str]]:
        """提取中文实体"""
        entities = {
            "person": [],
            "location": [],
            "organization": []
        }

        # 人名识别：姓氏 + 1-2个汉字
        import re
        # 匹配 姓氏 + 两个字
        person_pattern = f'[{"".join(self.CHINESE_SURNAMES)}][\u4e00-\u9fff]{{1,2}}(?=[，。！？；：\s]|$)'
        persons = set(re.findall(person_pattern, text))
        entities["person"] = list(persons)[:20]

        # 地名识别：匹配后缀模式
        location_pattern = f'[\u4e00-\u9fff]+?(?:{"|".join(self.LOCATION_SUFFIXES)})'
        locations = set(re.findall(location_pattern, text))
        entities["location"] = list(locations)[:20]

        # 组织名识别
        org_pattern = f'[\u4e00-\u9fff]+?(?:{"|".join(self.ORG_SUFFIXES)})'
        orgs = set(re.findall(org_pattern, text))
        entities["organization"] = list(orgs)[:20]

        return entities

    def _extract_english_entities(self, text: str) -> Dict[str, List[str]]:
        """提取英文实体（简单实现）"""
        entities = {
            "person": [],
            "location": [],
            "organization": []
        }

        # 大写单词可能是实体
        words = text.split()
        for w in words:
            # 清除标点
            clean = re.sub(r'[^\w\']', '', w)
            if clean and clean[0].isupper() and len(clean) > 1:
                if clean not in ['I', 'The', 'And', 'Of', 'To', 'In', 'For', 'With']:
                    entities["person"].append(clean)

        # 去重
        entities["person"] = list(set(entities["person"]))[:20]

        return entities

    def recognize(self, text: str) -> Dict[str, List[str]]:
        """
        识别文本中的实体

        返回:
        {
            "person": List[str],      # 人名
            "location": List[str],    # 地名
            "organization": List[str] # 组织名
        }
        """
        if not text or not isinstance(text, str):
            return {"person": [], "location": [], "organization": []}

        language = self._detect_language(text)

        if language == "zh":
            return self._extract_chinese_entities(text)
        else:
            return self._extract_english_entities(text)

    def recognize_batch(self, texts: List[str]) -> List[Dict]:
        """批量识别实体"""
        return [self.recognize(t) for t in texts]

    def get_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """
        获取实体统计

        参数:
        - results: recognize_batch 的结果列表

        返回:
        {
            "person": {"total": int, "unique": int, "top": List[Tuple[str, int]]},
            "location": {...},
            "organization": {...}
        }
        """
        stats = {}

        for entity_type in ["person", "location", "organization"]:
            all_entities = []
            for r in results:
                all_entities.extend(r.get(entity_type, []))

            counter = Counter(all_entities)
            stats[entity_type] = {
                "total": len(all_entities),
                "unique": len(counter),
                "top": counter.most_common(10)
            }

        return stats