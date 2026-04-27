"""
文本预处理模块 - 清洗、分词、去停用词、语言检测、模板检测、文本过滤
"""

import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional
import numpy as np


class TextPreprocessor:
    """文本预处理器"""

    # 中文停用词（基础）
    CHINESE_STOPWORDS = {
        '的', '了', '和', '与', '或', '是', '在', '有', '被', '把', '给', '让',
        '这', '那', '也', '都', '还', '就', '只', '但', '却', '而', '并',
        '啊', '哦', '嗯', '吧', '呢', '吗', '啦', '哟',
        '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们', '它们',
        '不', '没', '没有', '不是', '不要', '不能', '不会',
        '很', '太', '非常', '特别', '十分', '相当', '比较',
        '可以', '能够', '可能', '应该', '需要', '想要',
        '表示', '认为', '指出', '称', '说', '告诉', '强调', '宣布', '透露',
        '此外', '另外', '同时', '不过', '然而', '而且', '并且',
        '因此', '所以', '从而', '因而', '于是',
        '主要', '重要', '具体', '相关', '进行', '开展', '实施', '推进',
        '目前', '当前', '现在', '如今', '日前', '近日', '近期',
        '对于', '关于', '针对', '基于', '通过', '根据', '按照',
        '提供', '获得', '实现', '达到', '完成', '取得', '形成',
    }

    # 英文停用词
    ENGLISH_STOPWORDS = {
        'a', 'an', 'and', 'the', 'of', 'to', 'in', 'for', 'on', 'with', 'by', 'at',
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
        'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
        'this', 'that', 'these', 'those', 'some', 'any', 'no', 'all', 'each', 'every',
        'not', 'so', 'too', 'very', 'just', 'but', 'however',
    }

    def __init__(self, language: str = "auto"):
        """
        初始化预处理器

        参数:
        - language: 语言 ('auto', 'zh', 'en')
        """
        self.language = language
        self._jieba_loaded = False
        self.start_templates = set()
        self.end_templates = set()

    def _load_jieba(self):
        """加载 jieba 分词器"""
        if not self._jieba_loaded:
            try:
                import jieba
                self.jieba = jieba
                self._jieba_loaded = True
            except ImportError:
                raise ImportError("jieba 未安装，请运行: pip install jieba")

    def detect_language(self, text: str) -> str:
        """检测文本语言"""
        if not text or not isinstance(text, str):
            return 'unknown'

        try:
            from langdetect import detect
            lang = detect(text)
            if lang.startswith('zh'):
                return 'zh'
            elif lang.startswith('en'):
                return 'en'
        except:
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            if chinese_chars / max(len(text), 1) > 0.3:
                return 'zh'
            elif re.search(r'[a-zA-Z]', text):
                return 'en'

        return 'unknown'

    def clean_text(self, text: str) -> str:
        """
        基础清洗文本
        """
        if not isinstance(text, str):
            text = str(text)

        # 去除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)

        # 去除 URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # 去除邮箱
        text = re.sub(r'\S+@\S+\.\S+', '', text)

        # 去除特殊字符（保留中文、英文、数字、基本标点）
        text = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbfa-zA-Z0-9\s\.\,\!\?\;\:\'\"\(\)\[\]\{\}\-\+\=\*\&\%\$\#\@\!\~\`]', ' ', text)

        # 去除多余空白
        text = re.sub(r'\s+', ' ', text)

        # 去除首尾空白
        text = text.strip()

        return text

    def split_sentences(self, text: str, aggressive: bool = True) -> List[str]:
        """
        分句（激进模式：按所有标点切分）

        参数:
        - text: 文本
        - aggressive: 是否激进模式

        返回: 句子列表
        """
        if not text:
            return []

        if aggressive:
            delimiters = r'([。！？!?；;，,、：:·\n]+)'
            parts = re.split(delimiters, text)
        else:
            delimiters = r'([。！？!?]+)'
            parts = re.split(delimiters, text)

        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sentence = parts[i]
            punctuation = parts[i + 1] if i + 1 < len(parts) else ''
            full_sentence = (sentence + punctuation).strip()
            if full_sentence:
                sentences.append(full_sentence)

        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())

        # 如果句子太少，尝试按空格切分
        if len(sentences) < 3 and ' ' in text:
            parts = [p.strip() for p in text.split(' ') if p.strip()]
            temp = []
            for p in parts:
                if len(p) > 10:
                    temp.append(p)
                elif temp:
                    temp[-1] += ' ' + p
                else:
                    temp.append(p)
            if len(temp) >= 3:
                sentences = temp

        return sentences

    def tokenize(self, text: str, language: str = "auto") -> List[str]:
        """分词"""
        if not text:
            return []

        if language == "auto":
            language = self.detect_language(text)

        if language == "zh":
            self._load_jieba()
            return list(self.jieba.cut(text))
        else:
            return text.lower().split()

    def remove_stopwords(self, tokens: List[str], language: str = "auto") -> List[str]:
        """去除停用词"""
        if not tokens:
            return []

        if language == "auto" and tokens:
            has_chinese = any(ord(c) >= 0x4e00 and ord(c) <= 0x9fff for token in tokens for c in token)
            language = "zh" if has_chinese else "en"

        stopwords = self.CHINESE_STOPWORDS if language == "zh" else self.ENGLISH_STOPWORDS
        return [t for t in tokens if t not in stopwords and len(t.strip()) > 1]

    def filter_noise(self, text: str, min_len: int = 20) -> str:
        """
        过滤文本中的噪音

        过滤规则：
        1. 去除纯数字/代码行
        2. 去除股票代码（6位数字，0/6开头）
        3. 去除过短行
        4. 去除纯标点行
        5. 去除连续重复字符

        参数:
        - text: 文本
        - min_len: 最小长度阈值

        返回: 过滤后的文本
        """
        if not text:
            return text

        lines = text.split('\n')
        filtered_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 1. 过滤纯数字/代码行
            if re.match(r'^[\d\s\-\.\,]+$', line):
                continue

            # 2. 过滤股票代码（6位数字，0/6开头）
            if re.match(r'^[06]\d{5}$', line):
                continue

            # 3. 过滤过短行
            if len(line) < min_len:
                continue

            # 4. 过滤纯标点行
            if re.match(r'^[。！？!?；;，,、：:·\s]+$', line):
                continue

            # 5. 过滤连续重复字符（如 "........"）
            if re.match(r'^(.)\1{5,}$', line):
                continue

            filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def detect_template_words(self, texts: List[str],
                               min_abs: int = 10,
                               min_ratio: float = 0.05,
                               start_thresholds: Dict[int, float] = None,
                               end_thresholds: Dict[int, float] = None,
                               verbose: bool = True) -> Tuple[Set[str], Set[str]]:
        """
        基于位置分布检测模板词

        参数:
        - texts: 所有文本列表（已清洗）
        - min_abs: 绝对次数阈值
        - min_ratio: 占比阈值
        - start_thresholds: 开头位置阈值 {1: 0.3, 2: 0.2, 3: 0.1}
        - end_thresholds: 结尾位置阈值 {-1: 0.3, -2: 0.2, -3: 0.1}
        - verbose: 是否打印调试信息

        返回: (开头模板词集合, 结尾模板词集合)
        """
        if start_thresholds is None:
            start_thresholds = {1: 0.3, 2: 0.2, 3: 0.1}
        if end_thresholds is None:
            end_thresholds = {-1: 0.3, -2: 0.2, -3: 0.1}

        if not texts:
            return set(), set()

        total_articles = len(texts)
        word_positions = defaultdict(list)
        article_word_count = defaultdict(set)

        for idx, text in enumerate(texts):
            if not text or len(text) < 50:
                continue

            sentences = self.split_sentences(text, aggressive=True)
            if len(sentences) < 3:
                continue

            for i, sent in enumerate(sentences):
                words = self.tokenize(sent)
                words = [w for w in words if len(w) > 1]

                pos_forward = i + 1
                pos_backward = i - len(sentences)

                for w in words:
                    word_positions[w].append(pos_forward)
                    word_positions[w].append(pos_backward)
                    article_word_count[w].add(idx)

        if verbose:
            print(f"  统计: {len(word_positions)} 个词, {total_articles} 篇文章")

        if not word_positions:
            return set(), set()

        start_templates = set()
        end_templates = set()

        for word, positions in word_positions.items():
            total = len(positions)
            article_count = len(article_word_count[word])

            if article_count < min_abs and article_count / total_articles < min_ratio:
                continue

            pos_counts = {}
            for p in set(positions):
                pos_counts[p] = positions.count(p) / total

            is_start = False
            for pos, threshold in start_thresholds.items():
                if pos_counts.get(pos, 0) > threshold:
                    is_start = True
                    break
            if is_start:
                start_templates.add(word)

            is_end = False
            for pos, threshold in end_thresholds.items():
                if pos_counts.get(pos, 0) > threshold:
                    is_end = True
                    break
            if is_end:
                end_templates.add(word)

        self.start_templates = start_templates
        self.end_templates = end_templates

        if verbose:
            print(f"  开头模板词: {len(start_templates)} 个")
            print(f"  结尾模板词: {len(end_templates)} 个")
            if start_templates:
                print(f"  示例: {list(start_templates)[:10]}")
            if end_templates:
                print(f"  示例: {list(end_templates)[:10]}")

        return start_templates, end_templates

    def find_content_boundary(self, sentences: List[str],
                               start_templates: Set[str],
                               end_templates: Set[str],
                               min_consecutive: int = 2) -> Tuple[int, int]:
        """
        查找正文边界（开始索引和结束索引）
        """
        n = len(sentences)
        if n == 0:
            return 0, 0

        start_idx = 0
        for i in range(n):
            words = self.tokenize(sentences[i])
            if any(w in start_templates for w in words):
                continue

            is_content_start = True
            for j in range(1, min_consecutive + 1):
                if i + j >= n:
                    break
                words_next = self.tokenize(sentences[i + j])
                if any(w in start_templates for w in words_next):
                    is_content_start = False
                    break

            if is_content_start:
                start_idx = i
                break

        end_idx = n
        for i in range(n - 1, -1, -1):
            words = self.tokenize(sentences[i])
            if any(w in end_templates for w in words):
                continue

            is_content_end = True
            for j in range(1, min_consecutive + 1):
                if i - j < 0:
                    break
                words_prev = self.tokenize(sentences[i - j])
                if any(w in end_templates for w in words_prev):
                    is_content_end = False
                    break

            if is_content_end:
                end_idx = i + 1
                break

        min_keep = max(1, int(n * 0.3))
        if end_idx - start_idx < min_keep:
            return 0, n

        return start_idx, end_idx

    def remove_templates(self, text: str,
                         start_templates: Set[str],
                         end_templates: Set[str],
                         min_consecutive: int = 2) -> str:
        """切除文本中的模板"""
        if not text or (not start_templates and not end_templates):
            return text

        sentences = self.split_sentences(text, aggressive=True)
        if len(sentences) <= 3:
            return text

        start_idx, end_idx = self.find_content_boundary(
            sentences, start_templates, end_templates, min_consecutive
        )

        if start_idx == 0 and end_idx == len(sentences):
            return text

        return ''.join(sentences[start_idx:end_idx])

    def full_process(self, text: str,
                     remove_templates: bool = True,
                     filter_noise: bool = True,
                     filter_tokens: bool = True,
                     min_len: int = 20) -> Dict:
        """
        完整预处理流程（一键调用）

        参数:
        - text: 原始文本
        - remove_templates: 是否切除模板
        - filter_noise: 是否过滤噪音（行级别）
        - filter_tokens: 是否过滤无意义词（词级别）
        - min_len: 最小长度阈值

        返回: 各阶段结果
        """
        result = {
            "raw": text,
            "cleaned": None,
            "content": None,
            "filtered": None,
            "tokens": None,
            "tokens_cleaned": None,
            "tokens_filtered": None  # 新增：词级别过滤后的 tokens
        }

        cleaned = self.clean_text(text)
        result["cleaned"] = cleaned

        if remove_templates and (self.start_templates or self.end_templates):
            content = self.remove_templates(cleaned, self.start_templates, self.end_templates)
        else:
            content = cleaned
        result["content"] = content

        if filter_noise:
            filtered = self.filter_noise(content, min_len)
        else:
            filtered = content
        result["filtered"] = filtered

        language = self.detect_language(filtered)
        tokens = self.tokenize(filtered, language)
        tokens_cleaned = self.remove_stopwords(tokens, language)
        result["tokens"] = tokens
        result["tokens_cleaned"] = tokens_cleaned

        # 词级别过滤（用于聚类、主题建模）
        if filter_tokens:
            result["tokens_filtered"] = self.filter_tokens(tokens_cleaned)
        else:
            result["tokens_filtered"] = tokens_cleaned

        return result

    def process_batch(self, texts: List[str],
                      remove_templates: bool = True,
                      filter_noise: bool = True,
                      min_len: int = 20) -> List[Dict]:
        """批量完整预处理"""
        results = []
        for text in texts:
            results.append(self.full_process(text, remove_templates, filter_noise, min_len))
        return results

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        过滤无意义的词（用于聚类、主题建模前的词级别清洗）

        过滤规则：
        1. 纯数字
        2. 股票代码（6位数字，0/6开头）
        3. 纯日期格式（如 0424, 20260424）
        4. 单字词
        5. 数字+单位组合（如 38万亿元）
        6. 纯英文短词（<=2字母）
        """
        if not tokens:
            return []

        filtered = []
        for t in tokens:
            # 1. 过滤纯数字
            if re.match(r'^\d+$', t):
                continue

            # 2. 过滤股票代码（6位数字，0/6开头）
            if re.match(r'^[06]\d{5}$', t):
                continue

            # 3. 过滤纯日期格式（4-8位数字）
            if re.match(r'^\d{4,8}$', t):
                continue

            # 4. 过滤单字词
            if len(t) <= 1:
                continue

            # 5. 过滤数字+单位组合
            if re.match(r'^[\d\.]+[万亿千百十]?[元%倍]?$', t):
                continue
            if re.match(r'^[\d\.]+[年月日时分秒]$', t):
                continue

            # 6. 过滤纯英文短词（<=2字母）
            if re.match(r'^[a-zA-Z]+$', t) and len(t) <= 2:
                continue

            # 7. 过滤纯标点符号
            if re.match(r'^[^\u4e00-\u9fff\u3400-\u4dbfa-zA-Z]+$', t):
                continue

            filtered.append(t)

        return filtered