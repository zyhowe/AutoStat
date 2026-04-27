"""
文本分析单元测试
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotext.core.preprocessor import TextPreprocessor
from autotext.core.sentiment import SentimentAnalyzer
from autotext.core.entity import EntityRecognizer
from autotext.core.keyword import KeywordExtractor


class TestTextPreprocessor(unittest.TestCase):
    """测试文本预处理"""

    def setUp(self):
        self.preprocessor = TextPreprocessor()

    def test_clean_text(self):
        text = "<html>测试文本</html>"
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, "测试文本")

    def test_split_sentences_zh(self):
        text = "你好。世界！今天天气不错。"
        sentences = self.preprocessor.split_sentences(text, "zh")
        self.assertEqual(len(sentences), 3)

    def test_tokenize_zh(self):
        text = "我爱北京天安门"
        tokens = self.preprocessor.tokenize(text, "zh")
        self.assertGreater(len(tokens), 0)

    def test_remove_stopwords(self):
        tokens = ["我", "爱", "北京", "的", "天安门"]
        cleaned = self.preprocessor.remove_stopwords(tokens, "zh")
        self.assertNotIn("我", cleaned)
        self.assertNotIn("的", cleaned)


class TestSentimentAnalyzer(unittest.TestCase):
    """测试情感分析"""

    def setUp(self):
        self.analyzer = SentimentAnalyzer()

    def test_analyze_positive(self):
        result = self.analyzer.analyze("这个产品很好，非常满意")
        self.assertEqual(result["sentiment"], "positive")

    def test_analyze_negative(self):
        result = self.analyzer.analyze("太差了，垃圾产品")
        self.assertEqual(result["sentiment"], "negative")

    def test_analyze_neutral(self):
        result = self.analyzer.analyze("今天天气不错")
        # 中性或积极都有可能
        self.assertIn(result["sentiment"], ["positive", "neutral"])

    def test_empty_text(self):
        result = self.analyzer.analyze("")
        self.assertEqual(result["sentiment"], "neutral")


class TestEntityRecognizer(unittest.TestCase):
    """测试实体识别"""

    def setUp(self):
        self.recognizer = EntityRecognizer()

    def test_recognize_chinese(self):
        text = "张三在北京市工作，就职于腾讯公司"
        entities = self.recognizer.recognize(text)
        self.assertGreater(len(entities.get("person", [])), 0)
        self.assertGreater(len(entities.get("location", [])), 0)
        self.assertGreater(len(entities.get("organization", [])), 0)


class TestKeywordExtractor(unittest.TestCase):
    """测试关键词提取"""

    def test_extract_frequency(self):
        texts = ["苹果苹果苹果", "香蕉香蕉", "橙子"]
        tokens_list = [list(text) for text in texts]
        extractor = KeywordExtractor()
        keywords = extractor.extract_frequency(tokens_list, top_n=5)
        self.assertGreater(len(keywords), 0)
        self.assertEqual(keywords[0][0], "苹果")


if __name__ == "__main__":
    unittest.main()