from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import json


class RexUniNLUComplete:
    """RexUniNLU 全任务封装"""

    def __init__(self, model_path='damo/nlp_deberta_rex-uninlu_chinese-base'):
        self.pipeline = pipeline(
            task='rex-uninlu',
            model=model_path,
            model_revision='v1.2.1'
        )

    def extract(self, text: str, schema: dict):
        return self.pipeline(input=text, schema=schema)

    # ==================== 1. 命名实体识别 NER ====================
    def ner(self, text: str):
        """提取人物、地点、组织机构"""
        schema = {"人物": None, "地点": None, "组织机构": None}
        return self.extract(text, schema)

    # ==================== 2. 关系抽取 RE ====================
    def relation_extraction(self, text: str):
        """抽取实体间关系"""
        schema = {
            "创始人": [("人物", "组织机构")],
            "总部地点": [("组织机构", "地点")],
            "任职": [("人物", "组织机构")]
        }
        return self.extract(text, schema)

    # ==================== 3. 事件抽取 EE ====================
    def event_extraction(self, text: str):
        """抽取事件及其参与者"""
        schema = {
            "收购(事件触发词)": {
                "收购方": None,
                "被收购方": None,
                "收购金额": None,
                "收购时间": None
            }
        }
        return self.extract(text, schema)

    # ==================== 4. 属性级情感抽取 ABSA ====================
    def absa(self, text: str):
        """分析特定属性的情感"""
        schema = {
            "屏幕": ["正面", "负面", "中性"],
            "续航": ["正面", "负面", "中性"],
            "价格": ["正面", "负面", "中性"]
        }
        return self.extract(text, schema)

    # ==================== 5. 情感分类 ====================
    def sentiment(self, text: str):
        """整体情感判断"""
        schema = {"情感极性": ["正面", "负面", "中性"]}
        return self.extract(text, schema)

    # ==================== 6. 单标签分类 ====================
    def single_label_classify(self, text: str):
        """文章归入单一类别"""
        schema = {"类别": ["科技", "财经", "体育", "娱乐", "时政"]}
        return self.extract(text, schema)

    # ==================== 7. 多标签分类 ====================
    def multi_label_classify(self, text: str):
        """文章归入多个类别"""
        schema = {"标签": ["人工智能", "并购", "芯片", "自动驾驶", "创业"]}
        return self.extract(text, schema)

    # ==================== 8. 文本匹配 ====================
    def text_matching(self, text1: str, text2: str):
        """判断两段文本是否语义相似"""
        # 拼接两段文本，问是否相似
        combined = f"句子A：{text1}\n句子B：{text2}\n问题：这两句话意思相同吗？"
        schema = {"答案": ["是", "否"]}
        return self.extract(combined, schema)

    # ==================== 9. 自然语言推理 NLI ====================
    def nli(self, premise: str, hypothesis: str):
        """判断逻辑关系：蕴含/矛盾/中性"""
        combined = f"前提：{premise}\n假设：{hypothesis}"
        schema = {"关系": ["蕴含", "矛盾", "中性"]}
        return self.extract(combined, schema)

    # ==================== 10. 机器阅读理解 MRC ====================
    def mrc(self, context: str, question: str):
        """根据文章回答问题"""
        combined = f"文章：{context}\n问题：{question}"
        schema = {"答案": None}
        return self.extract(combined, schema)

    # ==================== 11. 指代消解 ====================
    def coref_resolution(self, text: str):
        """找出代词所指代的实体"""
        schema = {"代词": {"指代对象": None}}
        return self.extract(text, schema)


# ==================== 运行示例 ====================
if __name__ == "__main__":
    # 初始化
    model = RexUniNLUComplete()

    print("=" * 60)
    print("RexUniNLU 全任务演示")
    print("=" * 60)

    # ---------- 测试文本 ----------
    news = "2024年11月8日，中国人工智能芯片公司「算力无限」宣布成功收购美国初创公司NexaCore，交易金额25亿美元。算力无限CEO张伟表示，这将增强自动驾驶技术布局。"
    review = "这款手机屏幕显示很清晰，但续航太差了，价格偏贵。"

    # 1. NER
    print("\n【1. NER - 命名实体识别】")
    print(model.ner(news))

    # 2. 关系抽取
    print("\n【2. RE - 关系抽取】")
    print(model.relation_extraction("马云是阿里巴巴创始人，公司总部在杭州。"))

    # 3. 事件抽取
    print("\n【3. EE - 事件抽取】")
    print(model.event_extraction(news))

    # 4. ABSA
    print("\n【4. ABSA - 属性级情感】")
    print(model.absa(review))

    # 5. 情感分类
    print("\n【5. 情感分类】")
    print(model.sentiment("这部电影太好看了，强烈推荐！"))

    # 6. 单标签分类
    print("\n【6. 单标签分类】")
    print(model.single_label_classify(news))

    # 7. 多标签分类
    print("\n【7. 多标签分类】")
    print(model.multi_label_classify(news))

    # 8. 文本匹配
    print("\n【8. 文本匹配】")
    print(model.text_matching("苹果发布新款iPhone", "Apple推出全新智能手机"))

    # 9. NLI
    print("\n【9. 自然语言推理 NLI】")
    print(model.nli("所有猫都吃鱼", "我家猫不吃鱼"))

    # 10. MRC
    print("\n【10. 机器阅读理解 MRC】")
    print(model.mrc(news, "收购方是谁？"))

    # 11. 指代消解
    print("\n【11. 指代消解】")
    print(model.coref_resolution("张伟说他会亲自参与此次收购。"))