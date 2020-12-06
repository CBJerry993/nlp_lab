# -*- coding: utf-8 -*-
# @Time    : 2020/6/29 上午10:09
# @File    : test_bd_lac.py

from LAC import LAC


def test_lac():
    """NER"""
    # 装载LAC模型
    lac = LAC(mode='lac')

    # 单个样本输入，输入为Unicode编码的字符串
    text = "LAC是个优秀的分词工具"
    lac_result = lac.run(text)

    # 批量样本输入, 输入为多个句子组成的list，平均速率更快
    texts = ["LAC是个优秀的分词工具", "百度是一家高科技公司"]
    lac_result = lac.run(texts)
    for i in range(10):
        print(lac_result)


def test_seg():
    """分词"""
    # 装载分词模型
    lac = LAC(mode='seg')

    # 单个样本输入，输入为Unicode编码的字符串
    text = "命名实体识别的准确度，决定了下游任务的效果，是NLP中非常重要的一个基础问题。"
    seg_result = lac.run(text)

    # 批量样本输入, 输入为多个句子组成的list，平均速率会更快
    texts = ["命名实体识别的准确度，决定了下游任务的效果，是NLP中非常重要的一个基础问题。", "百度是一家高科技公司"]
    seg_result = lac.run(texts)
    print(seg_result)


test_lac()
