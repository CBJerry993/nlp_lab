# -*- coding: utf-8 -*-
# @Time    : 2020/6/25 下午6:59
# @File    : gensim_test.py

import jieba.analyse
import codecs


def segment():
    """切分文件，获得单词"""
    # 打开文件
    f = codecs.open('wik_00/AA/zhwiki_jian_zh.txt', 'r', encoding="utf-8")
    # 写入文件
    target = codecs.open("wiki_jian_zh_seg.txt", 'w', encoding="utf-8")

    print('open files')
    line_num = 1
    line = f.readline()

    while line:
        print('---- processing ', line_num, ' article----------------')
        line_seg = " ".join(jieba.cut(line))
        target.writelines(line_seg)
        line_num = line_num + 1
        line = f.readline()  # 若无，返回 -1

    # 关闭两个文件流，并退出程序
    f.close(), target.close()


def test():
    """测试训练好的模型"""
    from gensim.models import Word2Vec
    en_wiki_word2vec_model = Word2Vec.load('wiki_zh_jian_text.model')
    # testwords = ['金融', '上', '股票', '跌', '经济']
    testwords = ['工作', '喜欢', '问题', '宝贝']
    for i in range(len(testwords)):
        res = en_wiki_word2vec_model.most_similar(testwords[i])
        print(testwords[i])
        print(res)


test()
