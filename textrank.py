# -*- coding: utf-8 -*-
""" ------------------------------------------------- 
File Name: textrank
Description : 
Author : arlen
date：18-6-20
------------------------------------------------- """
import numpy as np
import jieba
import itertools


class TextRank(object):
    def __init__(self):
        """
        textrank文本摘要提取
        """
        try:
            self.word_vec = self.load_word2vec()
        except:
            raise

    def sentence_vector(self, sen):
        """
        构建句子向量，采用词向量求平均的方式
        :param sen:
        :return:
        """
        zero = np.zeros(300)  # word2vec 300 dim

        vec = np.array([self.word_vec.get(word, zero) for word in sen])
        vec = np.sum(vec, axis=0)/len(sen)

        return vec

    def create_graph(self, doc_sents):
        """
        构建句子相似矩阵
        :param doc_sents:
        :return:
        """
        num = len(doc_sents)
        board = np.zeros([num, num])
        doc_vec = [self.sentence_vector(i) for i in doc_sents]

        for i, j in itertools.product(range(num), repeat=2):
            if i > j:
                board[i][j] = self.calculate_similarity(doc_vec[i], doc_vec[j])

        board += board.T
        return board

    def weighted_pagerank(self, weight_graph):
        """
        计算各个句子分数
        :param weight_graph:
        :return:
        """
        scores = np.zeros(len(weight_graph)) + 0.5
        old_scores = np.zeros(len(weight_graph))

        n = 0  # 迭代次数, 限定最大迭代200次

        while self.different(scores, old_scores) or n > 200:
            old_scores[:][:] = scores[:][:]

            for i in range(len(weight_graph)):
                scores[i] = self.calculate_socre(weight_graph, scores, i)

            n += 1
        return scores

    def summary(self, doc_sents, n):
        """
        输入句子列表, 返回TOP N 句子索引
        :param doc_sents:
        :param n:
        :return:
        """
        similarity_grap = self.create_graph(doc_sents)
        similarity_grap = np.nan_to_num(similarity_grap)
        scores = self.weighted_pagerank(similarity_grap)
        top = np.argpartition(scores, -n)[-n:]
        top.sort()
        return top

    @staticmethod
    def different(scores, old_scores):
        """
        判断分数前后有没有变化, 小于0.0001则认为变化很小, 趋于稳定
        :param scores:
        :param old_scores:
        :return:
        """
        flag = False
        if np.max(np.fabs(scores - old_scores)) >= 0.001:
            flag = True
        return flag

    @staticmethod
    def calculate_socre(weight_graph, scores, i):
        """
        计算指定句子分数
        :param weight_graph:
        :param scores:
        :param i:
        :return:
        """
        d = 0.85

        fraction = weight_graph[:][i] * scores[:]
        denominator = np.sum(weight_graph, axis=0)
        denominator = np.nan_to_num(denominator) + 0.0001
        added_score = np.sum(fraction / denominator)

        weighted_score = (1 - d) + d * added_score

        return weighted_score

    @staticmethod
    def calculate_similarity(sen1, sen2):
        """
        计算两向量余弦相似度
        :param sen1:
        :param sen2:
        :return:
        """
        fraction = np.dot(sen1, sen2)
        denominator = (np.linalg.norm(sen1) * (np.linalg.norm(sen2))) + 0.0001
        return fraction / denominator

    @staticmethod
    def load_word2vec():
        """
        读取预训练word2vec词向量
        :return:
        """
        with open('word2vec/sgns.sogou.bigram', 'r', encoding='utf-8') as f:
            f.readline()
            word_vec = dict()
            for line in f:
                line_ = line.split(' ')
                key_ = line_[0]
                val_ = np.array(list(map(float, line_[1: -1])))
                word_vec[key_] = val_
        return word_vec


if __name__ == '__main__':
    text = '''俄罗斯申请加入联合国人权理事会。
海外网6月20日电 就在美国宣布退出联合国人权理事会不久后，俄罗斯常驻联合国代表团于当地时间周三（20日）表示，俄罗斯已经申请成为联合国人权理事会2021-2023届成员国。
据俄罗斯卫星通讯社报道，俄罗斯常驻联合国代表团第一秘书费多尔·斯特日诺夫斯基（Fedor Strzhizhovskiy）表示，俄罗斯想要继续在人权理事会开展有效工作，在人权领域保持平等对话与合作。为此，俄罗斯提议成为联合国人权理事会2021-2023届成员。
据悉，第71届联合国大会于2016年10月28日在纽约联合国总部改选人权理事会14个成员，中国等14个国家28日当选联合国人权理事会成员，任期从2017年至2019年。而俄罗斯当时获得112票，仅以2票之差败于克罗地亚，未能连任联合国人权理事会成员。
此前不久，美国于19日刚宣布退出联合国人权理事会。这是华盛顿退出巴黎气候协议和伊朗核协议之后，美国又一次拒绝国际多边机制。社会活动家们警告，此举将使推动全球人权状况的进程变得更加困难。
对于美国这次“退群”的原因，有媒体分析称，美国驻联合国大使黑莉去年曾表示，如果联合国人权理事会不解除“长期以来对以色列的不公正待遇”，华盛顿方面就将退出该组织。
据了解，在小布什当政时期，美国曾以人权理事会充满以色列的敌人为由，对其抵制长达三年，直到2009年奥巴马当政才重回该组织。
黑莉在美国宣布“退群”后表示，美国将继续主张联合国人权理事会进行改革，在此之后，该国可能重返该机构。黑莉保证称，退出人权理事会后，美国将在该机构框架外继续致力于保护人权。
联合国人权理事会现有47个席位，上周该组织曾发动内部投票，针对加沙地区的杀戮展开调查，并指控以色列过度使用武力。美国和澳大利亚是仅有的投反对票的两个国家。随后以色列驻日内瓦的联合国大使阿维娃·拉兹·谢克特（Aviva Raz Shechter）表示，理事会的行为是在“散播针对以色列的谎言”。（海外网 姚凯红）'''
    import time
    textrank = TextRank()
    text = text.split('。')
    text = [i for i in text if len(i) > 10]
    doc_sents = [jieba.lcut(i) for i in text]
    op = time.time()
    for i in range(1000):
        top = textrank.summary(doc_sents, 6)
        doc_summary = [text[i] for i in top]
        summary = '。'.join(doc_summary)
    print(time.time() - op)
    print(summary)