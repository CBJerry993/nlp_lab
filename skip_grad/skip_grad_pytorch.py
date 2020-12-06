# -*- coding: utf-8 -*-
# @Time    : 2020/6/29 下午8:52
# @File    : skip_grad_pytorch.py

"""
使用pytorch写一个skip_grad  参考知乎 https://zhuanlan.zhihu.com/p/82683575
"""

import torch
import numpy as np
from torch import nn, optim
import random
from collections import Counter
import matplotlib.pyplot as plt

# 训练数据
text = "I like dog i like cat i like animal dog cat animal apple cat dog like dog fish milk like dog \
cat eyes like i like apple apple i hate apple i movie book music like cat dog hate cat dog like"

# 参数设置
EMBEDDING_DIM = 2  # 词向量维度
PRINT_EVERY = 1000  # 可视化频率
EPOCHS = 1000  # 训练的轮数
BATCH_SIZE = 5  # 每一批训练数据大小
N_SAMPLES = 3  # 负样本大小
WINDOW_SIZE = 5  # 周边词窗口大小
FREQ = 0  # 词汇出现频率
DELETE_WORDS = False  # 是否删除部分高频词


# 文本预处理
def preprocess(text, FREQ):
    text = text.lower()
    text = text.replace('"', "").replace('.', '').replace(',', '').replace('!', '').replace('-', '').replace('/', '')
    words = text.split()
    # 去除低频词
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > FREQ]
    return trimmed_words


words = preprocess(text, FREQ)

# 构建词典
vocab = set(words)
vocab2int = {w: c for c, w in enumerate(vocab)}
int2vocab = {c: w for c, w in enumerate(vocab)}

# 将文本转化为数值
int_words = [vocab2int[w] for w in words]

# 计算单词频次
int_word_counts = Counter(int_words)
total_count = len(int_words)
word_freqs = {w: c / total_count for w, c in int_word_counts.items()}

# 去除出现频次高的词汇
if DELETE_WORDS:
    t = 1e-5
    prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in int_word_counts}
    train_words = [w for w in int_words if random.random() < (1 - prob_drop[w])]  # 随机取句子
else:
    train_words = int_words
print("train_words", train_words)

# 单词分布
word_freqs = np.array(list(word_freqs.values()))
unigram_dist = word_freqs / word_freqs.sum()
noise_dist = torch.from_numpy(unigram_dist ** 0.75 / np.sum(unigram_dist ** 0.75))  # 负采样


# 获取目标词汇
def get_target(words, idx, WINDOW_SIZE):
    target_window = np.random.randint(1, WINDOW_SIZE + 1)  # 随机取了窗口大小 1~WINDOW_SIZE
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    targets = set(words[start_point:idx] + words[idx + 1:end_point + 1])  # 不包含idx位置的词，skip_gram
    return list(targets)  # 返回不重复的目标词汇


# 批次化数据
def get_batch(words, BATCH_SIZE, WINDOW_SIZE):
    # words:传入的train_words列表，每个元素是int
    n_batches = len(words) // BATCH_SIZE  # 训练的batches数
    words = words[:n_batches * BATCH_SIZE]  # 分n_batches批
    for idx in range(0, len(words), BATCH_SIZE):
        batch_x, batch_y = [], []
        # 如果传入的是train_words则是用Int表示。后面的备注同！
        batch = words[idx:idx + BATCH_SIZE]  # ['i', 'like', 'dog', 'i', 'like']
        for i in range(len(batch)):
            x = batch[i]
            y = get_target(batch, i, WINDOW_SIZE)
            batch_x.extend([x] * len(y))
            batch_y.extend(y)  # 每个 batch_x, batch_y (['i', 'i'], ['like', 'dog'])
        """
        # 每次的batch_x和y的长度不一样，取决于get_target获得的y长度。
        # batch_x:['i', 'i', 'like', 'like', 'dog', 'dog', 'i', 'i', 'like']
        # batch_y:['like', 'dog', 'i', 'dog', 'like', 'i', 'like', 'dog', 'i']
        # batch_x:[10, 10, 10, 9, 9, 9, 5, 5, 10, 10, 10, 9, 9]
        # batch_y:[9, 10, 5, 9, 10, 5, 9, 10, 9, 10, 5, 10, 5]
        """
        yield batch_x, batch_y


# 定义模型
class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist
        # 定义词向量层
        self.in_embed = nn.Embedding(n_vocab, n_embed)  # 13,2
        self.out_embed = nn.Embedding(n_vocab, n_embed)  # 13,2
        # 词向量层参数初始化（把上诉向量随机初始化-1~1之间）
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    # 输入词的前向过程
    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors

    # 目标词的前向过程
    def forward_output(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors

    # 负样本词的前向过程
    def forward_noise(self, size, N_SAMPLES):
        noise_dist = self.noise_dist  # [13]
        # 从词汇分布中采样负样本,multinomial根据noise_dist的大小去采样取索引，取size * N_SAMPLES个。返回的是索引int
        noise_words = torch.multinomial(noise_dist, size * N_SAMPLES, replacement=True)  # [33] replace=T,有放回,不会取到为0的下标.
        noise_vectors = self.out_embed(noise_words)  # [33,2]
        noise_vectors = noise_vectors.view(size, N_SAMPLES, self.n_embed)  # [11,3,2]
        return noise_vectors  # [11,3,2]


# 定义损失函数，有公式的
class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        BATCH_SIZE, embed_size = input_vectors.shape  # 11,2
        # 将输入词向量与目标词向量作维度转化处理
        input_vectors = input_vectors.view(BATCH_SIZE, embed_size, 1)  # [11,2,1]
        output_vectors = output_vectors.view(BATCH_SIZE, 1, embed_size)  # [11,1,2]
        # 目标词损失
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()  # [11,1,1]
        out_loss = out_loss.squeeze()  # [11]
        # 负样本损失
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()  # [11,3,1]=[11,3,2],[11,2,1]
        noise_loss = noise_loss.squeeze().sum(1)  # [11,3] sum(1)后-> [11]
        # 综合计算两类损失a
        return -(out_loss + noise_loss).mean()


# 模型、损失函数及优化器初始化
model = SkipGramNeg(len(vocab2int), EMBEDDING_DIM, noise_dist=noise_dist)  # len(vocab2int)=13
criterion = NegativeSamplingLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# 训练
steps = 0
for e in range(EPOCHS):
    # 获取输入词以及目标词
    for input_words, target_words in get_batch(train_words, BATCH_SIZE, WINDOW_SIZE):
        steps += 1
        inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)  # [11] [11]-> 11是get_target随机
        # 输入、输出以及负样本向量
        input_vectors = model.forward_input(inputs)  # [11,2]
        output_vectors = model.forward_output(targets)  # [11,2]
        size, _ = input_vectors.shape  # size=11
        noise_vectors = model.forward_noise(size, N_SAMPLES)  # [11,3,2] = (11,3)
        # 计算损失
        loss = criterion(input_vectors, output_vectors, noise_vectors)
        # 打印损失
        if steps % PRINT_EVERY == 0:
            print("loss：", loss)
        # 梯度回传
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 可视化词向量
vectors = model.state_dict()["in_embed.weight"]  # 遍历每个词的向量，state_dict表示需要学习的参数
for i, w in int2vocab.items():
    x, y = float(vectors[i][0]), float(vectors[i][1])
    plt.scatter(x, y)
    plt.annotate(w, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()

"""
model.state_dict() is:
OrderedDict([('in_embed.weight', tensor([[-1.3737, -1.8997],
                      [-1.5391, -3.2624],
                      [-0.7309, -3.8968],
                      [-0.6842, -1.1463],
                      [-1.6471,  0.0805],
                      [-1.4861, -1.6632],
                      [-0.2397, -2.4783],
                      [-4.0195,  0.7375],
                      [-3.2610,  0.6795],
                      [-1.3382, -0.0562],
                      [-0.8774, -0.6674],
                      [-3.7043,  0.7699],
                      [-0.5483, -1.5039]])),
             ('out_embed.weight', tensor([[ 1.2225,  0.8790],
                      [ 2.1828,  0.9858],
                      [ 3.3304,  0.0257],
                      [ 0.8591, -0.0368],
                      [ 0.0477,  0.5192],
                      [ 0.9293,  0.9207],
                      [ 2.2949, -0.5741],
                      [ 0.5602,  3.6621],
                      [ 0.7391,  4.2809],
                      [ 0.0056,  0.0896],
                      [ 0.1862, -0.3433],
                      [ 0.6071,  3.9626],
                      [-0.1165, -0.3462]]))])
"""
