# author:Lenovo
# datetime:2023/4/18 19:29
# software: PyCharm
# project:pytorch项目


import math
import os
import random
import torch
from d2l import torch as d2l


d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

def read_ptb():

    data_dir=d2l.download_extract('ptb')

    with open(os.path.join(data_dir,'ptb.train.txt')) as f:
        raw_text=f.read()
    return [line.split() for line in raw_text.split('\n')]


#获取句子列表
sentences=read_ptb()
# print(sentences)
print("句子长度为：",len(sentences))


#构建word2id
vocab=d2l.Vocab(sentences,min_freq=10)
"""
一共有6719个种类的单词
"""
print("vocab_size:",len(vocab))

#下采样：
def subsample(sentences,vocab):

    #排除未知unk
    sentences=[[token for token in line if vocab[token] !=vocab.unk] for line in sentences]

    counter=d2l.count_corpus(sentences) #Counter计数器；
    num_tokens=sum(counter.values())

    # 如果在下采样期间保留词元，则返回True

    def keep(token):
        return (random.uniform(0,1) < math.sqrt(1e-4 /counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],counter)

#去掉出现概率较高的和较低的 subsampled是一个词的列表
subsampled, counter = subsample(sentences, vocab)

# print("subsampled:",subsampled)

d2l.show_list_len_pair_hist(
    ['origin', 'subsampled'], '# tokens per sentence',
    'count', sentences, subsampled);

def compare_counts(token):
    return (f'"{token}"的数量：'
            f'之前={sum([l.count(token) for l in sentences])}, '
            f'之后={sum([l.count(token) for l in subsampled])}')

print(compare_counts('the'))


corpus=[vocab[line] for line in subsampled]
print(corpus[:3])

# 中心词和上下文词的提取

def get_centers_and_contexts(corpus,max_windows_size):
    centers,contexts=[],[]
    #corpus是转为id的列表
    for line in corpus:
        if len(line)<2:
            continue
        centers+=line
        print("centers:",centers)
        for i in range(len(line)):# 上下文窗口中间i
            #窗口的大小
            window_size=random.randint(1,max_windows_size) #(1,2)之间生成一个
            # print("window_size:",window_size)
            indices=list(range(max(0,i-window_size),min(len(line),i+1+window_size)))
            # print("indices:",indices)
            #排除中心词：
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
            # print("contexts:",contexts)
    return centers,contexts

tiny_dataset = [list(range(7)), list(range(7, 10))]
print('数据集', tiny_dataset)

#max_window_size=2; 因为返回值有两个，所以需要加一个*
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('中心词', center, '的上下文词是', context)



all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# “中心词-上下文词对”的数量: {sum([len(contexts) for contexts in all_contexts])}'
"""
负采样
"""

class RandomGenerator:

    def __init__(self,sampling_weights):
        self.population=list(
            range(1,len(sampling_weights)+1)
        )
        self.sampling_weights=sampling_weights
        self.candidates=[]
        self.i=0

    def draw(self):

        if self.i == len(self.candidates):
            self.candidates=random.choices(
                self.population,self.sampling_weights,k=10000
            )
            self.i=0
        self.i+=1

        return self.candidates[self.i-1]

generator = RandomGenerator([2, 3, 4])
res=[generator.draw() for _ in range(10)]
print(res)


def get_negatives(all_contexts,vocab,counter,K):
    """返回负采样中的噪声词"""
    sampling_weights=[counter(vocab.to_tokens(i) ** 0.75) for i in range(1,len(vocab))]

    all_negatives,generator=[],RandomGenerator(sampling_weights)

    for contexts in all_contexts:
        negatives=[]

        while len(negatives)<len(context) *K:
            neg=generator.draw()

            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)

    return all_negatives

get_negatives(all_contexts,vocab,counter,5)



