#!/usr/bin/env python
# coding: utf-8
# %%
from collections import Counter,OrderedDict
import pickle


# %%
def getVocab(src_path,tgt_path,special_token,vocab_size,save_path):
    c = Counter()
    with open(src_path,'r',encoding = 'utf-8') as file:
        for item in file:
            c.update(item.split())
    with open(tgt_path,'r',encoding = 'utf-8') as file:
        for item in file:
            c.update(item.split())
    d = OrderedDict()
    count = 0
    for token in special_token:
        d[token] = count
        count += 1
    for double in c.most_common(vocab_size-count):
        d[double[0]] = count
        count += 1
    pickle.dump(d, open(save_path, "wb"))
    return c


# %%
# t = getVocab('wmt14_en_de/train.en','wmt14_en_de/train.de',['<unk>','<blank>','<s>','</s>'],32768,'vocab.pkl')

