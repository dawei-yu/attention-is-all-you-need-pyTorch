#!/usr/bin/env python
# coding: utf-8
# %%
import pickle
from torch.utils.data import Dataset
from collections import  namedtuple
import os
import torch


# %%
def avg_parameters(files, output_dir):
    input_dir = 'output'
    print([os.path.join(input_dir, _) for _ in files])
    models = [torch.load(os.path.join(input_dir, _))['model'] for _ in files]
    
    avg_model = torch.load(os.path.join(input_dir,files[0]))
    for key in avg_model['model']:
    #     avg_model['model'][key] = sum([_[key] for _ in models]) / len(models) 
        avg_model['model'][key] = torch.true_divide(sum([_[key] for _ in models]), len(models))
    torch.save(avg_model, output_dir)


# %%
class MyDataset(Dataset):
    def __init__(self, src_path, trg_path):
        self.data_list = []
        
        with open(src_path,encoding='utf-8') as file:
            src_lines = file.readlines()
            
        with open(trg_path,encoding='utf-8') as file:
            trg_lines = file.readlines()       
            
        if len(src_lines) != len(trg_lines):
            print('data len unequal')
            raise
        
        pair = namedtuple('translation', ['src','trg'])
        
        for index in range(len(src_lines)):
            src_line = src_lines[index].strip()
            trg_line = trg_lines[index].strip()
            self.data_list.append(pair(src_line,trg_line))
                
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self,idx):
        return self.data_list[idx]
    
    def getDataset(self):
        return self.data_list


# %%
class MyTokenizer(object):
    def __init__(self, vocab_path, begin_token, end_token, unk_token, pad_token):
        with open(vocab_path,'rb') as file:
            self.vocab = pickle.load(file)
        self.begin_token = begin_token
        self.end_token = end_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        
    def get_vocab_size(self):
        return len(self.vocab)
    
    def idToToken(self, idx):
        return list(self.vocab)[idx]
    
    def tokenToId(self, token):
        if token not in self.vocab:
            return self.vocab[self.unk_token]
        return self.vocab[token]
    
    def batchEncoder(self, batch, maxlen=None):
        if maxlen == None:
            maxlen = 0
            for tmp in batch:
                maxlen = max(len(tmp.split()),maxlen) #sent_max_len
                
        res = []
        for tmp in batch:
            tmp_split = tmp.split()
            
            ids_l = [self.tokenToId(self.begin_token)]
            ids_l+=[self.tokenToId(token) for token in tmp_split]
            ids_l.append(self.tokenToId(self.end_token))
            ids_l += [self.tokenToId(self.pad_token)] * (maxlen-len(tmp_split))
            
            res.append(ids_l)
        return res #return len = sent_maxlen + 2
    
    def batchDecoder(self, batch):
        res = []
        for tmp in batch:
            token_l = []
            for ids in tmp:
                token = self.idToToken(ids)
                if token not in [self.begin_token, self.end_token, self.pad_token]:
                    token_l.append(token)
            res.append(' '.join(token_l))
        return res

