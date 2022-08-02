#!/usr/bin/env python
# coding: utf-8
# %%
import sys
import argparse
import math
import time
from tqdm import tqdm
import random
import os
from collections import namedtuple
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

# %%
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Sampler, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR


# %%
from Transformer.tool import MyDataset, MyTokenizer
from Transformer import Constants
from Transformer.Models import Transformer


# %%
def cal_crossEntropy(pred, label, ignore_index, reduction='avg', smoothing=0.0):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    n_class = pred.size(-1)
    one_hot = F.one_hot(label,n_class)
    
#     soft_label = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1) #first implement
    soft_label = (1.0-smoothing)*one_hot + smoothing/n_class #second implement

    log_prb = F.log_softmax(pred, dim=-1)   
    loss = -(soft_label * log_prb).sum(-1)
    pad_mask = (label != ignore_index)
    loss = loss.masked_select(pad_mask).sum()
    
    pred = pred.max(-1)[1]
    n_correct = (pred == label).masked_select(pad_mask).sum().item()  
    n_word = pad_mask.sum().item()
     
    return loss/n_word, loss, n_correct, n_word


# %%
def seqShifted(seq):
    decoder_input, label = seq[:, :-1], seq[:, 1:] 
    return decoder_input, label


# %%
def train_epoch(model, training_data, optimizer, lr_scheduler, opt, gpu):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0 
    optimizer.zero_grad()
    desc = f'  - (Training) in gpu{gpu}'
    for batch in tqdm(training_data, mininterval=2, total=len(training_data), desc=desc, leave=False):
        # prepare data
        src_seq = batch.src.cuda(gpu)
        trg_seq, label = map(lambda x: x.cuda(gpu), seqShifted(batch.trg))
        
        #forward
        pred = model(src_seq, trg_seq)
        loss_word, loss_all, n_correct, n_word = cal_crossEntropy(
            pred, label, opt.trg_pad_idx, smoothing=opt.smoothing)
        loss_word = loss_word/opt.accum_step
        loss_word.backward()
        opt.train_step+=1
        
        if opt.train_step%opt.accum_step == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss_all.item()
        
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


# %%
def eval_epoch(model, validation_data, opt, gpu):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = f'  - (Validation) in gpu{gpu}'
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, total=len(validation_data), desc=desc, leave=False):

            # prepare data
            src_seq = batch.src.cuda(gpu)
            trg_seq, label = map(lambda x: x.cuda(gpu), seqShifted(batch.trg))

            # forward
            pred = model(src_seq, trg_seq)
            loss_word, loss_all, n_correct, n_word = cal_crossEntropy(
                pred, label, opt.trg_pad_idx, smoothing=opt.smoothing)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss_all.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


# %%
def train(model, training_data, validation_data, optimizer, lr_scheduler, opt, gpu):

    log_train_file = os.path.join(opt.output_dir, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, 'valid.log')

  
    if gpu == 0:
        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))
        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('device,epoch,loss,ppl,accuracy\n')
            log_vf.write('device,epoch,loss,ppl,accuracy\n')

    opt.train_step = 0

    for epoch_i in range(opt.epoch):
        training_data.batch_sampler.set_epoch(opt.seed + epoch_i)
        validation_data.batch_sampler.set_epoch(opt.seed + epoch_i)
        print('[ Epoch', epoch_i, ']')

        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, lr_scheduler, opt, gpu)            
        train_ppl = math.exp(min(train_loss, 100))
        
        # Current learning rate
        lr = optimizer.param_groups[0]['lr']

        valid_loss, valid_accu = eval_epoch(model, validation_data, opt, gpu)
        valid_ppl = math.exp(min(valid_loss, 100))
        

        checkpoint = {'settings': opt, 'model': model.module.state_dict()}

        if gpu == 0:
            model_name = f'model_epoch:{epoch_i}.chkpt'
            torch.save(checkpoint,os.path.join(opt.output_dir, model_name))
            print(f'model{epoch_i} saved.')
            
        dist.barrier()
        
        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{device_index},{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f},{lr:8.5f}\n'.format(
                device_index = gpu, epoch=epoch_i, loss=train_loss,
                ppl=train_ppl, accu=100*train_accu,lr=lr))
            log_vf.write('{device_index},{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                device_index = gpu, epoch=epoch_i, loss=valid_loss,
                ppl=valid_ppl, accu=100*valid_accu))


# %%
def learning_rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


# %%
def train_worker(
    gpu,
    ngpus,
    opt,
):
    
    is_main_process = gpu == 0

    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        random.seed(opt.seed)
        
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)
    torch.cuda.empty_cache()

    tokenizer = MyTokenizer(opt.vocab_path,\
                    begin_token=Constants.BOS_WORD, end_token=Constants.EOS_WORD, unk_token=Constants.UNK_WORD, pad_token=Constants.PAD_WORD)
    opt.src_vocab_size = opt.trg_vocab_size = tokenizer.get_vocab_size()
    opt.max_len = opt.filter_length + 2
    opt.src_pad_idx = opt.trg_pad_idx = tokenizer.tokenToId(Constants.PAD_WORD)
    opt.unk_idx = tokenizer.tokenToId(Constants.UNK_WORD)
    opt.bos_idx = tokenizer.tokenToId(Constants.BOS_WORD)
    opt.eos_idx = tokenizer.tokenToId(Constants.EOS_WORD)
    if is_main_process:
        print(f'src_pad_idx:{opt.src_pad_idx},trg_pad_idx:{opt.trg_pad_idx}')
        print('buildding model')
    model = Transformer(
        src_tokenNum = opt.src_vocab_size,
        tgt_tokenNum = opt.trg_vocab_size,
        d_model = opt.d_model,
        nhead = opt.n_head,
        num_encoder_layers = opt.n_layers,
        num_decoder_layers = opt.n_layers,
        dim_feedforward = opt.d_inner_hid,
        dropout = opt.dropout,
        max_len = opt.max_len,
        src_pad_idx = opt.src_pad_idx,
        tgt_pad_idx = opt.trg_pad_idx,
        share_weight = opt.share_weight,
        norm_first = opt.norm_first,
    ).cuda(gpu)
    
    dist.init_process_group(
        "nccl", init_method="env://", rank=gpu, world_size=ngpus
    )
    model = DDP(model, device_ids=[gpu])
    
    if is_main_process:
        print('reading data...')
        
    training_data, validation_data = prepareDataIter(opt,tokenizer)
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.base_lr, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer, lr_lambda=lambda step: learning_rate(step, model_size=opt.d_model, factor=opt.lr_factor, warmup=opt.warmup)
    )
   
    train(model, training_data, validation_data, optimizer, lr_scheduler, opt, gpu)


# %%
class DistributedBatchBucketSamplerByToken(Sampler):
    def __init__(self, dataset,num_replicas=None, rank=None, shuffle=True,batch_size = 10,filter_length=None,gap_length=20):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indices = []
        self.filter_length = filter_length
        self.gap_length = gap_length
        tmp_length = 0
        for i, example in enumerate(dataset):
            tokens_len = max(
                            len(example.src.split()),
                            len(example.trg.split())
                            )
            tmp_length = max(tmp_length,tokens_len)
            if filter_length:
                if tokens_len <= filter_length:
                    self.indices.append((i,tokens_len))
            else:
                self.indices.append((i,tokens_len))
        if not filter_length:
            self.filter_length = tmp_length
            
        self.num_samples = 0
        
        
    def __iter__(self):
        if self.shuffle:
            random.seed(self.epoch)
            
        gap = [i*self.gap_length for i in range(1,math.ceil(self.filter_length/self.gap_length)+1)]
        sentence_d = dict()
        count_d = dict()
        for index, length in self.indices:
            new_length = math.ceil(length/self.gap_length)*self.gap_length
            if new_length in sentence_d.keys():
                sentence_d[new_length].append(index)
            else:
                sentence_d[new_length] = [index]
            if new_length in count_d.keys():
                count_d[new_length] += length
            else:
                count_d[new_length] = length

        r_indices = []
        for g in sentence_d.keys():
            tmp_l = sentence_d[g]
            tmp_sentenceNum =  math.ceil(self.batch_size / (count_d[g] / len(tmp_l))) #一个batch应该蕴含的句子数
            tmp_iteration = math.ceil(len(tmp_l) * 1.0 / (self.num_replicas * tmp_sentenceNum)) #该gap每个gpu应该有的迭代次数
            tmp_total_size = tmp_iteration * self.num_replicas * tmp_sentenceNum
            if self.shuffle:
                random.shuffle(tmp_l)            
                
            tmp_l += tmp_l[:] * ((tmp_total_size - len(tmp_l)) // len(tmp_l)) + tmp_l[:((tmp_total_size - len(tmp_l)) % len(tmp_l))]    
            assert len(tmp_l) == tmp_total_size
            
            r_indices += [tmp_l[i+self.rank*tmp_sentenceNum:i + (self.rank+1)*tmp_sentenceNum] for i in range(0,len(tmp_l),tmp_sentenceNum*self.num_replicas)]


        if self.shuffle:
            random.shuffle(r_indices)
            
        self.num_samples = len(r_indices)
        
        return iter(r_indices)
        
    
    def __len__(self):
        return self.num_samples

    
    def set_epoch(self, epoch):
        self.epoch = epoch

# %%
# class DistributedBatchBucketSamplerBySentence(Sampler):
#     def __init__(self, dataset,tokenizer,src=None,trg=None,num_replicas=None, rank=None, shuffle=True,batch_size = 10,filter_length=None):
#         if src is None or trg is None:
#             raise RuntimeError("Need specify src")
#         if num_replicas is None:
#             if not dist.is_available():
#                 raise RuntimeError("Requires distributed package to be available")
#             num_replicas = dist.get_world_size()
#         if rank is None:
#             if not dist.is_available():
#                 raise RuntimeError("Requires distributed package to be available")
#             rank = dist.get_rank()
#         self.dataset = dataset
#         self.num_replicas = num_replicas
#         self.rank = rank
#         self.epoch = 0
#         self.shuffle = shuffle
#         self.batch_size = batch_size
#         self.indices = []
#         for i, example in enumerate(dataset):
#             tokens_len = max(
#                             len(example.src.split()),
#                             len(example.trg.split())
#                             )
#             if filter_length:
#                 if tokens_len <= filter_length:
#                     self.indices.append((i,tokens_len))
#             else:
#                 self.indices.append((i,tokens_len))
#         self.num_samples = int(math.ceil(len(self.indices) * 1.0 / (self.num_replicas * self.batch_size)))
#         self.total_size = self.num_samples * self.num_replicas * self.batch_size
        
        
#     def __iter__(self):
#         if self.shuffle:
#             random.seed(self.epoch)
#             random.shuffle(self.indices)
#         pooled_indices = []
        
#         self.indices += self.indices[:] * ((self.total_size-len(self.indices)) //  len(self.indices))+ \
#                         self.indices[:((self.total_size - len(self.indices)) % len(self.indices) )]
        
# #         self.indices += self.indices[:self.total_size - len(self.indices)]
#         assert len(self.indices) == self.total_size

#         for i in range(0, len(self.indices), self.batch_size * 100):
#             pooled_indices.extend(sorted(self.indices[i:i + self.batch_size * 100], key=lambda x: x[1]))

#         pooled_indices = [x[0] for x in pooled_indices]
#         r_indices = [pooled_indices[i+self.rank*self.batch_size:i + (self.rank+1)*self.batch_size] for i in range(0,len(pooled_indices),self.batch_size*self.num_replicas)]
#         assert len(r_indices) == self.num_samples
  
#         if self.shuffle:
#             random.shuffle(r_indices)
#         return iter(r_indices)

    
#     def __len__(self):
#         return self.num_samples

    
#     def set_epoch(self, epoch):
#         self.epoch = epoch

# %%
def prepareDataIter(opt, tokenizer):
    
    def collate_batch(batches):
        src_batch_l = []
        trg_batch_l = []
        sent_length = 0
        for b in batches:
            src_batch_l.append(b.src)
            trg_batch_l.append(b.trg)
        
        tokens_src_batch = tokenizer.batchEncoder(src_batch_l)
        tokens_trg_batch = tokenizer.batchEncoder(trg_batch_l)    
        Data = namedtuple('Data', ['src', 'trg'])
        data = Data(torch.tensor(tokens_src_batch, dtype = torch.int64),torch.tensor(tokens_trg_batch, dtype = torch.int64))
        return data
    
        
    training_dataset = MyDataset(opt.src_train_path,opt.trg_train_path)
    train_sampler = DistributedBatchBucketSamplerByToken(dataset = training_dataset ,batch_size = opt.batch_size, filter_length = opt.filter_length)
    training_data = DataLoader(
       training_dataset,
       batch_sampler=train_sampler,
       collate_fn=collate_batch,
    )
    
    validation_dataset = MyDataset(opt.src_val_path,opt.trg_val_path)
    val_sampler = DistributedBatchBucketSamplerByToken(dataset = validation_dataset ,batch_size = opt.batch_size, filter_length = opt.filter_length)
    validation_data = DataLoader(
        validation_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_batch,
    )    

    return training_data, validation_data


# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vocab_path', required=True)     # all-in-1 data pickle or bpe field
    parser.add_argument('-filter_length', type=int,default = 150) # the maximum length of sentences 
    
    parser.add_argument('-src_train_path', required=True)   # bpe encoded data
    parser.add_argument('-trg_train_path', required=True)
    parser.add_argument('-src_val_path', required=True)   # bpe encoded data
    parser.add_argument('-trg_val_path', required=True)
    
    parser.add_argument('-output_dir', default='output')             
    
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-accum_step', type=int, default=1)
    parser.add_argument('-b', '--batch_size', required=True, type=int)
    
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-share_weight',action='store_true')
    parser.add_argument('-norm_first',action='store_true')
    
    parser.add_argument('-warmup', type=int, default=4000)
    parser.add_argument('-base_lr', type=float, default=1.0)
    parser.add_argument('-lr_factor', type=float, default=1.0)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-smoothing', type = float, default=0.1)
    
    opt = parser.parse_args()

    # https://pytorch.org/docs/stable/notes/randomness.html
    # For reproducibility

    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise
        
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print(f"your sum of batch_size:{opt.batch_size}*{ngpus}*{opt.accum_step} = {opt.batch_size*ngpus*opt.accum_step}, sum of barch_size in paper is around 25000")
    print("Spawning training processes ...")

    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus,opt),
    )


# %%
if __name__ == '__main__':
    main()
