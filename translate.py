# -*- coding: utf-8 -*-
''' Translate input text with trained model. '''

import torch
import argparse
from tqdm import tqdm
import random
from Transformer.tool import MyDataset, MyTokenizer, avg_parameters
from Transformer.Translator import Translator
from Transformer import Constants
from Transformer.Models import Transformer
import os

# def load_model(opt, device):
def load_model(model, device):
    checkpoint = torch.load(model, map_location=device)
    
    model_opt = checkpoint['settings']
    model = Transformer(
        src_tokenNum = model_opt.src_vocab_size,
        tgt_tokenNum = model_opt.trg_vocab_size,
        d_model = model_opt.d_model,
        nhead = model_opt.n_head,
        num_encoder_layers = model_opt.n_layers,
        num_decoder_layers = model_opt.n_layers,
        dim_feedforward = model_opt.d_inner_hid,
        dropout = model_opt.dropout,
        max_len = model_opt.max_len,
        src_pad_idx = model_opt.src_pad_idx,
        tgt_pad_idx = model_opt.trg_pad_idx,
        share_weight = model_opt.share_weight,
        norm_first = model_opt.norm_first,
    ).to(device)
        
    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model 


# +
def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model weight file')
    parser.add_argument('-vocab_path',required=True,
                        help='Pickle file with both instances and vocabulary.')
    parser.add_argument('-src_path',required=True,
                        help='datasets folder path')
    parser.add_argument('-tgt_path',required=True
                       )
    parser.add_argument('-output', required=True,
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    
    parser.add_argument('-beam_size', type=int, default=4)
    parser.add_argument('-max_seq_len', type=int, default=150)
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    
#     print(opt.src_path,opt.tgt_path)
    dataset = MyDataset(opt.src_path,opt.tgt_path)
    tokenizer = MyTokenizer(opt.vocab_path,\
                    begin_token=Constants.BOS_WORD, end_token=Constants.EOS_WORD, unk_token=Constants.UNK_WORD, pad_token=Constants.PAD_WORD)
    
    opt.src_pad_idx = opt.trg_pad_idx = tokenizer.tokenToId(Constants.PAD_WORD)
    opt.unk_idx = tokenizer.tokenToId(Constants.UNK_WORD)
    opt.bos_idx = tokenizer.tokenToId(Constants.BOS_WORD)
    opt.eos_idx = tokenizer.tokenToId(Constants.EOS_WORD)
    device = torch.device('cuda:0' if opt.cuda else 'cpu')
    model = load_model(opt.model, device)
    
    translator = Translator(
        model=model,
        beam_size=opt.beam_size,
        max_seq_len=opt.max_seq_len,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_bos_idx=opt.bos_idx,
        trg_eos_idx=opt.eos_idx).to(device)

    

    with open(opt.output, 'w',encoding='utf-8') as f:
        for example in tqdm(dataset, mininterval=2, desc='  - (Test)', leave=False):
#            print(' '.join(example.src))
            src_ids = torch.LongTensor(tokenizer.batchEncoder([example.src])).to(device)
            pred_ids = translator.translate_sentence(src_ids)
            pred_line = tokenizer.batchDecoder([pred_ids])[0]
#            print(pred_line)
            f.write(pred_line.strip() + '\n')
    print('[Info] Finished.')


# +
if __name__ == "__main__":
    '''
    Usage: python translate.py -model trained.chkpt 
    '''
    
#     avg_parameters(['model_epoch%3A45.chkpt','model_epoch%3A46.chkpt','model_epoch%3A47.chkpt',\
#                 'model_epoch%3A48.chkpt','model_epoch%3A49.chkpt'],'output/avg.chkpt')
    
    main()
    
    os.system(r"cat pred.bpe.de | sed -E 's/(@@ )|(@@ ?$)//g' > pred.de")
    
    f = open ('predWithoutUNK.de','w',encoding = 'utf-8')
    with open('pred.de','r') as file:
        for i in file:
            f.write(i.replace(' <unk>',''))
    f.close()
    
    print('pred.de score:')
    os.system(r'perl mosesdecoder/scripts/generic/multi-bleu.perl wmt14_en_de/tmp/test.de < pred.de')
    print('predWithoutUNK.de score')
    os.system(r'perl mosesdecoder/scripts/generic/multi-bleu.perl wmt14_en_de/tmp/test.de < predWithoutUNK.de')
