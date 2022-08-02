# attention-is-all-you-need-pyTorch
A pyTorch implementations of the Transformer model in Attention is all you need.


The project is modified from [jadore801120/attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch). It seems that there are some problems in his project. 

1)In translation, beam search stop when k brances both get end token and the target sequence is picked from them.

2)The mask for decoder don't work.

3)Don't support multi-gpu training.

4)The package is so old that hard to learn and install.

5)Lack a function of averaging last five checkpoints.

Compared to the Fairseq, OpenMNT, the project is more clear for study and modifying.

But my reporduction can't reach 27.3 score on EN-DE task.

## Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)
- [Acknowledgement](#Acknowledgement)

## Requirements
torch>=1.7.1

## Usage
### preprocess data
```
bash prepare-wmt14en2de.sh
python getVocab.py
```

### train
I train on two RTX3090 24G. Only DDP training in train.py, you can set os.environ['CUDA_VISIBLE_DEVICES'] = '0' if you only train on one gpu.

If you are interesting at attention mechanism, you can read [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer/blob/master/AnnotatedTransformer.ipynb)
```
python train.py -vocab_path wmt14_en_de/vocab.pkl -src_train_path wmt14_en_de/train.en -trg_train_path wmt14_en_de/train.de -src_val_path wmt14_en_de/valid.en -trg_val_path wmt14_en_de/valid.de -share_weight -norm_first -epoch 50 -b 8192 -accum_step 2 
```

### test
There are two beam search strategies in translation. It seems that google use the second. Translation cost round 40 minutes. I got BLEU=25.0 in pred.de and BLEU=25.39 in predWithoutUNK.de
```
python translate.py -vocab_path wmt14_en_de/vocab.pkl -src_path wmt14_en_de/test.en -tgt_path wmt14_en_de/test.de -output pred.bpe.de -model output/model_epoch:49.chkpt
```

## Acknowledgement
- .sh is borrowed from [facebookresearch/faieseq](https://github.com/facebookresearch/fairseq/tree/main/examples/translation)
- Some model structures and some scripts is borrowed from [jadore801120/attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
