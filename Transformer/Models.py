#!/usr/bin/env python
# coding: utf-8
# %%
import torch
import math
import torch.nn as nn
from typing import Optional, Any
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerDecoder
import torch.nn.functional as F


# %%
class PositionalEncoding(nn.Module): 
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

        
    def forward(self, x):
#         print('x:',x.shape)
#         print('self.pe[:, : x.size(1)]',self.pe[:, : x.size(1)].shape)
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


# %%
class TransformerEncoderLayer(nn.Module):  ##Torch1.7 don't support norm_first, so overwrite
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, norm_first, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

#         src2 = self.self_attn(src, src, src, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)[0]
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)



class TransformerDecoderLayer(nn.Module): ##Torch1.7 don't support norm_first, so overwrite
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, norm_first, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
#         tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
#                               key_padding_mask=tgt_key_padding_mask)[0]
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)
#         tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
#                                    key_padding_mask=memory_key_padding_mask)[0]
#         tgt = tgt + self.dropout2(tgt2)
#         tgt = self.norm2(tgt)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout3(tgt2)
#         tgt = self.norm3(tgt)
#         return tgt
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


# %%
class Transformer(nn.Module):
    def __init__(self, src_tokenNum, tgt_tokenNum,
                 d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", norm_first = True,
                 max_len=150, src_pad_idx=None, tgt_pad_idx=None,share_weight=False
                ):
        super(Transformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, norm_first, dim_feedforward, dropout,
                                                activation)
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len = max_len)
        self.embedding_encoder = nn.Embedding(src_tokenNum, d_model,src_pad_idx)
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, norm_first, dim_feedforward, dropout,
                                                activation)
        decoder_norm = LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len = max_len)
        self.embedding_decoder = nn.Embedding(tgt_tokenNum, d_model,tgt_pad_idx)
        
        
        
        self.proj = nn.Linear(d_model, tgt_tokenNum, bias=False)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        if share_weight:  # Share the weight between target word embedding & last dense layer
            self.embedding_encoder.weight = self.proj.weight
            self.embedding_decoder.weight = self.proj.weight
            
    def forward(self, src: Tensor, tgt: Tensor, src_key_padding_mask=None, tgt_key_padding_mask=None) -> Tensor:
        
        tmp_device = src.device
        if src_key_padding_mask == None:
            src_key_padding_mask = (src == self.src_pad_idx) ##padding pos is True, True represent cover
        if tgt_key_padding_mask == None:
            tgt_key_padding_mask = (tgt == self.tgt_pad_idx)
        tgt_mask = Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tmp_device)
#         print(tgt_mask.shape)
#         print(src_key_padding_mask.shape)
#         print(tgt_key_padding_mask.shape)
#         print(src.shape)
        
        encoder_output = self.encode(src, src_key_padding_mask) #only use padding mask in encoder
        output = self.decode(tgt, encoder_output, 
                             tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, 
                             memory_key_padding_mask = src_key_padding_mask)
    
        return output
    
    def encode(self, src, src_key_padding_mask=None):
    
        src_embedding = self.embedding_encoder(src) * math.sqrt(self.d_model)
        encoder_input = self.pos_encoder(src_embedding) #pos and dropout
        
        encoder_input = encoder_input.transpose(0,1)
#         print('encoder_input:',encoder_input.shape)
        memory = self.encoder(encoder_input, src_key_padding_mask=src_key_padding_mask) 
        
        return memory.transpose(0,1)

    def decode(self, tgt, encoder_output, tgt_mask=None, tgt_key_padding_mask=None,memory_key_padding_mask=None):
        
        tgt_embedding = self.embedding_decoder(tgt) * math.sqrt(self.d_model)
        decoder_input = self.pos_decoder(tgt_embedding)
        
        decoder_input = decoder_input.transpose(0,1)
        encoder_output = encoder_output.transpose(0,1)
        
        decoder_output = self.decoder(decoder_input, encoder_output, 
                                    tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                      memory_key_padding_mask=memory_key_padding_mask
                                    )
        decoder_output = decoder_output.transpose(0,1)
#         print('decoder_output:',decoder_output.shape)
        output = self.proj(decoder_output)
    
        return output
    
    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


# %%
