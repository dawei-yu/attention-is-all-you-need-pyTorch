''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Models import Transformer
import heapq



# +
class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):
        

        super(Translator, self).__init__()

        self.alpha = 0.6
#        self.beam_size = beam_size #beam search first implemtn
        self.beam_size = 2*beam_size#beam search second implement
        self.ans_beam_size = beam_size#beam search second implement
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs', 
            torch.full((self.beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx


    def _model_decode(self, trg_seq, enc_output, memory_key_padding_mask=None):
        tmp_device = trg_seq.device
        tgt_mask = Transformer.generate_square_subsequent_mask(trg_seq.size(1)).to(tmp_device)
        dec_output = self.model.decode(trg_seq, enc_output,tgt_mask=tgt_mask)
        return F.softmax(dec_output, dim=-1)


    def _get_init_state(self, src_seq,enc_output, src_mask = None,src_key_padding_mask=None):
        beam_size = self.beam_size
        dec_output = self._model_decode(self.init_seq, enc_output)
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)
        scores = torch.log(best_k_probs)[0]
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        return gen_seq, scores


    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step,beam_size):
        assert len(scores.size()) == 1
        
        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)
        
        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)
        
        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = torch.div(best_k_idx_in_k2, beam_size, rounding_mode='trunc'), best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores


    def translate_sentence(self, src_seq):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0) == 1

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx 
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha 
        
        ansQ = [] 
        with torch.no_grad():
            enc_output = self.model.encode(src_seq)
        
            gen_seq, scores = self._get_init_state(src_seq,enc_output)
            enc_output = enc_output.repeat(beam_size,1,1)
            for step in range(2, max_seq_len):    # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], enc_output)
                
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step,beam_size)
                
                completed = torch.nonzero((gen_seq == trg_eos_idx).sum(1)).view(-1)
                
                
#                 if completed.size(0):  #beam search first implement
#                     incompleted = torch.tensor([i for i in range(beam_size) if i not in completed.tolist()])
#                     for i in completed:
#                         index = i.item()
#                         l.append((
#                                   scores[index].item() / (float(step+1) ** alpha),
#                                   gen_seq[index,:step+1]
#                                  ))
#                         beam_size -= 1
#                     if not beam_size:
#                         break
#                     gen_seq = gen_seq[incompleted]
#                     enc_output = enc_output[incompleted]
#                     scores = scores[incompleted]
                flag = 0
                if completed.size(0): #beam search second implement
                    for i in completed:
                        index = i.item()
                        score = scores[index].item() / (float(step+1) ** alpha)
                        if len(ansQ)<self.ans_beam_size or score>ansQ[0][0]:
                            if len(ansQ)==self.ans_beam_size: heapq.heappop(ansQ)
                            heapq.heappush(ansQ,(
                                  scores[index].item() / (float(step+1) ** alpha),
                                  gen_seq[index,:step+1]
                                  ))
                            flag = 1
                if (not flag and len(ansQ) == self.ans_beam_size) or (completed.size(0) == beam_size):
                    break

                    
                    
            maxscore = float('-inf')
            ans = []
#            print(ansQ)
            for score,seq in ansQ:
                if score > maxscore:
                    maxscore = score
                    ans = seq
        return ans.tolist()
