import flags as _flags
from torchdec import hlog
from model.seq import Encoder, Decoder, DecoderState, SimpleAttention, batch_seqs
from absl import flags
from collections import Counter, defaultdict
import numpy as np
import torch
from torch import nn


FLAGS = flags.FLAGS
flags.DEFINE_integer("n_emb", 64, "embedding size")
flags.DEFINE_integer("n_enc", 512, "encoder hidden size")
flags.DEFINE_float("dropout", 0, "dropout probability")
flags.DEFINE_boolean("copy_sup", False, "supervised copy")
flags.DEFINE_integer("beam", None, "decode with a beam")
flags.DEFINE_string("lstm_arch", 'akyurek', "use 'andreas'(GECA) or 'akyurek'(LexLSTM) architecture")

class GeneratorModel(nn.Module):
    def __init__(self, vocab, copy=False, self_attention=False):
        super().__init__()
        self.vocab = vocab

        if FLAGS.lstm_arch == 'andreas':
            lstm_arch_layer = 1
            FLAGS.n_emb = 64
        elif FLAGS.lstm_arch == 'akyurek':
            lstm_arch_layer = 2
            FLAGS.n_emb = 512
            
        self.encoder = Encoder(
            vocab,
            FLAGS.n_emb,
            FLAGS.n_enc,
            lstm_arch_layer,
            bidirectional=True,
            dropout=FLAGS.dropout
        )
        self.proj = nn.Linear(FLAGS.n_enc * 2, FLAGS.n_enc)
        self.decoder = Decoder(
            vocab,
            FLAGS.n_emb,
            FLAGS.n_enc,
            lstm_arch_layer,
            attention=[SimpleAttention(FLAGS.n_enc, FLAGS.n_enc)],
            copy=copy,
            self_attention=self_attention,
            dropout=FLAGS.dropout
        )
        #self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad())
        self.loss = nn.CrossEntropyLoss()

    def prepare(self, dataset):
        pass

    def forward(self, inp, out, _=None, __=None, batch_sum=True):
        enc, state = self.encoder(inp)
        enc = self.proj(enc)
        

        if FLAGS.lstm_arch == 'andreas':
            # andreas
            state = [s.sum(dim=0, keepdim=True) for s in state]
        elif FLAGS.lstm_arch == 'akyurek':
        # akyurek
            state = [
                s.view(2, -1, state[0].shape[1], 512).sum(dim=1)
                for s in state
                ]

        out_prev = out[:-1, :]
        out_next = out[1:, :]
        pred, _, _, (dpred, cpred) = self.decoder(
            state,
            out_prev.shape[0],
            out_prev,
            att_features=[enc],
            att_tokens=[inp]
        )
        n_seq, n_batch = out_next.shape
        if batch_sum == True:
            pred = pred.view(n_batch * n_seq, -1)
            out_next = out_next.contiguous().view(-1)
            loss = self.loss(pred, out_next)
        else:
            # loss (for each sample)
            sample_losses = list()
            for i in range(n_batch):
                sample_pred = pred[: , i , :] # shape = [out_len, label_num]
                sample_label = out_next[: , i] # shape = [out_len]
                # we now calculate loss for each single example
                sample_loss = self.loss(sample_pred, sample_label) # shape = []
                sample_losses.append(sample_loss)
            
            loss = torch.stack(sample_losses) # shape = [bs]
        return loss

    def aug_forward(self, inp, out, cand_repres, inp_mask,
                         subs_in_onehots, cand_repres_lens,
                         len_diffs, sta_ends, mask_cnts, padding, 
                         debugger=None, augmentor=None, manual_mode='train'):

        enc, state = self.encoder.aug_forward(inp, cand_repres,
                        inp_mask, subs_in_onehots, cand_repres_lens,
                        len_diffs, sta_ends, mask_cnts, padding, 
                        debugger, augmentor, manual_mode)
        enc = self.proj(enc)
        

        if FLAGS.lstm_arch == 'andreas':
            # andreas
            state = [s.sum(dim=0, keepdim=True) for s in state]
        elif FLAGS.lstm_arch == 'akyurek':
        # akyurek
            state = [
                s.view(2, -1, state[0].shape[1], 512).sum(dim=1)
                for s in state
                ]

        out_prev = out[:-1, :]
        out_next = out[1:, :]
        pred, _, _, (dpred, cpred) = self.decoder(
            state,
            out_prev.shape[0],
            out_prev,
            att_features=[enc],
            att_tokens=[inp],
            manual_mode=manual_mode
        )
        n_seq, n_batch = out_next.shape #shape = [49, 128]
  
        # print(pred.shape) # ([49, 128, 27] = [out_len, bs, label_num])
        if manual_mode == 'train':
            # overall loss
            pred = pred.view(n_batch * n_seq, -1)
            out_next = out_next.contiguous().view(-1)
            loss = self.loss(pred, out_next)

        elif manual_mode == 'eval':
            # loss (for each sample)
            sample_losses = list()
            for i in range(n_batch):
                sample_pred = pred[: , i , :] # shape = [out_len, label_num]
                sample_label = out_next[: , i] # shape = [out_len]
                # we now calculate loss for each single example
                sample_loss = self.loss(sample_pred, sample_label) # shape = []
                sample_losses.append(sample_loss)
            loss = torch.stack(sample_losses) # shape = [bs]
        return loss

    def sample(self, inp, greedy=False, beam=False):
        if beam and (FLAGS.beam is not None):
            preds = []
            scores = []
            for i in range(inp.shape[1]):
                p = self.beam(inp[:, i:i+1], FLAGS.beam)
                preds.append(p[0])
                scores.append(0)
            return preds, scores
        
        enc, state = self.encoder(inp)
        enc = self.proj(enc)
        #assert(enc.shape[1] == 1)
        #enc = enc.expand(-1, n_samples, -1).contiguous()

        if FLAGS.lstm_arch == 'andreas':
        # andreas
            state = [
            #s.sum(dim=0, keepdim=True).expand(-1, n_samples, -1).contiguous()
                s.sum(dim=0, keepdim=True)
                for s in state
                ]
        elif FLAGS.lstm_arch == 'akyurek':
        # akyurek
            state = [
                s.view(2, -1, state[0].shape[1], 512).sum(dim=1)
                for s in state
                ]
        
        return self.decoder.sample(
            state, 150, att_features=[enc], att_tokens=[inp], greedy=greedy
        )

    # TODO CODE DUP
    def beam(self, inp, beam_size):
        enc, state = self.encoder(inp)
        enc = self.proj(enc)
        state = [s.sum(dim=0, keepdim=True) for s in state]
        return self.decoder.beam(
            state, beam_size, 150, att_features=[enc], att_tokens=[inp]
        )
