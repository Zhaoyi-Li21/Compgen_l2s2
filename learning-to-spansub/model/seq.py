from collections import namedtuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from absl import flags
FLAGS = flags.FLAGS
#_VF = torch._C._VariableFunctions
EPS = 1e-7

def batch_seqs(seqs):
    max_len = max(len(s) for s in seqs)
    data = np.zeros((max_len, len(seqs)))
    for i, s in enumerate(seqs):
        for j in range(len(s)):
            data[j, i] = s[j]
    
    return torch.LongTensor(data)

class Encoder(nn.Module):
    def __init__(
            self,
            vocab,
            n_embed,
            n_hidden,
            n_layers,
            bidirectional=True,
            dropout=0,
    ):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(len(vocab), n_embed, vocab.pad())
        self.embed_dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            n_embed, n_hidden, n_layers, bidirectional=bidirectional
        )

    def forward(self, data):
        emb = self.embed_dropout(self.embed(data))
        return self.rnn(emb)
    
    def aug_forward(self, inp, cand_repres, inp_mask, subs_in_onehots, 
                    cand_repres_lens, len_diffs, sta_ends, mask_cnt, padding,
                    debugger = None, augmentor = None, manual_mode='train'):
        '''
        this function is to generate embeddings of the augmented data
        '''

        inp_emb = self.embed(inp) # shape = [seq_len, bs, emb_dim]
        inp_t_emb = torch.transpose(inp_emb, 1, 0) # shape = [bs, seq_len, emb_dim]
        # inp_mask.shape = [bs, seq_len]
        inp_mask = inp_mask.unsqueeze(2) # shape = [bs, seq_len, 1], for broadcast
     
        left_part = inp_t_emb * inp_mask # shape = [bs, seq_len, emb_dim]

        cand_repres_emb = self.embed(cand_repres) # shape = [bs, len(strucs), span_len, emb_dim]
        bs, len_strucs = cand_repres_emb.shape[0], cand_repres_emb.shape[1]
        span_len, emb_dim = cand_repres_emb.shape[2], cand_repres_emb.shape[3]
        # subs_in_onehots.shape = [bs, len(strucs)]
        subs_in_onehots = subs_in_onehots.unsqueeze(1) # shape = [bs, 1, len(strucs)]
        cand_repres_emb = cand_repres_emb.view(bs, len_strucs, -1)
        # shape = [bs, len(strucs), span_len*emb_dim]
        subsin_part = torch.bmm(subs_in_onehots, cand_repres_emb)# shape = [bs, 1, sl*ed]
        subsin_part = subsin_part.squeeze(1) # shape = [bs, span_len * emb_dim]
        subsin_part = subsin_part.view(bs, span_len, emb_dim)
        # shape = [bs, span_len, emb_dim]

        # cand_repres_lens is a len=bs list
        # padding.shape = [seq_len]
        padding_emb = self.embed(padding) # shape = [seq_len, emb_dim]
        aug_emb = 0
        for i in range(bs):
            left_part_ = left_part[i, :, :]
            subsin_part_ = subsin_part[i, :, :]            
            len_diff = int(len_diffs[i])
            cand_repre_len = int(cand_repres_lens[i])
            ori_len = left_part_.shape[0]

            if mask_cnt[i] == 1:
                
                start, end = sta_ends[i]
                '''
                print('1span')
                print('start:',start,'---','end:',end)
                print('cand_repre_len:',cand_repre_len)
                print('len_diff',len_diff)
                '''
                
                if len_diff >= 0:
                    aug_emb_ = torch.cat((left_part_[: start, :], 
                                subsin_part_[: cand_repre_len, :], 
                                left_part_[end+1 : ori_len-1*len_diff, :]), dim=0).unsqueeze(0)
                else:
                    aug_emb_ = torch.cat((left_part_[: start, :], 
                                subsin_part_[: cand_repre_len, :], 
                                left_part_[end+1 :, :],
                                padding_emb[: -1*len_diff, :]), dim=0).unsqueeze(0)
                                
            elif mask_cnt[i] == 2:
                sta1, end1, sta2, end2 = sta_ends[i]
                '''
                print('1span')
                print('start:',sta1,sta2,'---','end:',end1,end2)
                print('cand_repre_len:',cand_repre_len)
                print('len_diff',len_diff)
                '''
                if len_diff >= 0:
                    aug_emb_ = torch.cat((left_part_[: sta1, :], 
                                subsin_part_[: cand_repre_len, :], 
                                left_part_[end1+1 : sta2, :],
                                subsin_part_[: cand_repre_len, :],
                                left_part_[end2+1 : ori_len-1*len_diff, :]), dim=0).unsqueeze(0)
                else:
                    aug_emb_ = torch.cat((left_part_[: sta1, :], 
                                subsin_part_[: cand_repre_len, :], 
                                left_part_[end1+1 : sta2, :],
                                subsin_part_[: cand_repre_len, :], 
                                left_part_[end2+1 :, :],
                                padding_emb[: -1*len_diff, :]), dim=0).unsqueeze(0)

            if i==0:
                aug_emb = aug_emb_ # shape = [1, seq_len, emb_dim]
            else:
                aug_emb = torch.cat((aug_emb, aug_emb_), dim=0) 

        # debugger.check_aug_para_grad(aug_emb, augmentor, True)

        # aug_emb.shape = [bs, seq_len, emb_dim]
        if manual_mode == 'train':
            emb = self.embed_dropout(aug_emb)
        elif manual_mode == 'eval':
            emb = aug_emb

        emb = torch.transpose(emb, 1, 0)
        return self.rnn(emb)
        

class SimpleAttention(nn.Module):
    def __init__(self, n_features, n_hidden, key=True, value=False):
        super().__init__()
        self.key = key
        self.value = value
        self.make_key = nn.Linear(n_features, n_hidden)
        self.make_val = nn.Linear(n_features, n_hidden)
        self.n_out = n_hidden

    def forward(self, features, hidden, mask):
        # key
        if self.key:
            key = self.make_key(features)
        else:
            key = features

        # attention
        hidden = hidden.expand_as(key)
        scores = (key * hidden).sum(dim=2) + mask * -99999 # "infinity"
        distribution = F.softmax(scores, dim=0)
        weighted = (features * distribution.unsqueeze(2).expand_as(features))
        summary = weighted.sum(dim=0, keepdim=True)

        # value
        if self.value:
            return self.make_val(summary), distribution
        else:
            return summary, distribution

DecoderState = namedtuple("DecoderState", "feed rnn_state hiddens tokens")
BeamState = namedtuple("BeamState", "feed rnn_state hiddens tokens score parent done")

class Decoder(nn.Module):
    def __init__(
            self, 
            vocab, 
            n_embed, 
            n_hidden, 
            n_layers, 
            attention=None,
            copy=False,
            self_attention=False,
            dropout=0
    ):
        super().__init__()

        # setup
        self.vocab = vocab
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.copy = copy
        self.self_attention = self_attention

        # attention
        if attention is None:
            attention = ()
        attention = tuple(attention)
        if self_attention:
            attention = attention + (SimpleAttention(n_hidden, n_hidden),)
        self.attention = attention
        for i, att in enumerate(attention):
            self.add_module("attention_%d" % i, att)

        # modules
        self.embed = nn.Embedding(len(vocab), n_embed, vocab.pad())
        self.combine = nn.Linear(n_hidden * (1 + len(attention)), n_hidden)
        self.dropout_in = nn.Dropout(dropout)
        self.predict = nn.Linear(n_hidden, len(vocab))
        self.copy_switch = nn.Linear(n_hidden, 1 + len(attention))
        self.rnn = nn.LSTM(n_embed + n_hidden, n_hidden, n_layers)
        self.dropout_out = nn.Dropout(dropout)
    
    def step(
            self,
            decoder_state,
            att_features,
            att_tokens,
            att_masks,
            att_token_proj,
            self_att_proj,
            manual_mode = 'train'
    ):
        # advance rnn
        emb = self.embed(decoder_state.tokens[-1, :])
        if manual_mode == 'train':
            inp = self.dropout_in(torch.cat((emb, decoder_state.feed), dim=1))
        elif manual_mode == 'eval':
            inp = torch.cat((emb, decoder_state.feed), dim=1)

        hidden, rnn_state = self.rnn(inp.unsqueeze(0), decoder_state.rnn_state)
        hiddens = torch.cat(decoder_state.hiddens + [hidden], dim=0)

        # prep self-attention
        if self.self_attention:
            att_features = tuple(att_features) + (hiddens,)
            att_tokens = tuple(att_tokens) + (decoder_state.tokens,)
            att_masks = att_masks + (
                (decoder_state.tokens == self.vocab.pad()).float(),
            )
            att_token_proj = att_token_proj + (self_att_proj,)

        # advance attention
        attended = [
            attention(features, hidden, mask) 
            for attention, features, mask in zip(
                self.attention, att_features, att_masks
            )
        ]
        if len(attended) > 0:
            summary, distribution = zip(*attended)
        else:
            summary = distribution = ()
        all_features = torch.cat([hidden] + list(summary), dim=2)

        if manual_mode == 'train':
            comb_features = self.dropout_out(self.combine(all_features).squeeze(0))
        elif manual_mode == 'eval':
            comb_features = self.combine(all_features).squeeze(0)

        pred_logits = self.predict(comb_features)
        
        # copy mechanism
        ### if self.copy:
        ###     pred_probs = F.softmax(pred_logits, dim=1)
        ###     copy_probs = [
        ###         (dist.unsqueeze(2) * proj).sum(dim=0)
        ###         for dist, proj in zip(distribution, att_token_proj)
        ###     ]
        ###     all_probs = torch.stack([pred_probs] + copy_probs, dim=1)
        ###     copy_weights = F.softmax(self.copy_switch(comb_features), dim=1)
        ###     comb_probs = (copy_weights.unsqueeze(2) * all_probs).sum(dim=1)
        ###     pred_logits = torch.log(comb_probs)
        if self.copy:
            pred_probs = F.softmax(pred_logits, dim=1)
            dist, = distribution
            proj, = att_token_proj
            copy_probs = (dist.unsqueeze(2) * proj).sum(dim=0)
            copy_probs += EPS
            copy_weights = pred_probs[:, self.vocab.copy()].unsqueeze(1)
            comb_probs = (
                copy_weights * copy_probs + (1 - copy_weights) * pred_probs
            )
            direct_logits = pred_logits
            copy_logits = torch.log(copy_probs)
            pred_logits = torch.log(comb_probs)
        else:
            direct_logits = pred_logits
            copy_logits = None

        # done
        return (
            pred_logits,
            comb_features,
            rnn_state,
            hidden,
            direct_logits, copy_logits
        )

    def _make_projection(self, tokens):
        proj = tokens.new_zeros(
            tokens.shape[0], tokens.shape[1], len(self.vocab)
        ).float()
        for i in range(tokens.shape[0]):
            proj[i, range(tokens.shape[1]), tokens[i, :]] = 1
            #proj[i, :, tokens[i, :]] = 1
        return proj

    def forward(
            self,
            rnn_state,
            max_len,
            ref_tokens=None,
            att_features=None,
            att_tokens=None,
            token_picker=None,
            manual_mode='train'
    ):
        # token picker
        if token_picker is None:
            self_att_proj = self._make_projection(ref_tokens)
            token_picker = lambda t, logits: (
                (ref_tokens[t, :], self_att_proj[:t+1, :, :])
            )

        # attention
        if att_features is None:
            att_features = ()
            att_tokens = ()
            att_masks = ()
            att_token_proj = ()
        else:
            assert isinstance(att_features, list) \
                or isinstance(att_features, tuple)
            att_masks = tuple(
                (toks == self.vocab.pad()).float() for toks in att_tokens
            )
            att_token_proj = tuple(
                self._make_projection(toks) for toks in att_tokens
            )

        # init
        pred = None
        dummy_tokens, _ = token_picker(0, pred)
        feed = dummy_tokens.new_zeros(
            dummy_tokens.shape[0], self.n_hidden
        ).float()
        hiddens = []
        all_tokens = []
        all_preds = []
        all_extra = []

        # iter
        for t in range(max_len):
            tokens, self_att_proj = token_picker(t, pred)
            if tokens is None:
                break
            all_tokens.append(tokens)
            decoder_state = DecoderState(
                feed, rnn_state, hiddens, torch.stack(all_tokens)
            )

            pred, feed, rnn_state, hidden, *extra = self.step(
                decoder_state,
                att_features,
                att_tokens,
                att_masks,
                att_token_proj,
                self_att_proj,
                manual_mode
            )
            hiddens.append(hidden)
            all_preds.append(pred)
            all_extra.append(extra)

        return (
            torch.stack(all_preds),
            torch.stack(all_tokens),
            rnn_state,
            list(zip(*all_extra))
        )

    def sample(
            self,
            rnn_state,
            max_len,
            att_features=None,
            att_tokens=None,
            greedy=False
    ):
        # init
        n_batch = rnn_state[0].shape[1]
        device = rnn_state[0].device

        done = [False for _ in range(n_batch)]
        running_proj = torch.zeros(max_len, n_batch, len(self.vocab)).to(device)

        def token_picker(t, logits):
            # first step
            if t == 0:
                toks = torch.LongTensor(
                    [self.vocab.sos() for _ in range(n_batch)]
                ).to(device)
                running_proj[0, range(n_batch), toks] = 1
                #running_proj[0, :, toks] = 1
                return toks, running_proj[:1, :, :]
            if all(done):
                return None, None

            # sample
            probs = F.softmax(logits, dim=1)
            probs = probs.detach().cpu().numpy()
            tokens = []
            for i, row in enumerate(probs):
                if done[i]:
                    tokens.append(self.vocab.pad())
                    continue
                row[self.vocab.copy()] = 0
                if greedy:
                    choice = np.argmax(row)
                else:
                    row /= row.sum()
                    choice = np.random.choice(len(self.vocab), p=row)
                tokens.append(choice)
                if choice == self.vocab.eos():
                    done[i] = True

            toks = torch.LongTensor(tokens).to(device)
            running_proj[t, :, toks] = 1
            return toks, running_proj[:t+1, :, :]

        preds, tokens, rnn_state, *_ = self(
            rnn_state,
            max_len,
            att_features=att_features,
            att_tokens=att_tokens,
            token_picker=token_picker
        )
        tok_arr = tokens.detach().cpu().numpy().transpose()
        tok_out = []
        score_out = [0 for _ in range(tok_arr.shape[0])]
        for i, row in enumerate(tok_arr):
            row_out = []
            for t, c in enumerate(row):
                row_out.append(c)
                score_out[i] += preds[t, i, c].item()
                if c == self.vocab.eos():
                    break
            tok_out.append(row_out)
        return tok_out, score_out

    def beam(
            self,
            rnn_state,
            beam_size,
            max_len,
            att_features=None,
            att_tokens=None,
    ):
        assert rnn_state[0].shape[1] == 1
        device = rnn_state[0].device

        # init attention
        if att_features is None:
            att_features = ()
            att_tokens = ()
            att_masks = ()
            att_token_proj = ()
        else:
            assert isinstance(att_features, list) \
                or isinstance(att_features, tuple)
            att_masks = tuple(
                (toks == self.vocab.pad()).float() for toks in att_tokens
            )
            att_token_proj = tuple(
                self._make_projection(toks) for toks in att_tokens
            )

        # initialize beam
        beam = [BeamState(
            rnn_state[0].new_zeros(self.n_hidden),
            [s.squeeze(1) for s in rnn_state],
            [],
            [self.vocab.sos()],
            0.,
            None,
            False
        )]

        for t in range(max_len):
            if all(s.done for s in beam):
                break
            rnn_state = [
                torch.stack([s.rnn_state[i] for s in beam], dim=1)
                for i in range(len(beam[0].rnn_state))
            ]
            tokens = torch.LongTensor([
                [s.tokens[tt] if tt < len(s.tokens) else s.tokens[-1] for s in beam] 
                for tt in range(t+1)
            ]).to(device)
            decoder_state = DecoderState(
                torch.stack([s.feed for s in beam]),
                rnn_state,
                [torch.stack(
                    [s.hiddens[tt] if tt < len(s.hiddens) else s.hiddens[-1] for s in beam],
                dim=1) for tt in range(t)],
                tokens,
            )
            self_att_proj = self._make_projection(tokens)
            pred, feed, rnn_state, hidden, *_ = self.step(
                decoder_state,
                tuple(f.expand(f.shape[0], len(beam), f.shape[2]) for f in att_features),
                tuple(t.expand(t.shape[0], len(beam)) for t in att_tokens),
                tuple(m.expand(m.shape[0], len(beam)) for m in att_masks),
                tuple(p.expand(p.shape[0], len(beam), p.shape[2]) for p in att_token_proj),
                self_att_proj
            )

            logprobs = F.log_softmax(pred, dim=1)
            next_beam = []
            for i, row in enumerate(logprobs):
                row[self.vocab.copy()] = -np.inf
                scores, toks = row.topk(beam_size)
                if beam[i].done:
                    next_beam.append(beam[i])
                else:
                    for s, t in zip(scores, toks):
                        next_beam.append(BeamState(
                            feed[i, :],
                            [s[:, i, :] for s in rnn_state],
                            beam[i].hiddens + [hidden[:, i, :]],
                            beam[i].tokens + [t.item()],
                            beam[i].score + s,
                            beam[i],
                            t == self.vocab.eos()
                        ))
            next_beam = sorted(next_beam, key=lambda x: -x.score)
            beam = next_beam[:beam_size]

        return [s.tokens for s in beam]
