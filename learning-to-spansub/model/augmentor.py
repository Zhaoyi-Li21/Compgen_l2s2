'''
augmentation architecture flow
version2, for spansub
'''
from model.augmentor_utils import Span_Extractor_Module
from unicodedata import bidirectional
import torch.nn as nn
import torch
import torch.nn.functional as F
import random
import flags as _flags
from absl import app, flags, logging

flags.DEFINE_float("k_cos", 2., "k_cos * cos_sim belongs to [-k_cos, k_cos]")
flags.DEFINE_integer("num_sample_acts", 4, "sample how many acts for each sentence")
flags.DEFINE_float("so_scale_factor", 1., "real_prob = prob * scale_factor")
flags.DEFINE_float("si_scale_factor", 1., "real_prob = prob * scale_factor")

FLAGS = flags.FLAGS
neg_inf = -1e4
PAD = 0


class AugEncoder_Module(nn.Module):
    '''
    func descrip: to get encode for the whole input sentence,
    model type : Bi-LSTM(done) or Transformer-encoder(to do)
    '''
    def __init__(
        self,
        vocab,
        n_embed,
        n_hidden,
        n_layers=1,
        dropout=0
    ):
        super(AugEncoder_Module, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(len(vocab), n_embed, vocab.pad())
        self.embed_dropout = nn.Dropout(dropout)
        self.bi_lstm = nn.LSTM(
            n_embed, n_hidden, n_layers, bidirectional=True
        )
    
    def forward(self, input):
        # input.shape:(seq_len, batch_size)
        # embedding.shape:(seq_len, bs, emb_dim)
        emb = self.embed_dropout(self.embed(input))
        output, _ = self.bi_lstm(emb)
        # output.shape = (seq_length,batch_size,num_directions*hidden_size)
        return output


class AugmentModel(nn.Module):
    '''
    func descrip: top-model, leverage different modules to emit
        indicators for generating augmented data according to the original 
        training data
    '''
    def __init__(
        self,
        # `encoder` : LSTM
        vocab=None,
        encoder_n_embed=None,
        encoder_n_hidden=None,
        encoder_n_layers=1,
        encoder_dropout=0,

        # `span_extractor`
        span_pool=None, 
        inp_candidates=None, 
        span_encodes=None,
        exchanges=None,
        enc2type=None,
        # `embedding layer g`, embedding for candidates in the inp sentence
        g_n_embed=None,
        # `embedding layer h`, embedding for candidates in the structure pool
        f_n_embed=None,

        span_decodes=None,
        inp2out=None,
        ):
        super(AugmentModel, self).__init__()
        # `encoder` : LSTM
        self.vocab = vocab
        self.inp_encoder = AugEncoder_Module(vocab, encoder_n_embed, encoder_n_hidden,
                        encoder_n_layers, encoder_dropout)
        # `span_extractor`
        self.span_extractor = Span_Extractor_Module(span_pool, inp_candidates, 
                                span_encodes, exchanges, vocab, enc2type)
        # `embedding layer g`
        self.proj_1 = nn.Linear(4*encoder_n_hidden, g_n_embed)
        self.embed_g = nn.Embedding(len(span_encodes), g_n_embed, PAD)
        # `embedding layer f`
        self.embed_f = nn.Embedding(len(span_encodes), f_n_embed, PAD)
        self.proj_2 = nn.Linear(2*g_n_embed, f_n_embed)
        
        self.span_decodes = span_decodes
        self.inp2out = inp2out

    def gen_sub_info(self, inp, inp_ids, debugger, task='scan', temperature=1., sample_multi_acts=True):
        # inp.shape = [seq_len, bs]
        # inp_ids = list, len(inp_ids) = bs
        inp_enc = self.inp_encoder(inp)
        # inp_enc.shape = [seq_len, bs, num_directions*hidden_size]
        '''
        we need to transform inp_enc to a context vector
        '''
        
        _inp_context = torch.cat((inp_enc[0],inp_enc[-1]),-1)
        # _inp_context.shape = [bs, 2*num_directions*hidden_size]
        # inp_context = self.proj_1(_inp_context) 
        # inp_context.shape = [bs, g_n_embed]
        inp_context = _inp_context
        # debug, considering to removing proj_1

        debugger.check_vector_orient(inp_context)
        #debugger.check_vector_orient(_inp_context)


        inp_t = inp.transpose(1, 0) # inp_t.shape = [bs, seq_len]
        inps = inp_t.tolist()

        if task == 'scan': mode = 0
        elif task == 'cogs': mode = 1

        ret = self.span_extractor.get_inp_candidates(inps, inp_ids, mode)
        bat_cands_enc, _, _, _, _, _ = ret
        _bat_cands_enc = torch.tensor(bat_cands_enc, dtype=int).to(_flags.device()) # shape=[bs, fix_num]
        
        bat_cands_emb = self.embed_g(_bat_cands_enc) # shape=[bs, fix_num, g_n_emb]
       
        # sim_subs_out_logits = torch.bmm(bat_cands_emb, inp_context.unsqueeze(2)).squeeze(2)
        inp_context_ = inp_context.unsqueeze(1) # shape = [bs, 1, g_n_embed]
        sim_subs_out_logits = FLAGS.k_cos * torch.cosine_similarity(bat_cands_emb, inp_context_, dim=2)
        sim_subs_out_logits = sim_subs_out_logits * FLAGS.so_scale_factor
        # sim_subs_out_logits.shape = [bs, fix_num]

        _, batch_mask_indicat, _, _, _, _ = ret

        assert(len(batch_mask_indicat) == sim_subs_out_logits.shape[0])

        for i in range(len(batch_mask_indicat)):
            for j in range(FLAGS.fix_num_span):
                if j >= batch_mask_indicat[i]:
                    sim_subs_out_logits[i,j] = neg_inf
                    
                    # set those paddings as neg_inf 
                    # so that there is no prob for them to be subs out
        
        '''
        add FLAGS.num_sample_acts
        '''
        _, _, bat_reminder, inps_mask, inps_spans, inps_mask_cnt = ret
        # bat_reminder:list, only for `cogs` task, ['span2span','lex2span',...]
        # inps_mask : list [inp_mask1,inp_mask2,...], inp_mask1: [[0,1,1,...],[0,0,0,...],...]
        # inps_spans : list [inp1_spans,inp2_spans,...], inp1_spans: ['jump around left',...]

        _inps_mask = torch.tensor(inps_mask, dtype=float).float().to(_flags.device()) # shape = [bs, fns, seq_len]

        so_one_hots_li = list()
        si_one_hots_li = list()
        sel_cand_repres_lens_li = list()
        sel_len_diffs_li = list()
        sel_sta_ends_li = list()
        sel_out_spans_li = list()
        sel_in_spans_li = list()
        sel_inps_mask_cnt_li = list()
        
        if sample_multi_acts == True:
            num_sample_acts = FLAGS.num_sample_acts
        else:
            num_sample_acts = 1

        for _ in range(num_sample_acts):
            so_one_hots = F.gumbel_softmax(sim_subs_out_logits, tau=temperature, hard=True)
            # so_one_hots.shape = [bs, fix_num]
            so_one_hots_ = so_one_hots.unsqueeze(1) # shape = [bs, 1, fix_num]
            so_one_hots_li.append(so_one_hots_)

            prev_act_enc = torch.bmm(so_one_hots_, bat_cands_emb) # shape = [bs, 1, g_n_emb]
            prev_act_enc = prev_act_enc.squeeze(1)
            # prev_act_enc.shape = [bs, 2*num_directions*hidden_size]
            # inp_context.shape = [bs, 2*num_directions*hidden_size]
            lat_query_ = torch.cat((inp_context, prev_act_enc), 1)
            # lat_query_.shape = [bs, 2*2*num_direction*hidden_size]
            lat_query = self.proj_2(lat_query_)# lat_query.shape = [bs, f_n_embed]
            
            __bat_cands_enc = _bat_cands_enc.type(torch.float).unsqueeze(2) # shape = [bs, fix_num, 1]
            out_spans = torch.bmm(so_one_hots_, __bat_cands_enc) # shape = [bs, 1, 1]
            out_spans = out_spans.squeeze(2) # shape = [bs, 1]
            out_spans = out_spans.squeeze(1) # shape = [bs]

            out_spans = out_spans.detach().type(torch.int) # cut-off the gradient from this step
            out_spans = out_spans.tolist()
        
            ret = self.span_extractor.get_cands_from_pool(out_spans, bat_reminder, mode)
            pool_cands, exchangeables, sampled_in_repres = ret
            _pool_cands = torch.tensor(pool_cands, dtype=int).to(_flags.device())
            # _pool_cands.shape = [bs, len(strucs)]
            pool_cands_emb = self.embed_f(_pool_cands)
            # pool_cands_emb.shape = [bs, len(strucs), f_n_embed]
            
            # sim_subs_in_logits = torch.bmm(pool_cands_emb, lat_query.unsqueeze(2)).squeeze(2)
            
            _lat_query = lat_query.unsqueeze(1) # shape = [bs, 1, f_n_embed]
            sim_subs_in_logits = FLAGS.k_cos * torch.cosine_similarity(pool_cands_emb, _lat_query, dim=2)
            sim_subs_in_logits = sim_subs_in_logits * FLAGS.si_scale_factor
            # shape = [bs, len(strucs)]

            for i in range(sim_subs_in_logits.shape[0]):
                exchangeable = exchangeables[i]
                for j in range(sim_subs_in_logits.shape[1]):
                    if j not in exchangeable:
                        sim_subs_in_logits[i,j] = neg_inf
        
            si_one_hots = F.gumbel_softmax(sim_subs_in_logits, tau=temperature, hard=True)
            # shape = [bs, len(strucs)]
            si_one_hots_li.append(si_one_hots)

            '''
            (1):in this following part, we will leverage the 'so_one_hots'
            to fetch the so_mask, whose shape should be like [bs, seq_len];        
            '''
            # inps_mask is a 3-level list [bs, fixed_num_span, seq_len]
            # so_one_hots_.shape = [bs, 1, fixed_num_span]
            sel_inps_mask = torch.bmm(so_one_hots_, _inps_mask).squeeze(1) # shape = [bs, seq_len]
        
            '''
            (2):in this following part, we will do prepare-work about normalization
            of the recombined data:
            e.g.,: sampled_repres need to be padded to the same length;
            and original inps'padding also need to be modified in case that 
            the length of recombined seq > the default max_seq_len. 
            '''

            # sampled_in_repres is a 2-level list: [bs, len(strucs)], [str]
            # si_one_hots.shape = [bs, len(strucs)]
            si_ids = torch.argmax(si_one_hots, dim=1).detach().tolist()
            so_ids = torch.argmax(so_one_hots, dim=1).detach().tolist()
            sel_out_spans = list()
        
            for i in range(len(so_ids)):
                so_id = so_ids[i]
                sel_out_span = inps_spans[i][so_id]
                sel_out_spans.append(sel_out_span)
            sel_out_spans_li.append(sel_out_spans)
            
            # debugging
            debugger.detector_logits_onehot(sim_subs_out_logits, so_one_hots,
                                sim_subs_in_logits, si_one_hots, 
                                bat_cands_enc, batch_mask_indicat, exchangeables, pool_cands,
                                sample_multi_acts)

            ret = self.get_rest_info(so_ids, si_ids, inps_mask_cnt, 
                                        pool_cands, sel_inps_mask,
                                        sampled_in_repres, task)
            cand_repres, sel_cand_repres_lens, sel_len_diffs, sel_sta_ends, \
                 sel_in_spans, sel_inps_mask_cnt = ret
            # `cand_repres` is shared, `sel_*` is private
            sel_cand_repres_lens_li.append(sel_cand_repres_lens)
            sel_len_diffs_li.append(sel_len_diffs)
            sel_sta_ends_li.append(sel_sta_ends)
            sel_in_spans_li.append(sel_in_spans)
            sel_inps_mask_cnt_li.append(sel_inps_mask_cnt)

        if sample_multi_acts == True:
            return cand_repres, _inps_mask, so_one_hots_li, si_one_hots_li, \
                sel_cand_repres_lens_li, sel_len_diffs_li, sel_sta_ends_li, \
                    sel_out_spans_li, sel_in_spans_li, sel_inps_mask_cnt_li
        else:
            return cand_repres, _inps_mask, so_one_hots_li[0], si_one_hots_li[0], \
                sel_cand_repres_lens_li[0], sel_len_diffs_li[0], sel_sta_ends_li[0], \
                    sel_out_spans_li[0], sel_in_spans_li[0], sel_inps_mask_cnt_li[0]

    def get_rest_info(self, so_ids, si_ids, inps_mask_cnt, pool_cands, 
                    sel_inps_mask, sampled_in_repres, task):
        '''
        this function is mainly for our modification of original codes
        to adapt the FLAGS.num_sample_acts
        ''' 
        
        if task == 'scan':
    
            
            sel_inps_mask_cnt = list()
            for i in range(len(so_ids)):
                so_id = so_ids[i]
                cnt = inps_mask_cnt[i][so_id]
                #print(cnt)
                #print(inps_mask_cnt[i], so_id)
                assert cnt in [1,2]
                sel_inps_mask_cnt.append(cnt)

            max_len = -1
            #pool_cands: 2-level list bs->len(strucs)
            for cands in pool_cands:
                for cand in cands:
                    cand_str = self.span_decodes[cand]
                    cand_li = cand_str.split(' ')
                    if len(cand_li) > max_len:
                        max_len = len(cand_li)

            cand_repres = list()
            for cands in pool_cands:
                    cands_enc = list()
                    for cand in cands:
                        cand_enc = list()
                        cand_str = self.span_decodes[cand]
                        cand_li = cand_str.split(' ')
                    
                        for tok in cand_li:
                            cand_enc.append(self.vocab.encode_tok(tok))
                        while len(cand_enc) < max_len:
                            cand_enc.append(self.vocab.pad())

                        cands_enc.append(cand_enc)
                    cand_repres.append(cands_enc)        
            cand_repres = torch.tensor(cand_repres, dtype=int).to(_flags.device())
            # cand_repres.shape = [bs, len(strucs), span_len]

            # sel_inps_mask.shape = [bs, seq_len]
            cand_repres_lens = list()
            for i in range(len(si_ids)): # len(..) = bs
                id = int(si_ids[i])
                cand = pool_cands[i][id] # the i-th selected repre;
                cand_str = self.span_decodes[cand]
                cand_li = cand_str.split(' ') # here we suppose that repres are strings
                cand_repres_lens.append(len(cand_li))

            inps_mask = sel_inps_mask.detach().tolist()
            inps_mask_lens = list()
        
            sta_ends = list()
            for i in range(len(inps_mask)): # len(..) = bs
                inp_mask = inps_mask[i] # len(..) = seq_len
                temp_len = 0
                if sel_inps_mask_cnt[i] == 1:
                    for j in range(len(inp_mask)):
                        e = inp_mask[j]
                        if e == 0. : 
                            if temp_len == 0:
                                start = j # start = first masked position 
                            temp_len += 1
                            end = j
                    sta_ends.append((start,end))
                    inps_mask_lens.append(temp_len)
                elif sel_inps_mask_cnt[i] == 2:
                    find_cnt = 0
                    temp_lens = list()
        
                    for j in range(len(inp_mask)):
                        e = inp_mask[j]
                        if e == 0.:
                            if temp_len == 0:
                                if find_cnt == 0:
                                    find_cnt = 1
                                    sta_1 = j
                                elif find_cnt == 1:
                                    find_cnt = 2
                                    sta_2 = j
                            temp_len += 1
                            if find_cnt == 1:
                                end_1 = j
                            elif find_cnt == 2:
                                end_2 = j
                        if e == 1.:
                            if temp_len > 0:
                                temp_lens.append(temp_len)
                                temp_len = 0

                    sta_ends.append((sta_1,end_1,sta_2,end_2))
                    inps_mask_lens.append(sum(temp_lens)/2)

            len_diffs = list()
            for i in range(len(cand_repres_lens)):
                if sel_inps_mask_cnt[i] == 1:
                    len_diffs.append(cand_repres_lens[i]-inps_mask_lens[i])
                elif sel_inps_mask_cnt[i] == 2:
                    len_diffs.append(2*(cand_repres_lens[i]-inps_mask_lens[i]))
        


            sel_in_spans = list()
            for i in range(len(si_ids)):
                si_id = si_ids[i]
                cand = pool_cands[i][si_id]
                cand_str = self.span_decodes[cand]
                sel_in_spans.append(cand_str)

            return cand_repres, cand_repres_lens, len_diffs, sta_ends,  sel_in_spans, sel_inps_mask_cnt
        
        elif task == 'cogs':
            for i in range(len(sampled_in_repres)):
                pass

            repres_lens = list() # to save lens for every repre
            repres = list()
            for i in range(len(si_ids)): # len(..) = bs
                id = si_ids[i]
                repre = sampled_in_repres[i][id] # the i-th selected repre;
                repre = repre.split(' ') # here we suppose that repres are strings
                repres.append(repre)
                repres_lens.append(len(repre))

            inps_mask = sel_inps_mask.detach().tolist()
            inps_mask_lens = list()
            for i in range(len(inps_mask)): # len(..) = bs
                inp_mask = inps_mask[i] # len(..) = seq_len
                temp_len = 0
                for e in inp_mask:
                    if e == 0. : temp_len += 1
                inps_mask_lens.append(temp_len)

            len_diffs = [(repres_lens[i]-inps_mask_lens[i]) 
                                for i in range(len(repres_lens))]

    def gen_aug_utterances (self, inps, sel_out_spans, sel_in_spans, mode=0):
        '''
        this function is leveraged to generate correpesonding a batch of utterances
        usage : when is no need to pass gradient through inps
        mode = 0 : `scan` ; mode = 1 : `cogs`
        '''   
        if mode == 0:
            aug_inps = list()
            bs = len(inps)
            for i in range(bs):
                inp = inps[i]
                # out is a list
                inp_dec = self.vocab.decode(inp)
                inp_str = ' '.join(inp_dec)

                sel_out_span = sel_out_spans[i]
                sel_in_span = sel_in_spans[i]
                aug_inp_str = inp_str.replace(sel_out_span, sel_in_span)
                aug_inp = aug_inp_str.split(' ')
                aug_inp = self.vocab.encode(aug_inp)
                aug_inps.append(aug_inp)
            
            return aug_inps
        pass
    def gen_aug_parses (self, outs, sel_out_spans, sel_in_spans , mode=0):
        '''
        this function is leveraged to generate correpesonding a batch of parses
        usage : there is no need to pass gradient through parses
        mode = 0 : `scan` ; mode = 1 : `cogs`
        '''   
        if mode == 0:
            aug_outs = list()
            bs = len(outs)
            for i in range(bs):
                out = outs[i]
                # out is a list
                out_dec = self.vocab.decode(out)
                out_str = ' '.join(out_dec)

                sel_out_span = self.inp2out[sel_out_spans[i]]
                sel_in_span = self.inp2out[sel_in_spans[i]]
                aug_out_str = out_str.replace(sel_out_span, sel_in_span)
                aug_out = aug_out_str.split(' ')
                aug_out = self.vocab.encode(aug_out)
                aug_outs.append(aug_out)
            
            return aug_outs
        pass



        # derta_lens might be like [3, 10, -2, 0, 11, 3, 1, 0, 1, 1, ...]
        # where 10 means that the seq_len will increase by 10
        # and -3 means that the seq_len will decrease by 3
        # note that we need to consider that the original padding, 
        # e.g., if we have 4 unit padding and we need to increase by 3 
        # then we need not to change.




