# construct aug_parses for augmentor

import random
from absl import app, flags, logging
import numpy as np
FLAGS = flags.FLAGS
neg_inf = -1e3
PAD = 0
flags.DEFINE_integer("fix_num_span", 10, "the fix number of extracted spans in a single datum")
flags.DEFINE_float("p_tree2tree", 0.3, "prob for sub a tree with another tree")
flags.DEFINE_float("p_lex2lex", 0.3, "prob for sub a lex with another lex")
flags.DEFINE_float("p_span2span", 0.1, "prob for sub a span with another span")
flags.DEFINE_float("p_lex2span", 0.3, "prob for sub a lex with a span")


class Span_Extractor_Module():
    '''
    func descrip: 
        1) : to get the spans or span-clusters from an input sentence. 
        2) : to sample span from the off-the-shelf span pools 
            or the temp span candidates constructed from the input sentence.
    '''
    def __init__(self, span_pool, inp_candidates, span_encodes, exchanges, vocab, enc2type=None):
        '''
        [task: cogs]
        
        @span_pool: dict
            --key1: span_type
                --key2: (in_type, out_type)
                    --key3: struc_type

        @inp_candidates: dict
            --key1: inp_idx
                --key2: span_type
                    --key3: struc_type
        
        @span_encodes: dict
            --key1: span->struc_tuple (span_A and span_B are the same ALA they have the same struc)
                --value: a single int
        @exchange: dict
            --key1: encode(span)->encode(struc_tuple)
                --value: a singe int: also the encode of a type of spans

        @enc2type: dict, for `cogs`-type task only;
            --key1: enc(span)
                --value: type = 'span', 'lex', or 'tree'.

        [task: scan]
        
        @span_pool: list,[span_str1, span_str2,...]
        
        @inp_candidates: dict
            --key1: inp_idx

        @span_encodes: dict
            --key1: span's string(span_A and span_B are different)
                --value: a single int
        @exchange: dict
            --key1: encode(span)
                --value: a single int: also the encode of a type of spans
        '''
        self.span_pool = span_pool
        self.inp_cands = inp_candidates
        self.span_enc_dict = span_encodes
        self.exchanges = exchanges
        self.vocab = vocab
        self.enc2type = enc2type

    def get_span_encode(self, span, mode = 0):
        '''
        mode = 0: `scan`-style task, encode span to an int
        mode = 1: `cogs`-style task, encode struc to an int
        '''
        return self.span_enc_dict[span]

    def get_inp_candidates(self, inps, inp_ids, mode=0):
        '''
        inp.type = list, inp.'shape' = [bs, seq_len]
        mode = 0: each unique span as a singe candidate;
        mode = 1: span with the same tag would be treated as a single candidate;
        '''
        batch_cands_encode = list()
        batch_mask_indicat = list()
        inps_mask = list()
        inps_mask_cnt = list()
        inps_spans = list()

        if mode == 0:
            bs = len(inp_ids)
            for i in range(bs):
                inp = inps[i]

                inp_spans = list(self.inp_cands[inp_ids[i]])
                inps_spans.append(inp_spans)

                datum_cands_encode = list()

                for inp_span in inp_spans:
                    inp_span_enc = self.get_span_encode(inp_span, mode = 0)
                    # here we assume that inp_span_enc is a `int`-type data
                    datum_cands_encode.append(inp_span_enc)
                batch_mask_indicat.append(len(datum_cands_encode))
                

                while len(datum_cands_encode) < FLAGS.fix_num_span:
                    datum_cands_encode.append(PAD)

                batch_cands_encode.append(datum_cands_encode)
                '''
                e.g., batch_cands_encode is like : 
                [
                    [e11, e12, e13, ..., e17, PAD, PAD, PAD],
                    [e21, e22, e23, ..., e26, PAD, PAD, PAD, PAD],
                    ...,
                    [e{bs,1}, e{bs,2},..., e{bs,9}, PAD]
                ]
                '''
                inp_mask,inp_mask_cnt = self.get_inp_masks(inp, inp_spans, mode=0)
                inps_mask_cnt.append(inp_mask_cnt)
                inps_mask.append(inp_mask)



            '''
            second: leverage function: `get_inp_masks` to get the mask vector
                indicating which tokens need to be subsed out for each inp.
            '''
            
            return batch_cands_encode, batch_mask_indicat, None, inps_mask, inps_spans, inps_mask_cnt
            
        elif mode == 1:
            '''
            span with the same tag would be treated as a single candidate;
            to do
            '''
            batch_reminder = list()
            bs = len(inp_ids)
            for i in range(bs):
                inp = inps[i]
                inp_spans = self.inp_candidates[inp_ids[i]]
                '''
                @inp_candidates: dict
                --key1: inp_idx
                --key2: span_type
                --key3: struc_type
                '''
                rand_val = random.random()

                while 1:
                    if rand_val < FLAGS.p_tree2tree:
                        if 'tree' not in inp_spans:
                            continue
                        inp_strucs = inp_spans['tree']
                        reminder = 'tree2tree'
                        break

                    elif rand_val < FLAGS.p_lex2lex + FLAGS.p_tree2tree:
                        if 'lex' not in inp_spans:
                            continue
                        inp_strucs = inp_spans['lex']
                        reminder = 'lex2lex'
                        break

                    elif rand_val < FLAGS.p_span2span + FLAGS.p_lex2lex + FLAGS.p_tree2tree:
                        if 'span' not in inp_spans:
                            continue
                        inp_strucs = inp_spans['span']
                        reminder = 'span2span'
                        break

                    else:
                        if 'lex' not in inp_spans:
                            continue
                        inp_strucs = inp_spans['lex']
                        reminder = 'lex2span'
                        break

                batch_reminder.append(reminder)
                # this reminder is to prompt which kind of struc need to be subsed in

                datum_cands_encode = list()
                

                sampled_inp_spans = self.sample_repre(inp_strucs, 1)
                inps_spans.append(sampled_inp_spans)

                # `sampled_inp_spans` here should be like `inp_spans` in the scan-style task
                inp_mask = self.get_inp_masks(inp, sampled_inp_spans, mode=1)
                inps_mask.append(inp_mask)

                for inp_struc in inp_strucs.keys():
                    inp_struc_enc = self.get_span_encode(inp_struc, mode = 0)
                    # here we assume that inp_span_enc is a `int`-type data
                    datum_cands_encode.append(inp_struc_enc)

                batch_mask_indicat.append(FLAGS.fix_num_span - len(inp_struc_enc))
                

                while len(datum_cands_encode) < FLAGS.fix_num_span:
                    datum_cands_encode.append(PAD)
                batch_cands_encode.append(datum_cands_encode)
            
            '''
            to do
            first: leverage function: `sample_repre` to sample the represents
                from each cluster;
            second: leverage function: `get_inp_masks` to get the mask vector
                indicating which tokens need to be subsed out for each inp.
            '''

            return batch_cands_encode, batch_mask_indicat, batch_reminder, inps_mask, inps_spans, None
            pass
    
    def sample_repre(self, cands=None, mode=1):
        '''
        only be used when dealing with `cogs`-style dataset.
        mode = 0: sample the representants from span_bool;
        mode = 1: sample the representants from inp_cands;
        cands are supposed to be like 
        cands: dict
            ---key1: struc_type
                ---value: concrete span(str...)
        return:
            list: representatives for each type of struc
        '''
        pass
    
    def get_inp_masks(self, inp, span_cands, mode =0):
        '''
        get the mask vector indicating which ids of the inp are supposed
        to be subsed out
        e.g., 
        `inp`: <SOS> jump opposite right twice and look left thrice <EOS>
        `span`: 
        {
            jump opposite right, look left thrice, look, jump, right, left
        }
        `masks`:(all of the element are torch.tensors), shape[0] = fix_len_span;
        shape[1] = seq_len
        { 
            [1,0,0,0,1,1,1,1,1,1],
            [1,1,1,1,1,1,0,0,0,1],
            [1,1,1,1,1,1,0,1,1,1],
            [1,0,1,1,1,1,1,1,1,1],
            [1,1,1,0,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,0,1,1],
            PAD,
            PAD,
            ...,
            PAD
        }

        P.S.:
        mode = 0 : `scan`-style
        mode = 1 : `cogs`-style
        '''
        if mode == 0:
            inp_mask = list()
            inp_mask_cnt = list()
            # len(inp) = seq_len, 
            # [<sos>, 3, 21, ..., 17, <eos>, <pad>, <pad>]
            inp_ = list()
            for tok_enc in inp:
                inp_.append(self.vocab.decode_tok(tok_enc))
            inp_str = ' '.join(inp_)
            for cand in span_cands:
                cand_cnt = inp_str.count(cand)
                assert cand_cnt in [1,2]
                inp_mask_cnt.append(cand_cnt)

                tok_num = cand.count(' ')+1
                mask_rep = '<mask>'
                for i in range(tok_num-1):
                    mask_rep = mask_rep+' <mask>'
                inp_str_r = inp_str.replace(cand, mask_rep)
                inp_li = inp_str_r.split(' ')
                mask = list()
                for i in range(len(inp_li)):
                    if inp_li[i] == '<mask>':
                        mask.append(0)
                    else:
                        mask.append(1)
                inp_mask.append(mask)
            
            pad = list()
            for i in range(len(inp)):
                pad.append(1)

            while len(inp_mask_cnt) < FLAGS.fix_num_span:
                inp_mask.append(pad)
                inp_mask_cnt.append(0)
            
            return inp_mask, inp_mask_cnt
                
    def get_cands_from_pool(self, out_spans, reminders=None, mode = 0):
        '''
        out_spans: list, len(out_spans) = bs
        e.g., out_spans = [3, 9, 19, 2, 28, 6, 4, ...]
        where `3(28)` may represent span_encodes['jump around right']
        mode = 0:for `scan`-type task
        mode = 1:for `cogs`-type task 
        return: 
        [
            1:[span_encodes[key1],...,span_encodes[last_key]],
            2:[],
            ...,
            bs:[]
        ]
        ***: we need to additionally record which value we would set to -inf;
        ***: for `cogs`-type task, we additionally need to record the representives;
        reminder: (only for `cogs`-type task) = ['span2span','lex2span',...],len=bs
        '''
        bs = len(out_spans)
        pool_cands=list()
        exchangeables = list()
        all_spans = list()
        if mode == 0:
            for span in self.span_pool:
                # span_pool = list[cand_str1,...]
                all_spans.append(self.get_span_encode(span, 0))
                # [encode(cand_str1),...]
            #sampled_repres = None

        elif mode == 1:
            strucs = dict()
            for key1 in self.span_pool:
                # first-layer key = span_type
                for key2 in self.span_pool[key1]:
                    # second-layer key = (inp_node_type, out_node_type)
                    for key3 in self.span_pool[key1][key2].keys():
                        # third-layer key = struc_type
                        all_spans.append(self.get_span_encode(key3, 1))
                        assert key3 not in strucs
                        # key3 are not supposed to already exist in strucs
                        strucs[key3] = self.span_pool[key1][key2][key3]
                        # a potential bug: is there arranged by order which I set?
            #sampled_repres = self.sample_repre(strucs, 0)
            #note, for each datum in a batch, we should resample repres for it.

        sampled_repres = list()
        # it is supposed to contain bs list, and each one of them are supposed
        # to be a repre_set
        for i in range(bs):
            out_span = out_spans[i]
            exchange = self.exchanges[out_span]
            # e.g. exchange = [3, 5, 8, 14, 41, 124, ...]
            
            if mode == 0:
                temp_exch = list()
                for j in range(len(all_spans)):
                    if all_spans[j] in exchange:
                        temp_exch.append(j)
                exchangeables.append(temp_exch)
                # only those appear in the exchangeable ids won't be set to -inf
                pool_cands.append(all_spans)

            elif mode == 1:
                reminder = reminders[i] # e.g.,'lex2span'
                out_span_type = self.enc2type[out_span] # e.g.,'lex'
                temp_exch = list()
                for j in range(len(all_spans)):
                    if all_spans[j] in exchange:
                        in_span_type = self.enc2type[all_spans[j]]
                        if out_span_type+'2'+in_span_type == reminder:
                            temp_exch.append(j)
                exchangeables.append(temp_exch)
                pool_cands.append(all_spans)
                sampled_repre = self.sample_repre(strucs, 0)
                sampled_repres.append(sampled_repre)

        return pool_cands, exchangeables, sampled_repres



def recombine_parse_scan():
    pass

def recombine_parse_cogs():
    pass