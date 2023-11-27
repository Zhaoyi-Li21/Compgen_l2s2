from .builder import OneShotDataset

from absl import flags
import os
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string("scan_data_dir", None, "data directory")
flags.DEFINE_string("scan_split", "add_prim_split", "data split")
flags.DEFINE_string("scan_file", "addprim_jump", "data file")

flags.DEFINE_boolean("use_dev", False,"whether to use existing devfile or not")
flags.DEFINE_boolean("val_test", False, "use a subpart of test set as val set")
TRAIN = "tasks_train_%s.txt"
TEST = "tasks_test_%s.txt"

DEV = "tasks_dev_%s.txt"

REF = "../tasks.txt"

class ScanDataset(OneShotDataset):
    def __init__(self, **kwargs):
        train = self.load_split(TRAIN % FLAGS.scan_file)
        np.random.shuffle(train)
        
        if FLAGS.use_dev == True:
            val = self.load_split(DEV % FLAGS.scan_file)
        else: 
            val = train[:64 * 10]
        train = train[64 * 10:]
        if FLAGS.TEST:
            test = self.load_split(TEST % FLAGS.scan_file)
        else:
            test = val

        ref_data = self.load_split(REF)
        self.ref = {tuple(k): tuple(v) for k, v in ref_data}

        super().__init__(
            train, val, test, 
            #holdout={("jump",), ("I_JUMP",)},
            **kwargs
        )

    def load_split(self, split_file):
        data = []
        with open(os.path.join(FLAGS.scan_data_dir, FLAGS.scan_split, split_file)) as fh:
            for line in fh:
                toks = line.strip().split()[1:]
                split = toks.index("OUT:")
                inp = toks[:split]
                out = toks[split+1:]
                data.append((inp, out))
        return data

    def score(self, pred, ref_out, ref_inp):
        if FLAGS.invert:
            # NACS eval
            pred_str = tuple(self.vocab.decode(pred))
            inp_str = tuple(self.vocab.decode(ref_inp))
            if pred_str not in self.ref:
                return 0
            return 1 if self.ref[pred_str] == inp_str else 0
        else:
            return 1 if pred == ref_out else 0

    def construct_inp_cands(self, structure):

        def is_overlap(li, subli):
            prev = -1
            for i in range(len(li)):
                if li[i:i+len(subli)] == subli:
                    if i <= prev:
                        return True
                    else:
                        prev = i + len(subli) -1
            return False

        self.inp_cands = dict()

        for idx in range(len(self.train_utts)):
            inp, out = self.train_utts[idx]
            # inp is a list
            inp_str = ' '.join(inp)
            cands = set()
            for pat in structure.span_pool:
                if pat in inp_str:
                    cands.add(pat)
            
            _cands = set()
            for cand in cands:
                cover_flag = 0
                for cover in structure.covered[cand]:
                    if cover in inp_str:
                        cover_flag = 1
                        break
                if cover_flag == 1:
                    # delete cand from cands
                    continue

                out_span = structure.inp2out[cand]
                out_span_li = out_span.split(' ')
                #out_str = ' '.join(out)

                if is_overlap(out, out_span_li) == True:
                    continue

                inp_tokens = cand.split(' ')
                
                sta_flag = 0
                end_flag = 0
                for ids in structure.tagging:
                    if inp_tokens[0] in structure.tagging[ids]:
                        sta = ids
                        sta_flag = 1
                    if inp_tokens[-1] in structure.tagging[ids]:
                        end = ids
                        end_flag = 1
                        
                if (sta_flag == 0) or (end_flag == 0):
                    continue

                _cands.add(cand)

            self.inp_cands[idx] = _cands


    def construct_inp_cands_waug(self, structure):

        def is_overlap(li, subli):
            prev = -1
            for i in range(len(li)):
                if list(li[i:i+len(subli)]) == subli:
                    if i <= prev:
                        return True
                    else:
                        prev = i + len(subli) -1
            return False

        self.inp_cands = dict()

        for idx in range(len(self.train_utts)):
            inp, out = self.train_utts[idx]
            # inp is a list
            inp_str = ' '.join(inp)
            cands = set()
            for pat in structure.span_pool:
                if pat in inp_str:
                    cands.add(pat)
            
            _cands = set()
            for cand in cands:
                cover_flag = 0
                for cover in structure.covered[cand]:
                    if cover in inp_str:
                        cover_flag = 1
                        break
                if cover_flag == 1:
                    # delete cand from cands
                    continue

                out_span = structure.inp2out[cand]
                out_span_li = out_span.split(' ')
                #out_str = ' '.join(out)


                if is_overlap(out, out_span_li) == True:
                    continue

                inp_tokens = cand.split(' ')
                
                sta_flag = 0
                end_flag = 0
                for ids in structure.tagging:
                    if inp_tokens[0] in structure.tagging[ids]:
                        sta = ids
                        sta_flag = 1
                    if inp_tokens[-1] in structure.tagging[ids]:
                        end = ids
                        end_flag = 1
                        
                if (sta_flag == 0) or (end_flag == 0):
                    continue

                _cands.add(cand)
   
            self.inp_cands[idx] = _cands
        
        for _idx in range(len(self.aug_utts)):
            inp, out = self.aug_utts[_idx]
            idx = len(self.train_utts) + _idx # append 
            # inp is a list
            inp_str = ' '.join(inp)
            cands = set()
            for pat in structure.span_pool:
                if pat in inp_str:
                    cands.add(pat)
            
            _cands = set()
            for cand in cands:
                cover_flag = 0
                for cover in structure.covered[cand]:
                    if cover in inp_str:
                        cover_flag = 1
                        break
                if cover_flag == 1:
                    # delete cand from cands
                    continue

                out_span = structure.inp2out[cand]
                out_span_li = out_span.split(' ')
                #out_str = ' '.join(out)


                if is_overlap(out, out_span_li) == True:
                    continue

                inp_tokens = cand.split(' ')
                
                sta_flag = 0
                end_flag = 0
                for ids in structure.tagging:
                    if inp_tokens[0] in structure.tagging[ids]:
                        sta = ids
                        sta_flag = 1
                    if inp_tokens[-1] in structure.tagging[ids]:
                        end = ids
                        end_flag = 1
                        
                if (sta_flag == 0) or (end_flag == 0):
                    continue

                _cands.add(cand)

            self.inp_cands[idx] = _cands
            
    


            
            