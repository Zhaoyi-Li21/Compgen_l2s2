from absl import flags
from collections import Counter, defaultdict
import heapq
import itertools as it
import numpy as np
from torchdec import hlog
from torchdec.vocab import Vocab
import pygtrie
import random

FLAGS = flags.FLAGS
flags.DEFINE_boolean("dedup", False, "deduplicate training examples")
flags.DEFINE_integer("wug_limit", None, "wug limit")
flags.DEFINE_integer("wug_size", 4, "wug size")
flags.DEFINE_integer("wug_count", 2, "number of wugs to insert")
flags.DEFINE_boolean("compute_adjacency", False, "do compositionality")
flags.DEFINE_boolean("use_trie", False, "store indices in tries")
flags.DEFINE_integer("max_comp_len", None, "maximum seq length to attempt")
flags.DEFINE_integer("max_adjacencies", None, "blah")
flags.DEFINE_enum("template_sim", "none", ["none", "window"], "similarity function")
flags.DEFINE_integer("sim_window_size", 2, "similarity window size")
flags.DEFINE_integer("variants", 2, "number of different args to swap in")

variants = 5
wug_template = "WUG%d"
def _wugs():
    return [wug_template % i for i in range(FLAGS.wug_count)]

sep = "##"

class DefaultTrie(object):
    def __init__(self, initializer):
        self.initializer = initializer
        self.trie = pygtrie.Trie()

    def __setitem__(self, key, item):
        self.trie[key] = item

    def __getitem__(self, key):
        if key not in self.trie:
            self.trie[key] = self.initializer()
        return self.trie[key]

    def __len__(self):
        return len(self.trie)

    def __iter__(self):
        return iter(self.trie)

    def keys(self):
        return self.trie.keys()

    def items(self):
        return self.trie.items()

def t_subseq(subseq, seq):
    for i in range(len(seq)-len(subseq)+1):
        if seq[i:i+len(subseq)] == subseq:
            return True
    return False

def t_replace(old_subseq, new_subseq, seq):
    for i in range(len(seq)-len(old_subseq)+1):
        if seq[i:i+len(old_subseq)] == old_subseq:
            return seq[:i] + new_subseq + seq[i+len(old_subseq):]
    return seq

def t_replace_all(old_subseq, new_subseq, seq):
    assert not t_subseq(old_subseq, new_subseq)
    before = None
    after = seq
    while after != before:
        before = after
        after = t_replace(old_subseq, new_subseq, before)
    return after

def t_split(sep, seq, vocab):
    index = seq.index(sep)
    inp, out = seq[:index], seq[index+1:]
    assert inp[0] == vocab.sos()
    assert vocab.eos() not in inp
    inp = inp + (vocab.eos(),)
    assert out[-1] == vocab.eos()
    assert vocab.sos() not in out
    out = (vocab.sos(),) + out
    return inp, out

class OneShotDataset(object):
    def __init__(
            self,
            train_utts,
            val_utts,
            test_utts,
            aug_data=(),
            invert=False,
    ):
        # update max_len_x, max_len_y
        max_len_x, max_len_y = 0, 0
        

        vocab = Vocab()
        for i in range(FLAGS.wug_count):
            vocab.add(wug_template % i)
        vocab.add(sep)
        for utts in (train_utts, val_utts, test_utts):
            for inp, out in utts:
                max_len_x = max(len(inp), max_len_x)
                max_len_y = max(len(out), max_len_y)

                for seq in (inp, out):
                    for word in seq:
                        vocab.add(word)

        aug_utts = [(tuple(d["inp"]), tuple(d["out"])) for d in aug_data]

        for inp,out in aug_utts:
            max_len_x = max(len(inp), max_len_x)
            max_len_y = max(len(out), max_len_y)
            for seq in (inp, out):
                for word in seq:
                    vocab.add(word)
     
        if FLAGS.dedup:
            train_utts = [(tuple(i), tuple(o)) for i, o in train_utts]
            train_utts = sorted(list(set(train_utts)))
        hlog.value("train", len(train_utts))
        hlog.value("aug", len(aug_utts))

        if invert:
            train_utts = [(o, i) for i, o in train_utts]
            aug_utts = [(o, i) for i, o in aug_utts]
            val_utts = [(o, i) for i, o in val_utts]
            test_utts = [(o, i) for i, o in test_utts]

        self.vocab = vocab
        self.sep = sep
        self.train_utts = train_utts
        self.aug_utts = aug_utts
        self.val_utts = val_utts
        self.test_utts = test_utts

        self.max_len_x = max_len_x
        self.max_len_y = max_len_y
        
        if FLAGS.compute_adjacency:
            self._compute_adjacency(train_utts)

        '''
        invert table building
        '''
        self.inv_table = dict()
        
    def novel(self, inp=None, out=None):
        if inp is None:
            out = tuple(out)
            return not any(o == out for i, o in self.train_utts)
        if out is None:
            inp = tuple(inp)
            return not any(i == inp for i, o in self.train_utts)
        return (tuple(inp), tuple(out)) not in self.train_utts

    def realize(self, seq, names):
        dec = list(self.vocab.decode(seq))
        out = []
        used = set()
        for w in dec:
            if w in names:
                used.add(w)
                out += list(names[w])
            else:
                out.append(w)
        return tuple(out), used

    
    def _compute_adjacency(self, utts):
        counts = Counter()
        for utt in utts:
            inp, out = utt
            for seq in (inp, out):
                enc = self.vocab.encode(seq)[1:-1]
                for span in range(1, FLAGS.wug_size+1):
                    for i in range(len(enc)+1-span):
                        counts[tuple(enc[i:i+span])] += 1
        if FLAGS.wug_limit is None:
            keep_args = set(counts.keys())
        else:
            keep_args = set([c for c, n in counts.items() if n <= FLAGS.wug_limit])

        def make_store(initializer):
            if FLAGS.use_trie:
                return DefaultTrie(initializer)
            else:
                return defaultdict(initializer)

        def compute_templ_sim(templates):
            wugs = {self.vocab[w] for w in _wugs()}
            size = FLAGS.sim_window_size
            def wug_indices(templ):
                return tuple(i for i, t in enumerate(templ) if t in wugs)
            templ_to_sig = make_store(set)
            sig_to_templ = make_store(set)

            for templ in templates:
                indices = wug_indices(templ)
                sig = tuple(templ[i-size:i+size+1] for i in indices)
                templ_to_sig[templ].add(sig)
                sig_to_templ[sig].add(templ)

            templ_sim = make_store(set)
            for templ1 in templates:
                for sig in templ_to_sig[templ1]:
                    for templ2 in sig_to_templ[sig]:
                        templ_sim[templ1].add(templ2)
            return templ_sim

        def enumerate_templates():
            for i, utt in enumerate(utts):
                inp, out = utt
                seq = inp + (sep,) + out
                if FLAGS.max_comp_len is not None and len(seq) >= FLAGS.max_comp_len:
                    continue
                if i % 1000 == 0:
                    hlog.value("template_utt", "%d/%d" % (i, len(utts)))
                for generic in self._make_generic(seq, keep_args):
                    yield generic, utt

        arg_to_templ = make_store(set)
        templ_to_arg = make_store(set)
        templ_to_templ = make_store(set)
        #sim_templ = FuzzyIndex(tfidf=True)
        #templ_to_orig = defaultdict(set)
        for (templ, args), orig in enumerate_templates():
            arg_to_templ[args].add(templ)
            templ_to_arg[templ].add(args)
            #sim_templ.put(templ, args)
            #templ_to_orig[templ].add(orig)

        if FLAGS.template_sim == "window":
            templ_sim = compute_templ_sim(templ_to_arg.keys())
        else:
            templ_sim = {t: set([t]) for t in templ_to_arg.keys()}

        multiplicity = make_store(lambda: 0)
        for i_arg, args1 in enumerate(arg_to_templ.keys()):
            if i_arg % 10000 == 0:
                hlog.value("template_arg", "%d/%d" % (i_arg, len(arg_to_templ)))
            for templ1 in arg_to_templ[args1]:
                multiplicity[templ1] += 1
                c = 0
                for templ2_pre in arg_to_templ[args1]:
                    for templ2 in templ_sim[templ2_pre]:
                        if templ1 == templ2:
                            continue
                        #if (templ1, templ2) in templ_to_templ:
                        #    continue
                        templ_to_templ[templ2].add(templ1)
                        c += 1
                        if (
                            FLAGS.max_adjacencies is not None 
                            and c >= FLAGS.max_adjacencies
                        ):
                            break

        self.templ_to_arg = templ_to_arg
        #self.arg_to_templ = arg_to_templ
        self.templ_to_templ = templ_to_templ
        self.multiplicity = multiplicity

        comp_pairs = []
        for templ1 in self.templ_to_templ:
            if self.multiplicity[templ1] <= 1:
                continue
            for templ2 in self.templ_to_templ[templ1]:
                comp_pairs.append((templ1, templ2))
        self.comp_pairs = sorted(comp_pairs)
        self.templates = sorted(self.templ_to_arg.keys())

    def compute_similarity(self, sim_model):
        wugs = [self.vocab[w] for w in _wugs()]
        for templ in self.templates:
            idx = [templ.index(w) for w in wugs if w in templ]

    def _make_generic(self, seq, keep):
        enc_seq = tuple(self.vocab.encode(seq))
        wugs = [self.vocab[w] for w in _wugs()]
        out = self._make_generic_helper(enc_seq, keep, 0, 0, (), wugs)
        return out

  
    def _make_generic_helper(self, seq, keep, begin, i_wug, used_args, wugs):
        for span in range(1, FLAGS.wug_size+1):
            for i in range(begin, len(seq)+1-span):
                arg = seq[i:i+span]
                arg_enc = arg
                templ = t_replace_all(arg, (wugs[i_wug],), seq)
                #templ = seq[:i] + (wugs[i_wug],) + seq[i+span:]
                templ_enc = templ
                if self.vocab[sep] in arg:
                    continue
                #arg_enc = tuple(self.vocab.encode(arg)[1:-1])
                if arg_enc not in keep:
                    continue
                if any(len(set(uarg) & set(arg_enc)) > 0 for uarg in used_args):
                    continue
                next_args = used_args + (arg_enc,)
                assert self.vocab[sep] in templ_enc
                yield (
                    #tuple(self.vocab.encode(templ)),
                    templ_enc,
                    next_args,
                )
                if i_wug+1 < FLAGS.wug_count:
                    for rest in self._make_generic_helper(
                        templ, keep, i+1, i_wug+1, next_args, wugs
                    ):
                        yield rest

    def split(self, templ):
        return t_split(self.vocab[self.sep], templ, self.vocab)

    def join(self, templ):
        inp, out = templ
        return inp[:-1] + (self.vocab[self.sep],) + out[1:]

    def overlap(self, arg, ref_args):
        if arg in ref_args:
            return True
        if all(len(set(arg) & set(a)) > 0 for a in ref_args):
            return True
        return False

    def enumerate_comp(self):
        for templ2 in self.templ_to_templ:
            args2 = self.templ_to_arg[templ2]
            args = [
                arg 
                for templ1 in self.templ_to_templ[templ2]
                if self.multiplicity[templ1] > 1
                for arg in self.templ_to_arg[templ1]
                if not self.overlap(arg, args2)
            ]
            if len(args) == 0:
                continue
            np.random.shuffle(args)
            #args = it.islice(it.chain.from_iterable(it.repeat(args)), variants)
            args = args[:FLAGS.variants]
            for arg in args:
                dec_arg = [self.vocab.decode(a) for a in arg]
                names = dict(zip(_wugs(), dec_arg))
                yield self.split(templ2), names

    def enumerate_freq(self):
        for templ, count in self.multiplicity.items():
            if count <= 1:
                continue
            args = list(self.templ_to_arg[templ])
            np.random.shuffle(args)
            for arg in args[:FLAGS.variants]:
                dec_arg = [self.vocab.decode(a) for a in arg]
                names = dict(zip(_wugs(), dec_arg))
                yield self.split(templ), names

    def _sample_comp(self):
        i = np.random.randint(len(self.comp_pairs))
        return self.comp_pairs[i]

    def sample_comp_train(self):
        templ1, templ2 = self._sample_comp()
        return self.split(templ1), self.split(templ2)

    def sample_ctx_train(self):
        templ = self.templates[np.random.randint(len(self.templates))]
        args = sorted(self.templ_to_arg[templ])
        arg = args[np.random.randint(len(args))]
        wugs = [self.vocab[w] for w in _wugs()]
        named_arg = list(zip(arg, wugs))
        arg_part, name = named_arg[np.random.randint(len(named_arg))]

        del_arg_part = (self.vocab.sos(),) + arg_part + (self.vocab.eos(),)
        return templ, del_arg_part, templ.index(name)

    def _sample(self, utts, index=None):
        if index is None:
            index = np.random.randint(len(utts))
        inp, out = utts[index]
        inp = self.vocab.encode(inp)
        out = self.vocab.encode(out)
        return inp, out

    def sample_train(self, aug_ratio=0.):
        if np.random.random() < aug_ratio:
            return self._sample(self.aug_utts)
        else:
            return self._sample(self.train_utts)

    def sample_train_wid(self):
        index = np.random.randint(len(self.train_utts))
        inp, out = self.train_utts[index]
        inp = self.vocab.encode(inp)
        out = self.vocab.encode(out)
        return (inp, out), index
    
    def sample_train_wid_waug(self, aug_ratio=0.):
        if np.random.random() < aug_ratio:
            index = np.random.randint(len(self.aug_utts))
            inp, out = self.aug_utts[index]
            inp = self.vocab.encode(inp)
            out = self.vocab.encode(out)
            return (inp, out), len(self.train_utts)+index 
        else:
            index = np.random.randint(len(self.train_utts))
            inp, out = self.train_utts[index]
            inp = self.vocab.encode(inp)
            out = self.vocab.encode(out)
            return (inp, out), index


    def sample_train_ablation(self, structure, aug_ratio=0., task='scan'):
        if np.random.random() < aug_ratio:
            index = np.random.randint(len(self.aug_utts))
            inp, out = self.aug_utts[index]
            inp_cands = self.inp_cands[index+len(self.train_utts)]

        else:
            index = np.random.randint(len(self.train_utts))
            inp, out = self.train_utts[index]
            inp_cands = self.inp_cands[index]

        if task == 'scan':
            subout_span = random.choice(list(inp_cands))
            subout_span_enc = structure.encode(subout_span)
            exchangeables = structure.exchanges[subout_span_enc]
            subin_span_enc = random.choice(list(exchangeables))
            subin_span = structure.decode(subin_span_enc)

            inp_str = ' '.join(inp)
            out_str = ' '.join(out)
            aug_inp_str = inp_str.replace(subout_span, subin_span)
            aug_out_str = out_str.replace(structure.inp2out[subout_span], structure.inp2out[subin_span])

            aug_inp = self.vocab.encode(aug_inp_str.split(' '))
            aug_out = self.vocab.encode(aug_out_str.split(' '))
            '''
            print(inp_str)
            print(aug_inp_str)
            print(out_str)
            print(aug_out_str)
            print(inp_cands)
            print(subin_span)
            print(subout_span)
            '''
            return aug_inp, aug_out



    def sample_swap(self):
        # only for scan, an additional simple augment 
        index = np.random.randint(len(self.train_utts))
        inp, out = self.train_utts[index]
        swap_inp = []
        swap_out = []
        if 'left' in inp and 'right' in inp:
            for i in range(len(inp)):
                if inp[i] == 'left':
                    swap_inp.append('right')
                elif inp[i] == 'right':
                    swap_inp.append('left')
                else:
                    swap_inp.append(inp[i])
            
            for j in range(len(out)):
                if out[j] == 'I_TURN_LEFT':
                    swap_out.append('I_TURN_RIGHT')
                elif out[j] == 'I_TURN_RIGHT':
                    swap_out.append('I_TURN_LEFT')
                else:
                    swap_out.append(out[j])
        inp = self.vocab.encode(inp)
        out = self.vocab.encode(out)
        if len(swap_inp) > 0:
            swap_inp = self.vocab.encode(swap_inp)
            swap_out = self.vocab.encode(swap_out)
            return (inp, out), (swap_inp,swap_out)
        else:
            return (inp,out), None
         
    def sample_train_conlex(self, lexicon):
        '''
        sample-aim : inp-sentence at least contains one lex belongs to lexicon
        '''
        while(1):
            index = np.random.randint(len(self.train_utts))
            inp, out = self.train_utts[index]
            inp = self.vocab.encode(inp)
            res = list(set(inp) & set(lexicon))
            if len(res) == 0:
                continue
            else:
                break
        out = self.vocab.encode(out)
        return inp, out
                
    def sample_train_substr(self, substr):
        while(1):
            index = np.random.randint(len(self.train_utts))
            inp, out = self.train_utts[index]
            inpstr = ' '.join(inp)
            if substr not in inpstr:
                continue
            else:
                break
        inp = self.vocab.encode(inp)
        out = self.vocab.encode(out)
        return inp, out
    
    def build_inverted_table(self, lexicon):
        '''
        build inverted_table with lexicon 
        '''
        decode_lexicon = list()
        for lex in lexicon:
            decode_lexicon.append(self.vocab.decode_singlex(lex))

        for i in range(len(self.train_utts)):
            inp, _ = self.train_utts[i]
            for lex in inp:
                if lex in decode_lexicon:
                    if lex not in self.inv_table.keys():
                        self.inv_table[lex] = list()
                        self.inv_table[lex].append(i)
                    else:
                        self.inv_table[lex].append(i)

    def sample_train_with_lex(self, lex):
        '''
        use inverted table to accelerate, btw, lex here are suppo to follow a string(decode-form)
        '''
        index = random.choice(self.inv_table[lex])
        inp, out = self.train_utts[index]
        inp = self.vocab.encode(inp)
        out = self.vocab.encode(out)
        return inp, out
    
    def locate_lex_in_sent(self, sent, lex, dataset):
        '''
        notification: sent is a list(encode-form); lex is a string(decode-form)
        '''
        sent = self.vocab.decode(sent)
        cand = list()
        for i in range(len(sent)):
            if sent[i] == lex:
                
                if lex == 'left' and dataset == 'scan':
                    '''
                    just for scan task hhh
                    '''
                    if sent[i-1] != 'around':
                        continue
                
                cand.append(i)
        return random.choice(cand)
        
    def get_one_shots(self, lexs):
        one_shots = list()
        for i in range(0, len(self.train_utts)-1):
            inp, _ = self.train_utts[i]
            for lex in lexs:
                if lex in inp:
                    one_shots.append(i)
                    break
        return one_shots

    def sample_train_oneshots(self, one_shots):
        idx = random.choice(one_shots)
        inp,out = self.train_utts[idx]
        inp = self.vocab.encode(inp)
        out = self.vocab.encode(out)
        return inp, out   
    '''
    for test
    '''
    def _sample_train(self, aug_ratio=0.):
        if np.random.random() < aug_ratio:
            return self._sample(self.aug_utts),1
        else:
            return self._sample(self.train_utts),0
    
    

    '''
    for test
    '''
    def get_train(self):
        return [
            self._sample(self.train_utts, i) for i in range(len(self.train_utts))
        ]

    def get_val(self):
        return [
            self._sample(self.val_utts, i) for i in range(len(self.val_utts))
        ]

    def get_test(self):
        return [
            self._sample(self.test_utts, i) for i in range(len(self.test_utts))
        ]
    def get_val_test(self):
        return [
            self._sample(self.test_utts) for i in range(10*64)
        ]
