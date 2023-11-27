#!/usr/bin/env python

import flags as _flags
from model.parser import GeneratorModel
from model.transformer import TransformerEncDec
from l2s2_trainer import train, make_batch, Datum
from l2s2_debug import Debugger
import l2s2_setting

from data.scan import ScanDataset
from data.semparse import SemparseDataset
from data.cogs import CogsDataset
from data.colors import ColorsDataset

from span_utils.structure import Structure
from model.augmentor import AugmentModel


from absl import app, flags, logging
import json
import numpy as np
import os
import torch
from torchdec import hlog
import random

FLAGS = flags.FLAGS
flags.DEFINE_string("augment", None, "file with composed data for augmentation")
flags.DEFINE_float("aug_ratio", 0, "fraction of samples to draw from augmentation")
flags.DEFINE_boolean("invert", False, "swap input/output")
flags.DEFINE_boolean("test_curve", True, "test in place")
flags.DEFINE_integer("gpu",0,"gpu-id")
flags.DEFINE_string("task", "scan", "is this is a scan-style task or a cogs-style task")
flags.DEFINE_string("align", "", "the file-path of the extracted alignment")

flags.DEFINE_string("model_arch", "lstm", "lstm or transformer")
flags.DEFINE_string("transformer_config", "3layer", "transformer config")
flags.DEFINE_integer("beam_size", 1, "beam size")
flags.DEFINE_integer("dim", 512, 'transformer dimension')
flags.DEFINE_integer("lr_warmup_steps", 4000,"noam warmup_steps")
flags.DEFINE_float("aug_clip", 1., "augment model gradient clipping")

def get_dataset(**kwargs):
    if FLAGS.dataset == "semparse":
        return SemparseDataset(**kwargs)
    if FLAGS.dataset == "scan":
        return ScanDataset(**kwargs)
    if FLAGS.dataset == "cogs":
        return CogsDataset(**kwargs)
    if FLAGS.dataset == "colors":
        return ColorsDataset(**kwargs)
    assert False, "unknown dataset %s" % FLAGS.dataset

def main(argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    hlog.flags()
    
    if FLAGS.augment is not None:
        with open(FLAGS.augment) as fh:
            aug_data = json.load(fh)
    else:
        aug_data = []

    dataset = get_dataset(aug_data=aug_data, invert=FLAGS.invert)
    print(len(dataset.val_utts))
    
    structure = Structure()
    
    if FLAGS.task == 'scan':
        # some data structures
        structure.construct_from_file_scan(FLAGS.align)
        structure.construct_span_pool('scan') # structure.span_pool
        structure.construct_tagging('scan') # structure.tagging (clusters)
        #dataset.construct_inp_cands(structure) # dataset.inp_cands : dict
        dataset.construct_inp_cands_waug(structure)
        structure.construct_exchanges('scan')

    elif FLAGS.task == 'cogs':
        '''
        to do
        '''
        pass
    
    debugger = Debugger(dataset.vocab, structure.map_enc2str, structure.span_pool)
    
    
    if FLAGS.model_arch == 'lstm':
        seq2seq_model = GeneratorModel(
            dataset.vocab,
            copy=True,
            self_attention=False
        ).to(_flags.device())
    elif FLAGS.model_arch == 'transformer':
        # FLAGS.model_arch == 'transformer'
        seq2seq_model = TransformerEncDec(dataset.vocab, dataset.vocab, 
                                dataset.max_len_y, 
                                FLAGS.transformer_config).to(_flags.device())
        FLAGS.dim = seq2seq_model.output_dim
        pass
    else:
        raise Exception("Invalid 'model_arch'('lstm' or 'transformer') parameter!")

    augmentor = AugmentModel(
        dataset.vocab,
        FLAGS.aug_encoder_n_embed,
        FLAGS.aug_encoder_n_hidden,
        FLAGS.aug_encoder_n_layer,
        FLAGS.aug_encoder_dropout,
        structure.span_pool,
        dataset.inp_cands,
        structure.map_str2enc,
        structure.exchanges,
        None,
        FLAGS.aug_g_n_embed,
        FLAGS.aug_f_n_embed,
        structure.map_enc2str,
        structure.inp2out
    ).to(_flags.device())


    def sample():
        # return dataset.sample_train_wid()
        # return ((inp, out), idx)
        return dataset.sample_train_wid_waug(FLAGS.aug_ratio)

    def callback(i_epoch):

        seq2seq_model.eval()
        final = i_epoch == FLAGS.n_epochs - 1
        with hlog.task("eval_train", timer=False):
            train_data = [dataset.sample_train() for _ in range(1000)]
            evaluate(seq2seq_model, train_data, dataset)
        with hlog.task("eval_val", timer=False):
            val_data = dataset.get_val()
            val_acc = evaluate(seq2seq_model, val_data, dataset, vis=final, beam=final)
        with hlog.task("eval_val_test", timer=False):
            val_test_data = dataset.get_val_test()
            val_test_acc = evaluate(seq2seq_model, val_test_data, dataset, vis=False, beam=final)
        if FLAGS.TEST and (final or FLAGS.test_curve):
            with hlog.task("eval_test", timer=False):
                test_data = dataset.get_test()
                evaluate(seq2seq_model, test_data, dataset, vis=True, beam=final)
        
        return val_acc

    train(dataset, seq2seq_model, augmentor, sample, callback, debugger)

def evaluate(model, data, dataset, vis=False, beam=False):
    correct = 0
    total = 0
    for i in range(0, len(data), FLAGS.n_batch):
        if FLAGS.model_arch == 'lstm':
            batch = make_batch(data[i:i+FLAGS.n_batch])
            preds, _ = model.sample(batch.inp_data, greedy=True, beam=beam)
        elif FLAGS.model_arch == 'transformer':
            batch, lens = make_batch(data[i:i+FLAGS.n_batch])
            preds, _ = model.sample(inp=batch.inp_data, lens=lens,
                                    temp=1.0, max_len=model.MAXLEN,
                                    beam_size=FLAGS.beam_size, # in align with previous works, stop beam_size
                                    calc_score=False
                                    )
            
        for j in range(len(preds)):
            if FLAGS.model_arch == 'lstm':
                score_here = dataset.score(preds[j], batch.out[j], batch.inp[j])
            elif FLAGS.model_arch == 'transformer':
                score_here = dataset.score(preds[j], batch.out[j][1:], batch.inp[j])
            '''
            we only output the wrong examples
            '''
            if vis and score_here == 0:
                with hlog.task(str(total)):
                    hlog.value("input", " ".join(dataset.vocab.decode(batch.inp[j])))
                    hlog.value("pred", " ".join(dataset.vocab.decode(preds[j])))
                    hlog.value("gold", " ".join(dataset.vocab.decode(batch.out[j])))
                    hlog.value("corr", score_here)
                    hlog.log("")
            total += 1
            correct += score_here
    acc = 1. * correct / total
    hlog.value("acc", acc)
    return acc

if __name__ == "__main__":

    app.run(main)
