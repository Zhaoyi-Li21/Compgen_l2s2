import flags as _flags

from absl import flags
from collections import namedtuple
import torch
from torch import nn, optim
from torch.optim import lr_scheduler as opt_sched
from torch.nn.utils.clip_grad import clip_grad_norm_
from torchdec import hlog
from model.seq import batch_seqs
from utils import NoamLR
import numpy as np


FLAGS = flags.FLAGS
flags.DEFINE_integer("n_epochs", 512, "number of training epochs")
flags.DEFINE_integer("n_epoch_batches", 32, "batches per epoch")
flags.DEFINE_integer("n_batch", 0, "batch size")
flags.DEFINE_float("lr", 0.001, "learning rate")
flags.DEFINE_float("clip", 1., "gradient clipping")
flags.DEFINE_float("sched_factor", 0.5, "opt scheduler reduce factor")
flags.DEFINE_integer("warmup", 80, "warmup epoch")
flags.DEFINE_integer("update_ratio", 1, "# updates augmentor / # updates seq2seq")
flags.DEFINE_float("temperature", 1., "temperature of gumbel-softmax")
FLAGS = flags.FLAGS

Datum = namedtuple(
    "Datum", 
    "inp out inp_data out_data"
)

def make_batch(samples):
    device = _flags.device()
    seqs = zip(*samples)

    inp, out, *extra = seqs
    #print(inp)
    inps = list(inp)
    #print([len[s] for s in inps])
    lens = torch.tensor([len(s) for s in inps]).to(device)

    inp_data = batch_seqs(inp).to(device)
    out_data = batch_seqs(out).to(device)
    if FLAGS.model_arch == 'lstm':
        return Datum(
            inp, out, inp_data, out_data
        )
    elif FLAGS.model_arch == 'transformer':
        return Datum(inp, out, inp_data, out_data), lens



def warp_make_batch(sample):
    inp_ids = list()
    samples = list()
    for _ in range(FLAGS.n_batch):
        datum, idx = sample()
        samples.append(datum)
        inp_ids.append(idx)
    seqs = zip(*samples)
    inp, out, *extra = seqs
    max_len = max(len(s) for s in inp)
    pad_lens = list()
    for s in inp:
        pad_lens.append(max_len-len(s))
    if FLAGS.model_arch == 'lstm':
        batch = make_batch(samples)
        return batch, inp_ids, pad_lens
    elif FLAGS.model_arch == 'transformer':
        batch, lens = make_batch(samples)
        return batch, lens, inp_ids, pad_lens


def repad_batch(inp, max_pad_num):
    max_len = max(len(s) for s in inp)
    data = np.zeros((max_len + max_pad_num, len(inp)))
    for i, s in enumerate(inp):
        for j in range(len(s)):
            data[j, i] = s[j]
    return torch.LongTensor(data).to(_flags.device())

def assemble_loss(loss_li):
    '''
    considering sample multiple actions for each sentence;
    this function is to assemble the loss for optimize the augmentor
    '''
    act_losses = torch.stack(loss_li, dim=1) # shape = [bs, FLAGS.num_sample_acts]
    max_loss = torch.max(act_losses, 1)[0] # shape = [bs]
    min_loss = torch.min(act_losses, 1)[0] # shape = [bs]
    loss = torch.mean(max_loss - min_loss) * (-1.) # *(-1), max -> encourage; min -> decourage
    print('act_losses:', act_losses[FLAGS.detect_id])
    print('max_loss:', max_loss[FLAGS.detect_id])
    print('min_loss:', min_loss[FLAGS.detect_id])
    return loss


def assemble_loss_rl(loss_li, so_onehots_li, si_onehots_li):
    '''
    reward = \Sigma_{n=1}^{bs} \Sigma_{t=1}^{T} [\pai(a;\theta) * loss(s;a)];
    tot_loss = reward * (-1);
    so_onehots.shape = [bs, 1, num_act1];
    si_onehots.shape = [bs, num_act2];
    '''
    act_losses = torch.stack(loss_li, dim=1).detach() # shape = [bs, FLAGS.num_sample_acts]
    baseline = torch.mean(act_losses, 1) # shape = [bs]
    act_bat_loss_li = list()
    for i in range(FLAGS.num_sample_acts):
        so_onehots = so_onehots_li[i].squeeze(1)
        si_onehots = si_onehots_li[i]
        loss = act_losses[:, i] - baseline # shape = [bs]
        so_onehots = so_onehots.unsqueeze(2) # shape = [bs, num_act1, 1]
        si_onehots = si_onehots.unsqueeze(1) # shape = [bs, 1, num_act2]
        p_action = torch.bmm(so_onehots, si_onehots) # shape = [bs, num_act1, num_act2]
        #print(p_action.shape)
        #print(p_action[0])
        #print(loss.shape, loss)
        temp = torch.ones_like((so_onehots.squeeze(2)))
        loss = loss.unsqueeze(1) # shape = [bs, 1]
        loss = loss.expand_as(temp) # shape = [bs, num_act1]
        loss = loss.unsqueeze(2) # shape = [bs, num_act1, 1]
        loss = loss.expand_as(p_action) # shape = [bs, num_act1, num_act2]
        zeros = torch.zeros_like(p_action)
        loss = torch.where(p_action>0, loss, zeros) # shape = [bs, num_act1, num_act2]
        # elemwise_prod = torch.mm(p_action, loss) # shape = [bs, num_act1, num_act2]
        elemwise_prod = torch.einsum('ijk,ijk->ijk',[p_action,loss]) # shape = [bs, num_act1, num_act2]
        #print(elemwise_prod[0])
        act_bat_loss = torch.sum(elemwise_prod, (1,2)) # shape = [bs]
        act_bat_loss_li.append(act_bat_loss)
    
    act_bat_losses = torch.stack(act_bat_loss_li, dim=1) # shape = [bs, FLAGS.num_sample_acts]
    rl_loss = torch.sum(act_bat_losses, dim=1) # shape = [bs]
    print('act_losses:', act_bat_losses[FLAGS.detect_id])
    print('rl_reward:',rl_loss[FLAGS.detect_id])
    rl_loss = torch.mean(rl_loss) * (-1.)

    return rl_loss


@hlog.fn("train")
def train(dataset, model, augmentor, sample, callback, debugger):
    if not isinstance(model, nn.Module):
        return
    if not isinstance(augmentor, nn.Module):
        return
    if FLAGS.model_arch == 'lstm':
        opt_seq = optim.Adam(model.parameters(), lr=FLAGS.lr)
        if FLAGS.sched_factor < 1:
            sched = opt_sched.ReduceLROnPlateau(opt_seq, mode='max', factor=FLAGS.sched_factor, verbose=True)
    elif FLAGS.model_arch == 'transformer':
        opt_seq = optim.Adam(model.parameters(), lr=FLAGS.lr, betas=(0.9, 0.998))
        sched = NoamLR(opt_seq, FLAGS.dim, warmup_steps=FLAGS.lr_warmup_steps)

    opt_aug = optim.Adam(augmentor.parameters(), lr=FLAGS.lr)



    for i_epoch in hlog.loop("%05d", range(FLAGS.n_epochs)):
        if i_epoch < FLAGS.warmup: # warming up epochs
            model.train()
            epoch_loss = 0
            for i_batch in range(FLAGS.n_epoch_batches):
                #sched.step()
                opt_seq.zero_grad()
                if FLAGS.model_arch == 'lstm':
                    datum, inp_ids, _ = warp_make_batch(sample)
                    loss = model(datum.inp_data, datum.out_data)
                elif FLAGS.model_arch == 'transformer':
                    samples = list()
                    for _ in range(FLAGS.n_batch):
                        datum, _ = sample()
                        samples.append(datum)

                    datum, lens = make_batch(samples)
                    loss = model(datum.inp_data, datum.out_data, lens)
                # we temporarily stop here. 2023/3/12    
                loss.backward()

                gnorm = clip_grad_norm_(model.parameters(), FLAGS.clip)
                if not np.isfinite(gnorm.cpu()):
                    raise Exception("=====GOT NAN=====")
                
                opt_seq.step()
                epoch_loss += loss.item()
                if FLAGS.model_arch == 'transformer':
                    sched.step()

            epoch_loss /= FLAGS.n_epoch_batches
            hlog.value("warmup loss", epoch_loss)
            val_score = callback(i_epoch)
            if FLAGS.sched_factor < 1 and FLAGS.model_arch=='lstm':
                sched.step(val_score)

        else: # training for both augmentor and seq model
            debugger.draw_epoch(i_epoch)

            augmentor.train()
            model.train()
            epoch_loss = 0
            aug_loss = 0
            for i_batch in range(FLAGS.n_epoch_batches):
                '''
                optimizing augmentor --- maximizing step
                '''
                augmentor.train()
                model.train()

                debugger.draw_batch(i_batch)
                bat_aug_loss = 0

                for __ in range(FLAGS.update_ratio):
                    
                    opt_aug.zero_grad()
                    opt_seq.zero_grad()
                    if FLAGS.model_arch == 'lstm':
                        datum, inp_ids, pad_lens = warp_make_batch(sample)
                    elif FLAGS.model_arch == 'transformer':
                        datum, lens, inp_ids, pad_lens = warp_make_batch(sample)
                    # batch are sharing among different actions sampling

                    debugger.detector_inp(datum.inp_data)
                    ret = augmentor.gen_sub_info(datum.inp_data, inp_ids, debugger, 
                                    temperature = FLAGS.temperature, sample_multi_acts=True)


                    share_cand_repres, share_inp_mask, so_one_hots_li, si_one_hots_li, \
                        cand_repres_lens_li, len_diffs_li, sta_ends_li, \
                            sel_out_spans_li, sel_in_spans_li, inp_mask_cnt_li = ret
                    
                    loss_li = list()

                    for t in range(FLAGS.num_sample_acts):

                        '''
                        get specific info from the info-list
                        '''
                        cand_repres = share_cand_repres
                        inp_mask = share_inp_mask

                        so_one_hots_ = so_one_hots_li[t]
                        si_one_hots = si_one_hots_li[t]
                        cand_repres_lens = cand_repres_lens_li[t]
                        len_diffs = len_diffs_li[t]
                        sta_ends = sta_ends_li[t]
                        sel_out_spans = sel_out_spans_li[t]
                        sel_in_spans = sel_in_spans_li[t]
                        inp_mask_cnt = inp_mask_cnt_li[t]

                        if FLAGS.model_arch == 'transformer':
                            len_diffs = torch.tensor(len_diffs).to(_flags.device())
                            _lens = lens + len_diffs
                        
                        aug_inp = augmentor.gen_aug_utterances(datum.inp, sel_out_spans, sel_in_spans)
                        aug_inp_data = batch_seqs(aug_inp).to(_flags.device())
                        aug_out = augmentor.gen_aug_parses(datum.out, sel_out_spans, sel_in_spans)
                        aug_out_data = batch_seqs(aug_out).to(_flags.device())

                        # debug for 'are the output are correctly produced ?' 
                        #debugger.detector_sub_behavior(inp_data, inp_mask, aug_out_data, 
                                                # si_one_hots, cand_repres)
                        '''
                        # old version code, assemble the aug_inp with embedding. 
                        _loss = model.aug_forward(
                            inp_data, aug_out_data,
                            cand_repres, inp_mask,
                            si_one_hots, cand_repres_lens,
                            len_diffs, sta_ends, inp_mask_cnt, padding,
                             manual_mode='eval'
                        )
                        '''
                        model.eval()
                        if FLAGS.model_arch == 'lstm':
                            _loss = model(aug_inp_data, aug_out_data, batch_sum=False)
                        elif FLAGS.model_arch == 'transformer':
                            _loss = model(aug_inp_data, aug_out_data, _lens, batch_sum=False)
                        # note that here _loss.shape = [bs] (without reduction)

                        loss_li.append(_loss)
                        # debug, check gradients
                        # debugger.get_grad(loss, si_one_hots, halt=False)
                        # debugger.get_grad(loss, so_one_hots_)

                    #loss = assemble_loss(loss_li) 
                    loss = assemble_loss_rl(loss_li, so_one_hots_li, si_one_hots_li)

                    loss.backward()
                    clip_grad_norm_(augmentor.parameters(), FLAGS.aug_clip)
                    opt_aug.step()
                    bat_aug_loss += loss.item()
                    # hlog.value("aug_update", loss.item())

                aug_loss += bat_aug_loss/FLAGS.update_ratio
                
                '''
                optimizing parser --- minimizing step
                '''
                augmentor.eval()
                model.train()

                opt_aug.zero_grad()
                opt_seq.zero_grad()
                if FLAGS.model_arch == 'lstm':
                    datum, inp_ids, pad_lens = warp_make_batch(sample)
                elif FLAGS.model_arch == 'transformer':
                    datum, lens, inp_ids, pad_lens = warp_make_batch(sample)
                
                debugger.detector_inp(datum.inp_data)
                with torch.no_grad():
                    ret = augmentor.gen_sub_info(datum.inp_data, inp_ids, debugger,
                                        temperature = FLAGS.temperature, sample_multi_acts = False)

                cand_repres, inp_mask, so_one_hots_, si_one_hots, cand_repres_lens,\
                     len_diffs, sta_ends, sel_out_spans, sel_in_spans, inp_mask_cnt = ret
                

                if FLAGS.model_arch == 'transformer':
                    len_diffs = torch.tensor(len_diffs).to(_flags.device())
                    _lens = lens + len_diffs
                    debugger.detector_lens_transformer(_lens)
                aug_inp = augmentor.gen_aug_utterances(datum.inp, sel_out_spans, sel_in_spans)
                aug_inp_data = batch_seqs(aug_inp).to(_flags.device())
    
                aug_out = augmentor.gen_aug_parses(datum.out, sel_out_spans, sel_in_spans)
                aug_out_data = batch_seqs(aug_out).to(_flags.device())

                #debugger.detector_sub_behavior(inp_data, inp_mask, aug_out_data, 
                                        #si_one_hots, cand_repres)
                '''
                loss = model.aug_forward(
                    inp_data, aug_out_data,
                    cand_repres, inp_mask,
                    si_one_hots, cand_repres_lens,
                    len_diffs, sta_ends, inp_mask_cnt, padding, 
                    manual_mode='train'
                )
                '''
                if FLAGS.model_arch == 'lstm':
                    loss = model(aug_inp_data, aug_out_data)
                elif FLAGS.model_arch == 'transformer':
                    loss = model(aug_inp_data, aug_out_data, _lens)
                print(loss)
                loss.backward()
                gnorm = clip_grad_norm_(model.parameters(), FLAGS.clip)
                if not np.isfinite(gnorm.cpu()):
                    raise Exception("=====GOT NAN=====")
                opt_seq.step()
                epoch_loss += loss.item()
                if FLAGS.model_arch == 'transformer':
                    # according to MET-PRIM, we update lr every batch
                    sched.step()
            aug_loss /= FLAGS.n_epoch_batches
            epoch_loss /= FLAGS.n_epoch_batches
            hlog.value("aug_loss", aug_loss)
            hlog.value("epoch_loss", epoch_loss)
            val_score = callback(i_epoch)
            if FLAGS.sched_factor < 1 and FLAGS.model_arch == 'lstm':
                sched.step(val_score)
