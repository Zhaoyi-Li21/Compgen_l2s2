import torch
import random
from absl import flags

flags.DEFINE_integer("detect_id", 10, "")
FLAGS = flags.FLAGS
class Debugger():
    def __init__(self, vocab, map_enc2str, span_pool):
        self.vocab = vocab
        self.map_enc2str = map_enc2str
        self.span_pool = span_pool

    def draw_epoch(self, epoch):
        print('------ %d th epoch ------'%epoch)

    def draw_batch(self, batch):
        print('------ %d th batch ------'%batch)
    
    def detector_inp(self, inp):
        det_idx = FLAGS.detect_id
        det_inp = torch.transpose(inp, 1, 0)[det_idx].long().tolist()
        det_inp = self.vocab.decode(det_inp)
        print('inp:',det_inp)

    def detector_sub_behavior(self,inp, inp_mask, out, 
                        subs_in_onehots, cand_repres):
        det_idx = FLAGS.detect_id
        det_inp = torch.transpose(inp, 1, 0)[det_idx].long().tolist()
        det_inp = self.vocab.decode(det_inp)
        det_inp_mask = inp_mask[det_idx].tolist()
        det_out = torch.transpose(out, 1, 0)[det_idx].long().tolist()
        det_out = self.vocab.decode(det_out)
        temp = torch.argmax(subs_in_onehots,dim=1)
        temp = temp.tolist()[det_idx]
        det_cand = cand_repres[det_idx,temp].tolist()
        det_cand = self.vocab.decode(det_cand)
        print('inp:',det_inp)
        print('mask:',det_inp_mask)
        print('cand:',det_cand)
        print('out:',det_out)
        pass


    def detector_logits_onehot(self, out_logits, out_onehots, in_logits, in_onehots,
                                bat_cands_enc, batch_mask_indicat, exchangeable,
                                pool_cands, sample_multi_acts=False):
        det_idx = FLAGS.detect_id
        det_out_logits = out_logits[det_idx]
        det_out_onehot = out_onehots[det_idx]
        det_in_logits = in_logits[det_idx]
        det_in_onehot = in_onehots[det_idx]
        det_cands = pool_cands[det_idx]

        #print('out_logits:',det_out_logits)
        #print('out_onehot:',det_out_onehot)
        det_batch_mask_indicat = int(batch_mask_indicat[det_idx])
        if sample_multi_acts == False:
            print('<--- subout span candidates : --->')
            for i in range(det_batch_mask_indicat):
                cand_enc = int(bat_cands_enc[det_idx][i])
                print(self.map_enc2str[cand_enc],'--prob:%f'%det_out_logits[i].float())
            print('<--- selected out span : --->')
            temp = torch.argmax(det_out_onehot)
            sel_cand_enc = int(bat_cands_enc[det_idx][temp])
            print(self.map_enc2str[sel_cand_enc])
        
            print('<--- subin span candidates : --->')
            #print(len(self.span_pool))
            #print(len(det_out_logits.tolist()))
            for i in exchangeable[det_idx]:
                print(self.map_enc2str[det_cands[i]], '--prob:%f'%det_in_logits[i].float())
            print('<--- selected in span : --->')
            temp = torch.argmax(det_in_onehot)
            print(self.map_enc2str[det_cands[temp]])
            print('<--- over --->')
        else:
            print('<--- selected out-in span : --->')
            temp = torch.argmax(det_out_onehot)
            sel_cand_enc = int(bat_cands_enc[det_idx][temp])
            print(self.map_enc2str[sel_cand_enc])

            temp = torch.argmax(det_in_onehot)
            print(self.map_enc2str[det_cands[temp]])


        #print('in_logits:',det_in_logits)
        #print('in_onehot:',det_in_onehot)
    
    def get_var_grad(self, loss, bat_var, halt=True):
        det_idx = FLAGS.detect_id
        print(bat_var[det_idx])
        grad = torch.autograd.grad(loss, bat_var, 
                            retain_graph=True, allow_unused=True)[0]
        print(grad[det_idx])
        if halt == True:
            raise Exception('halted after fetching grad, debug!')

    def check_aug_para_grad(self, vec, augmentor, halt=True):

        temp_loss = torch.sum(vec)
        grads = torch.autograd.grad(temp_loss, augmentor.parameters(), allow_unused=True)
        cnt = 0
        for name, params in augmentor.named_parameters():
            print(name)
            print(grads[cnt].shape)
            print(grads[cnt])
            cnt += 1

        if halt==True:
            raise Exception('halted after fetching grad for parameters, debug!')
    
    def check_vector_orient(self, vec):
        # vec.shape = [bs, vec_dim]
        #vec_dim = vec.shape[1]
        #det_idx = FLAGS.detect_id
        #det_vec = vec[det_idx].unsqueeze(0)
        if random.random() < 0.2:
            det_vec = vec
            anchor = torch.ones_like(det_vec)
            cos_sim = torch.cosine_similarity(det_vec, anchor)
            print('<--- cos_sim_between_inp_and_anchor --->')
            print(cos_sim)
    
    def detector_lens_transformer(self, lens):
        det_idx = FLAGS.detect_id
        print('check lens for transformer input :', lens[det_idx])
        