from .builder import OneShotDataset

from absl import flags
import os
import numpy as np
import json
FLAGS = flags.FLAGS

flags.DEFINE_string("text2funql_dir", None, "location of text2funql(q.g., geoquery) data")
flags.DEFINE_string("text2funql_split", '', "test for iid(''), len('len'), or template('template') generalization capacity")
flags.DEFINE_string("text2funql_orig", "", "use ori dataset('orig_') or not('')")
class Text2funqlDataset(OneShotDataset):
    def __init__(self, **kwargs):
        return super().__init__(
            self.load_split("train"),
            self.load_split("dev"),
            self.load_split("test"),
            # parameter : "test"
            # test for i.i.d test, len test, or template test
            **kwargs
        )

    def load_split(self, split):
        data = []
        with open(os.path.join(FLAGS.text2funql_dir, split + '_'+ FLAGS.text2funql_orig + FLAGS.text2funql_split + '.json')) as reader:
            count = 0
            for line in reader:
                count += 1
                if count == 427:
                    m = 1
                temp = json.loads(line)
                inp = temp["question"]
                out = temp["program"]
                # inp, out, _ = line.strip().split("\t")
                # if 'new mexico' in out, then we set them as a single token
                _out_list = out.split()
                out_list = list()
                cnt = 0
                for token in _out_list:

                    if token == "(" or token == ")":
                        # if we do not take "(" or ")" into consideration
                        cnt += 1 
                        continue

                    if "'" in token:
                        if token[0] == "'" and token[-1] == "'":
                            # ‘utah’
                            out_list.append(token[0])
                            out_list.append(token[1:-1])
                            out_list.append(token[-1])
                        elif token[0] == "'" and token[-1] == ",":
                            # 'rochester',
                            out_list.append(token[0])
                            out_list.append(token[1:-2])
                            if token[-2] == "'":
                                out_list.append(token[-2])
                            else:
                                print("ff")
                            out_list.append(token[-1])
                        elif token[0] == "'" and token[-1] != ",":
                            # 'new york'
                            out_list.append(token[0])
                            out_list.append(token[1:])
                        elif token[0] != "'" and token[-1] == "'":
                            # 'new york'
                            out_list.append(token[:-1])
                            out_list.append(token[-1])
                        
                        elif token[0] != "'" and token[-1] == ",":
                            # 'new orleans',
                            out_list.append(token[:-2])
                            if token[-2] == "'":
                                out_list.append(token[-2])
                            else:
                                print("ff")
                            out_list.append(token[-1])
                        else :
                            # no this case
                            print(token)
                            print("fff")
                    else:
                        out_list.append(token)
                #print(cnt)
                # proc_out_list is a list make 'mount mckinley' as a single lex
                proc_out_list = list()
                flag = -1 #flag = 0 : start; flag = 1 : end
                adhesive = list()
                for i in range(len(out_list)):
                    if out_list[i] != "'":
                        if flag == 0:
                            # now we have already a start
                            adhesive.append(out_list[i])
                        else:
                            # now this token not in any [start, end]
                            proc_out_list.append(out_list[i])
                    else:
                        # not append ' token, we set ' as a part of 'new york' for example.
                        # proc_out_list.append(out_list[i])

                        if flag != 0:
                            # means now we have a end or not even start
                            flag = 0
                            # start
                        elif flag == 0:
                            # means now we have already a start
                            token = ' '.join(adhesive)
                            token = "'"+token+"'"
                            proc_out_list.append(token)
                            adhesive = list()
                            flag = 1
                            # end      
                data.append((
                    tuple(inp.split()),
                    tuple(proc_out_list)
                ))
        return data

    def score(self, pred, ref_out, ref_inp):
        return 1 if pred == ref_out else 0