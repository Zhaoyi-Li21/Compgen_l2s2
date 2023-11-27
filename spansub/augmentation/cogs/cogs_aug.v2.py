'''
v1: modifications:
@1: treat some of the subtrees as spans
    e.g., `the cake beside the rod`
    criterion: look if this is a single-chain(e.g., `cake <- beside <- rod`)
@2: about the extraction of the spans
    #1: `span` requires in-(head) and out-(tail) are different words.
    (e.g., `Emma valued` is not a span, for in-head and out-tail are both `valued`)
    #2: if no tail -> @1
@3: after finishing @1 and @2, we need to prohibit the exchange between `subtree` & `lex`
@4: now, our extracted spans are also required to be single-chain structure.
substitable rule:
(1): span <-> lex
(2): span <-> span
(3): lex <-> lex
(4): tree <-> tree
'''

'''
v2: modifications:
augmentation code of this version realizes the encoding on distinct substructures
Here we concatenate the In-order traverse order AND parents of in-order travers order 
as the encoding of  
detailed augmentation process:
given a sentence, firstly extract its all kinds of structure-categories.
Then secondly, we randomly select one of these kinds of structure-categories and a specific 
example of this selected category as the structure-cate which will be substituted out soon.
Thirdly, we need to identify the span-cate of the selected structure (e.g., a span, a lex or a tree?)
Fuorthly, select all of the possible categories of span into one pool
(e.g., span(struc-cate1,2,3...) + lex(struc-cate1,2,3...)), and randomly select one kind of structure
to substitute in soon.

@1: after extracting a span, I also get its encoding;
@2: modify the augment function, randomly substitute according to different encoding categories.
'''


import json
import random
import sys
import copy
sys.path.append(r"/data2/home/zhaoyi/compsub")
from data.COGS.seq_tag_gen import LabeledExample, reconstruct_target
from augs.src.cogs.cogs_utils import encode_frag
'''
frpath = '/data2/home/zhaoyi/compsub/augs/data/cogs/seqtags/train_seqtag.json'
aug_frpath = '/data2/home/zhaoyi/compsub/augs/data/cogs/seqtags/aug_cogs_seqtag.4.json'
'''
frpath ='/data2/home/zhaoyi/compsub/augs/data/cogs/seqtags/aug_cogs_seqtag.4.json'
aug_frpath = '/data2/home/zhaoyi/compsub/augs/data/cogs/seqtags/train_seqtag.json'

cogs_ir_set = list()
aug_set = list()
with open(frpath, "r") as fr:
    for line in fr:
        cogs_ir_set.append(json.loads(line.rstrip(';\n')))

with open(aug_frpath, "r") as fr:
    for line in fr:
        aug_set.append(json.loads(line.rstrip(';\n')))

min_span_length = 3
min_tree_length = 3
'''
extracted component
'''

class fragment():
    def __init__(self, span, entrance, exit, category, index, lc, rc):
        self.span = span
        self.ent = entrance
        self.ext = exit
        self.category = category
        self.idx = index
        self.lc = lc
        self.rc = rc

def subcompon_extract(datum):
    '''
    add description for this function
    '''
    if datum["tokens"] == ['Liam', 'meant', 'that', 'Sophia', 'rolled', 'a', 'teacher', 'on', 'a', 'seat', '.']:
        print(111)
    childs = dict()
    parents = datum["parent"]
    root = -1
    # constructing the tree
    nodes = set()
    for i in range(len(parents)):
        if parents[i] != -1:
            nodes.add(i)
            nodes.add(parents[i])
            if parents[i] not in childs.keys():
                childs[parents[i]] = list()
            childs[parents[i]].append(i)
            if parents[parents[i]] == -1:
                root = parents[i]
    # primitive
    if len(nodes) == 0:
        if datum["distribution"] == "primitive":
            # return prim_flag, subcompons_list
            return True, None, [0]
        else:
            print(datum["distribution"])
            return False, None, []
            #raise Exception("current datum is not a primitive, but do not have a parent-node either.")

    # non-primitive
    def between(v, sta, end):
        if v >= sta and v <= end:
            return True
        else:
            return False

    nodes = sorted(list(nodes))# inorder traversal seq
    subcomps = list()
    for sta_ in range(len(nodes)):
        in_edge = {-1:set()}
        # key = entrance_node, value = list of another nodes (of the edges)
        out_edge = {-1:set()}
        for end_ in range(sta_, len(nodes)):
            sta = nodes[sta_]
            end = nodes[end_]
            # [sta, end] is considering interval
            '''
            # pre-update for in_edge and out_edge:
            in_edge_keys = list(in_edge.keys())
            if in_edge_keys[0] != -1:
                key = in_edge_keys[0]
                ant_nodes = in_edge[key]
                if end in ant_nodes:
                    ant_nodes.remove(end)
                    if len(ant_nodes) == 0:
                        in_edge = {-1:set()}

            out_edge_keys = list(out_edge.keys())
            if out_edge_keys[0] != -1:
                key = out_edge_keys[0]
                ant_nodes = out_edge[key]
                if end in ant_nodes:
                    ant_nodes.remove(end)
                    if len(ant_nodes) == 0:
                        out_edge = {-1:set()}
            '''
            in_edge_keys = list(in_edge.keys())
            if in_edge_keys[0] != -1:
                for key in in_edge_keys:
                    ant_nodes = in_edge[key]
                    if end in ant_nodes:
                        ant_nodes.remove(end)
                        if len(ant_nodes) == 0:
                            if len(in_edge.keys()) > 1:
                                in_edge.pop(key)
                            else:
                                in_edge = {-1:set()}

            out_edge_keys = list(out_edge.keys())
            if out_edge_keys[0] != -1:
                for key in out_edge_keys:
                    ant_nodes = out_edge[key]
                    if end in ant_nodes:
                        ant_nodes.remove(end)
                        if len(ant_nodes) == 0:
                            if len(out_edge.keys()) > 1:
                                out_edge.pop(key)
                            else:
                                out_edge = {-1:set()}


            jump_flag = False

            if len(in_edge.keys()) > 1 or len(out_edge.keys()) > 1:
                jump_flag = True

            if between(parents[end], sta, end) == False:
                # means this new-add node bring some new in-edges
                in_edge_keys = list(in_edge.keys())
                if in_edge_keys[0] != -1:
                    # which means already an existing in-node
                    # push this {end:parents[end]} in the dictionary
                    in_edge[end] = set()
                    in_edge[end].add(parents[end])
                    jump_flag = True
                    # jump this end but still need to update out_edge
                else:
                    # which means no existing in-node
                    in_edge = {end:set([parents[end]])}
            
            
            temp = set()
            temp_ = set()
            if end in childs.keys():
                for child in childs[end]:
                    if between(child, sta, end) == False:
                        # means this child (not covered by the current set) 
                        # is brought with this new-add node 'end'
                        out_edge_keys = list(out_edge.keys())
                        if out_edge_keys[0] != -1:
                            # which means already an existing out-node

                            #push the new out-node in the dictionary
                            temp_.add(child)

                            jump_flag = True
                            # break
                            # jump this end
                        else:
                            temp.add(child)

            if jump_flag == True:
                #push the new out-node in the dictionary
                if len(temp_) > 0:
                    out_edge[end] = temp_
                elif len(temp) > 0:
                    out_edge = {end:temp}
                # jump this end
                continue
            else:
                if len(temp) > 0:
                    # means the out-node mustbe the end
                    # otherwise the out-node stays unchanged
                    out_edge = {end:temp}
            
            # tips: we need to tag the extracted-spans : "tree","span","lex"
            # tag priority : lex > tree > span
            # which means if a span is both a lex and a tree, we only need to tag it 'lex'

            # consider child distribute:(lc,rc),1--exist; 0--not exist
            out_edge_keys = list(out_edge.keys())
            in_edge_keys = list(in_edge.keys())
            if out_edge_keys[0] == -1:
                c_flag = (0, 0)
            else:
                out_node = out_edge_keys[0]
                lc = 0
                rc = 0
                for c_node in out_edge[out_node]:
                    if c_node < sta:
                        lc = 1
                    elif c_node > end:
                        rc = 1
                    else:
                        raise Exception("current child of outnode in span is between sta and end !")
                c_flag = (lc, rc)

            if sta == end:
                subcomps.append((sta, end, sta, end, 'lex', c_flag))
                if out_edge_keys[0] == -1:
                    # considering do the comparsion-exp for 'SUBS'
                    subcomps.append((sta, end, in_edge_keys[0], -1, 'tree_', c_flag))
            else:
                if out_edge_keys[0] == -1:
                    '''
                    modification, see if this `tree` is a single-chain structure?
                                --yes: label it `span`
                                --no: label it `tree`
                                (always label it `tree_`, for `tree_` is used in `SUBS`)
                    '''
                    is_tree = False
                    prev = in_edge_keys[0]
                    temp = prev
                    sub_nodes = list()
                    for node in nodes:
                        if node >= sta and node <= end:
                            sub_nodes.append(node)
                    closed = set()
                    closed.add(prev)
                    while len(closed) < len(sub_nodes):
                        find_flag = False
                        for node in nodes:
                            if parents[node] == prev:
                                if find_flag == False:
                                    find_flag = True
                                    if node < prev:
                                        is_tree = True
                                        break
                                    if node in sub_nodes:
                                        closed.add(node)
                                    temp = node
                                else:
                                    is_tree = True
                                    break
                        assert find_flag == True
                        # this program split is supposed to find at least one child for each prev
                        if is_tree == True:
                            break
                        prev = temp
                    if is_tree == True:
                        # this means no out_edge-> a tree
                        subcomps.append((sta, end, in_edge_keys[0], -1, 'tree', c_flag))
                    else:
                        assert in_edge_keys[0] != prev
                        # in common cases, head and tail are not supposed to share one token
                        # this means no out_edge, and meanwhile a single-chain structure
                        subcomps.append((sta, end, in_edge_keys[0], prev, 'span', c_flag))
                    '''
                    modification over
                    '''
                    # considering do the comparsion-exp for 'SUBS'
                    subcomps.append((sta, end, in_edge_keys[0], -1, 'tree_', c_flag))
                    # btw, we only care about the right-most (or the left-most) node in the subsequence for a tree
                    # as a potential out-node
                else:
                    # this means a normal span(not a 'lex' nor a 'tree')
                    if in_edge_keys[0] == out_edge_keys[0]:
                        continue
                    '''
                    here we constraint the span are single-chain structure...
                    '''
                    is_span = True
                    prev = in_edge_keys[0]
                    temp = prev
                    sub_nodes = list()
                    for node in nodes:
                        if node >= sta and node <= end:
                            sub_nodes.append(node)
                    closed = set()
                    closed.add(prev)
                    while len(closed) < len(sub_nodes):
                        find_flag = False
                        for node in nodes:
                            if parents[node] == prev:
                                if find_flag == False:
                                    find_flag = True
                                    if node < prev:
                                        is_span = False
                                        break
                                    if node in sub_nodes:
                                        closed.add(node)
                                    temp = node
                                else:
                                    is_span = False
                                    break
                        assert find_flag == True
                        # this program split is supposed to find at least one child for each prev
                        if is_span == False:
                            break
                        prev = temp
                    if is_span == False:
                        continue

                    if in_edge_keys[0] == out_edge_keys[0]:
                        pass
                    else:
                        subcomps.append((sta, end, in_edge_keys[0], out_edge_keys[0], 'span', c_flag))
            #(interval_start, interval_end, in_node, out_node, frag_type)

    return False, subcomps, nodes # false means not a primitive
    

class span():
    def __init__(self, start, end):
        self.start = start
        self.end = end

class root():
    def __init__(self, entrance, exit):
        self.entrance = entrance
        self.exit = exit

def sub(datum_a, out_span, nodes_A, root_A, datum_b, in_span, nodes_B, root_B):
    '''
    this function realizes the augmentation with the original datum('ori_datum') and substitution.
    sub_out & sub_in are continuous fragments of datum input index list.
    e.g., A(ori_datum) = ["The", "child", "appreciated", "that", "Emma", "valued", "that", "a", "cookie", "was", "slid", "."]
    B = ["Chloe", "was", "handed", "the", "raisin", "on", "the", "table", "by", "Emma", "."]
    B - DAG: [ 0 -> 2 -> 4 -> 5 -> 7 ]  ||| A - DAG:               4->   8->
                    |                                                |     |
                    -> 9 ]                               [ 1 -> 2 -> 5 -> 10 ]
    sub_out for A : *.start = *.end = 1; (in our rule, which means [0(the),1])
    sub_in for B : *.start = 4; *.end = 7; (in our rule, which means [3(the),4,5,6,7])
    (it is a pretty complicated description), let us list some examples as follows:
    ex@1: sub_span for A : [4,5] -> "Emma valued", but 2 is the parent of 5, so we have to use (2, 5]
    (which is [3,4,5]->"that Emma valued") as a entriety.
    ex@2: sub_span for A : [1,1] -> "child", [1,1] only cover one node in the DAG, 
    so we only check CNOUN or not , if CNOUN, attend to determiner of it.

    ret format: (key is the transformation of parent-row)
    [0(B3), 1(B4), 2(B5), 3(B6), 4(B7), 5(A2), 6(A3), 7(A4), 8(A5), 9(A6), 10(A7), 11(A8), 12(A9), 13(A10), 14(A11)]
    aug - DAG:                    [A4->  A8->          ||
                                      |     |          ||
                        [B4 -> A2 -> A5 -> A10]        ||
                         |                             ||
                         -> B5 -> B7]                  ||
    '''
    
    datum_A = copy.deepcopy(datum_a)
    datum_B = copy.deepcopy(datum_b)
    def len_span(span):
        return span.end - span.start

    # in our augment policy, we only allow a longer span substitutes a shorted span.
    # (this encourages the productivity.) 
    # if len(in_span) >= len(out_span): exception:'Paula' -> 'the cat'..
    aug_datum = dict()
    keys = ["tokens", "parent", "role", "category", "noun_type", "verb_name", "distribution"]
    for key in keys:
        if key not in aug_datum.keys():
            aug_datum[key] = list()
        if key != "parent":
            # we need to carefully consider "parent" row
            if key == "distribution":
                aug_datum[key] = "in_distribution"
            elif key == "role":
                # take care! 
                '''
                if datum_B["category"][root_B.entrance] != datum_A["category"][root_A.entrance]:
                    pass
                else:
                    datum_B["role"][root_B.entrance] = datum_A["role"][root_A.entrance]

                if datum_B["category"][root_B.exit] != datum_A["category"][root_A.exit]:
                    pass
                else:
                    datum_B["role"][root_B.exit] = datum_A["role"][root_A.exit]
                '''
                '''
                tips: seems that we are not supposed to constraint that
                root_A.exit => root_B.exit (in role level)
                '''
                '''
                #original code
                if root_A.entrance != -1:
                    datum_B["role"][root_B.entrance] = datum_A["role"][root_A.entrance]
                if root_A.exit != -1:
                    datum_B["role"][root_B.exit] = datum_A["role"][root_A.exit]
                '''
                #newly code
                if root_A.entrance != -1:
                    datum_B["role"][root_B.entrance] = datum_A["role"][root_A.entrance]

                in_part = datum_B[key][in_span.start : in_span.end+1]
                out_part_1 = datum_A[key][:out_span.start]
                out_part_2 = datum_A[key][out_span.end+1:]
                aug_datum[key] = out_part_1 + in_part + out_part_2
            elif key == "tokens":
                # change uppercase & lowercase
                if out_span.start == 0:
                    if datum_B[key][in_span.start] == 'a':
                        datum_B[key][in_span.start] = 'A'
                    elif datum_B[key][in_span.start] == 'the':
                        datum_B[key][in_span.start] = 'The'
                else:
                    if datum_B[key][in_span.start] == 'A':
                        datum_B[key][in_span.start] = 'a'
                    elif datum_B[key][in_span.start] == 'The':
                        datum_B[key][in_span.start] = 'the'
                
                in_part = datum_B[key][in_span.start : in_span.end+1]
                out_part_1 = datum_A[key][:out_span.start]
                out_part_2 = datum_A[key][out_span.end+1:]
                aug_datum[key] = out_part_1 + in_part + out_part_2

            else:
                # we just need to simply concatenate 
                in_part = datum_B[key][in_span.start : in_span.end+1]
                out_part_1 = datum_A[key][:out_span.start]
                out_part_2 = datum_A[key][out_span.end+1:]
                aug_datum[key] = out_part_1 + in_part + out_part_2
    # now we are going to deal with the "parent"
    # for it is clear that all of the outside node 
    # either pointed to ( be pointed by ) the start node or the end node in the subout.[first-order-traverse]

    # firstly we adjust the in_span's "parent" info("parent" in the in_span)
    for node in nodes_B:
        if node >= in_span.start and node <= in_span.end:
            if datum_B["parent"][node] >= in_span.start and datum_B["parent"][node] <= in_span.end:
                datum_B["parent"][node] = datum_B["parent"][node] - in_span.start + out_span.start
          
    # and then we adjust the template's "parent" info
    for node in nodes_A:
        if node < out_span.start or node > out_span.end:
            if datum_A["parent"][node] < out_span.start:
                # do not need to change
                pass
            elif datum_A["parent"][node] > out_span.end:
                datum_A["parent"][node] = datum_A["parent"][node] - len_span(out_span) + len_span(in_span)
            else:
                '''
                seems there exists a bug, we should only check == root_A.exit ?
                because a out_span are supposed to have only one entrance 
                and one exit(for those nodes not in the out_span), 
                and now we are checking other nodes' parents, 
                the only possible is that some nodes' parent == root_A.exit
                '''
                '''
                # original codes
                if datum_A["parent"][node] == root_A.entrance:
                    datum_A["parent"][node] = root_B.entrance - in_span.start + out_span.start
                elif datum_A["parent"][node] == root_A.exit:
                    datum_A["parent"][node] = root_B.exit - in_span.start + out_span.start
                '''
                # newly codes
                if datum_A["parent"][node] == root_A.exit:
                    datum_A["parent"][node] = root_B.exit - in_span.start + out_span.start
                
        # considering the both of the entrance point's and the exit point's parent
        elif node == root_A.entrance:
            if datum_A["parent"][node] < out_span.start:
                datum_B["parent"][root_B.entrance] = datum_A["parent"][root_A.entrance]
            elif datum_A["parent"][node] > out_span.end:
                datum_B["parent"][root_B.entrance] = datum_A["parent"][root_A.entrance] - len_span(out_span) + len_span(in_span)
        elif node == root_A.exit:
            if datum_A["parent"][node] < out_span.start:
                datum_B["parent"][root_B.exit] = datum_A["parent"][root_A.exit]
            elif datum_A["parent"][node] > out_span.end:
                datum_B["parent"][root_B.exit] = datum_A["parent"][root_A.exit] - len_span(out_span) + len_span(in_span)
    
    in_part = datum_B["parent"][in_span.start : in_span.end+1]
    out_part_1 = datum_A["parent"][:out_span.start]
    out_part_2 = datum_A["parent"][out_span.end+1:]
    aug_datum["parent"] = out_part_1 + in_part + out_part_2
    if aug_datum["tokens"] == ["A", "lamb", "gave", "the", "lawyer", "the", "cake", "beside", "the", "rod", 
                                 "beside", "a", "torch", "."]:
        print(111)
    if aug_datum["tokens"] == ['The', 'girl', 'was', 'given', 'a', 'table',
                                    'on', 'a', 'stool', 'on', 'the', 'table', '.']:
        print(111)   
    
    if aug_datum["tokens"] == ['Noah', 'froze', 'Liam', 'beside', 'a', 'sword', '.']:
        print(111)
    return aug_datum

def proc_span(ori_span, noun_type):
    '''
    this function mainly process span to cover the noun-determiner:'a' and 'the'.
    here we note that, we do not need to consider conj word 'that'.
    '''  
    if noun_type[ori_span.start] != "":
            # this means this one node represent a CNOUN(with a NOUN_determiner decorating it)
        proced_span = span(start=ori_span.start-1, end=ori_span.end)
    else:
            # this means this one node represent a PNOUN, a VERB or a PERP
        proced_span = span(start=ori_span.start, end=ori_span.end)

    return proced_span

def cates_transform(datum, node):

    ori_cate = datum["category"][node]
    if ori_cate != "":
        if ori_cate == "CNOUN" or ori_cate == "PNOUN":
            return "NOUN"
        elif ori_cate == "VERB":
            if datum["distribution"] == "primitive":
                return "xcomp"
            return "VERB"
        else:
            raise Exception("unknown category %s"%(ori_cate))
    else:
        ori_role = datum["role"][node]
        if ori_role == "PREP":
            return "PREP"
        elif ori_role == "xcomp":
            return "xcomp"
        else:
            raise Exception("unknown role %s"%(ori_role))


def augment(aug_num):
    '''
    this function recieves the intermediate representation format training set,
    and realize augment them with different mannual policies.
    here for cogs task, we always use the longer subsequences to substitute shorter ones
    to encourage systemactically generalizing to longer sequences.
    '''
    # the following are tips:
    # tips@1: one-node span should only be CNOUN/PNOUN/PERP/VERB,(xcomp is not in the consideration)
    # tips@2: we use larger part (or equal-large part, which means subtree, span, or lex) to substitute smaller part(lex)
    # tips@3: we should enable multiply-times substitute, which encourge deeper-recursion
    # the following are detailed policies:
    # in our method : 30% (larger)span->span; 30% lex->lex; 20% tree->lex; 20% span->lex
    aug_list = list()
    r1 = 0.3 # corresponding to [0, 0.2] tree->tree
    # debugging
    #r1 = 1.
    r2 = 0.3 # corresponding to [0.2, 0.5] lex->lex
    r3 = 0.1 # corresponding to [0.5, 0.7] span->span
    r4 = 0.3 # corresponding to [0.7, 1.0] span->lex 
    def len_span(span):
        return span.end - span.start

    for i in range(aug_num):
        if i == 304:
            print(1)
        if i%5000 == 0:
            print(i)
        if i == 483:
            print(1)

        idx = random.randint(0, len(cogs_ir_set)-1)
        desk_datum = cogs_ir_set[idx]
        desk_nodes = nodes_set[idx]
        rand_val = random.random()
        if rand_val <= r1 and len(datum_spans[idx]) > 0: # corresponding to [0, 0.3] (larger)tree->tree
            if 'tree' not in datum_strucs[idx]:
                continue
            desk_frag_encoding = random.choice(list(datum_strucs[idx]['tree'].keys()))
            desk_frag = random.choice(datum_strucs[idx]['tree'][desk_frag_encoding])
            desk_cate = desk_frag.category
            tgt_frag_encoding = random.choice(strucs_type['tree'][desk_cate])
            tgt_frag = random.choice(strucs[tgt_frag_encoding])


        elif rand_val <= r1+r2: # corresponding to [0.3, 0.6] lex->lex
            if len(datum_strucs[idx]) == 0:
                continue
            desk_frag_encoding = random.choice(list(datum_strucs[idx]['lex'].keys()))
            desk_frag = random.choice(datum_strucs[idx]['lex'][desk_frag_encoding])
            desk_cate = desk_frag.category
            tgt_frag_encoding = random.choice(strucs_type['lex'][desk_cate])
            tgt_frag = random.choice(strucs[tgt_frag_encoding])


        elif rand_val <= r1+r2+r3: # corresponding to [0.6, 0.7] span -> span
            if len(datum_spans[idx]) == 0:
                continue

            desk_frag_encoding = random.choice(list(datum_strucs[idx]['span'].keys()))
            desk_frag = random.choice(datum_strucs[idx]['span'][desk_frag_encoding])
            desk_cate = desk_frag.category
            tgt_frag_encoding = random.choice(strucs_type['span'][desk_cate])
            tgt_frag = random.choice(strucs[tgt_frag_encoding])

        else: # corresponding to [0.7, 1.0] span->lex

            max_tor = 3
            if len(datum_strucs[idx]) == 0:
                continue
            desk_frag_encoding = random.choice(list(datum_strucs[idx]['lex'].keys()))
            desk_frag = random.choice(datum_strucs[idx]['lex'][desk_frag_encoding])
            desk_cate = desk_frag.category
            while (desk_cate, desk_cate) not in spans.keys() and (max_tor > 0):
                desk_frag_encoding = random.choice(list(datum_strucs[idx]['lex'].keys()))
                desk_frag = random.choice(datum_strucs[idx]['lex'][desk_frag_encoding])
                desk_cate = desk_frag.category
                max_tor -= 1
            if (desk_cate, desk_cate) not in spans.keys():
                continue

            tgt_frag_encoding = random.choice(strucs_type['span'][(desk_cate, desk_cate)])
            tgt_frag = random.choice(strucs[tgt_frag_encoding])
        
        # now we have desk_frag and tgt_frag
        desk_datum = cogs_ir_set[desk_frag.idx]
        tgt_datum = aug_set[tgt_frag.idx]
        desk_span = desk_frag.span
        tgt_span = tgt_frag.span
        desk_nodes = nodes_set[desk_frag.idx]
        tgt_nodes = aug_nodes_set[tgt_frag.idx]
        desk_root = root(desk_frag.ent, desk_frag.ext)
        tgt_root = root(tgt_frag.ent, tgt_frag.ext)
        aug_datum = sub(desk_datum, desk_span, desk_nodes, desk_root,
                        tgt_datum, tgt_span, tgt_nodes, tgt_root)    
        if aug_datum["tokens"] == ['The', 'girl', 'was', 'given', 'a', 'table',
                                    'on', 'a', 'stool', 'on', 'the', 'table', '.']:
            print(111)
        if aug_datum["tokens"] == ['The', 'girl', 'proved', 'that', 'Harper', 'said', 
                                'that', 'a', 'girl', 'believed', 'that', 'a', 'banana', 'was', 
                                'burned', 'by', 'a', 'chicken', '.']:
            print(111)

        aug_list.append(aug_datum)
        #if x != cogs_ir_set[8337]:
        #    print(1)
    return aug_list

def subs_augment(aug_num):

    aug_list = list()
    def len_span(span):
        return span.end - span.start

    for i in range(aug_num):
        if i == 304:
            print(1)
        if i%5000 == 0:
            print(i)
        if i == 483:
            print(1)
        idx = random.randint(0, len(cogs_ir_set)-1)
        desk_datum = cogs_ir_set[idx]
        desk_nodes = nodes_set[idx]
        
        desk_frag = random.choice(datum_trees_[idx])
        # frag = fragment(proced_span, in_node, out_node, category, i, lc, rc)
        desk_cate = desk_frag.category
        tgt_frag = random.choice(trees_[desk_cate])
        if len_span(tgt_frag.span) < len_span(desk_frag.span):
            tgt_frag = random.choice(trees[desk_cate])
            if len_span(tgt_frag.span) < len_span(desk_frag.span):
                tgt_frag = random.choice(trees[desk_cate])
                if len_span(tgt_frag.span) < len_span(desk_frag.span):
                    tgt_frag = random.choice(trees[desk_cate])
        
        # now we have desk_frag and tgt_frag
        desk_datum = cogs_ir_set[desk_frag.idx]
        tgt_datum = aug_set[tgt_frag.idx]
        desk_span = desk_frag.span
        tgt_span = tgt_frag.span
        desk_nodes = nodes_set[desk_frag.idx]
        tgt_nodes = aug_nodes_set[tgt_frag.idx]
        desk_root = root(desk_frag.ent, desk_frag.ext)
        tgt_root = root(tgt_frag.ent, tgt_frag.ext)
        if i == 156433:
            print(1)
        aug_datum = sub(desk_datum, desk_span, desk_nodes, desk_root,
                        tgt_datum, tgt_span, tgt_nodes, tgt_root)   
        if len(aug_datum["tokens"]) > 3:
            if aug_datum["tokens"][1] == 'shark' and aug_datum["tokens"][2] == 'wanted':
                print(1) 
        aug_list.append(aug_datum)
        #if x != cogs_ir_set[8337]:
        #    print(1)
    return aug_list

def recover(aug_data):
    '''
    this function transforms augmented data list (intermediate representation) 
    to the original (input string, output string, type) triplets set.
    besides, we have a dedup operation in this function
    '''
    rec_data = set() # dedup
    for idx in range(len(aug_data)):
        datum = aug_data[idx]
        if datum["tokens"] == ['Emma', 'was', 'offered','a', 'cake','beside','the','table','.']:
            print(1)
        labeled_datum = LabeledExample(
            tokens = datum["tokens"],
            parent = datum["parent"],
            role = datum["role"],
            category = datum["category"],
            noun_type = datum["noun_type"],
            verb_name = datum["verb_name"],
            distribution = datum["distribution"] 
        ) 
        rec_inp = " ".join(datum["tokens"])
        rec_out = reconstruct_target(labeled_datum)
        rec_data.add((rec_inp, rec_out))
    return rec_data



def aug_format(augment_examples, outfile):
    data = []
    for (inp, out) in augment_examples:
        elem = dict()
        elem["inp"] = inp
        elem["out"] = out
        if out != [""]:
            data.append(elem)
    # data = [{"inp": inp, "out": out} for (inp, out) in augment_examples]
    with open(outfile, "w") as fh:
        json.dump(data, fh, indent=2)
    pass

trees = dict()
trees_ = dict()
# trees_ contains part of contents in lexs list, 
# trees_ would be leveraged when we are doing 'SUBS' exps
spans = dict()
lexs = dict()
lexs_dedup = set()
# deduplicate for lexs!!!
_trees_dedup = set()
trees_dedup = set()
spans_dedup = set()
nodes_set = dict()
aug_nodes_set = dict()
datum_lexs = dict()
datum_spans = dict()
datum_trees = dict()
# datum_lexs and datum_spans stores the spans & lexs for each datum
datum_trees_ = dict()

datum_strucs = dict()

strucs = dict()
strucs_type = dict()

x = copy.deepcopy(cogs_ir_set[8337])
def main():
    '''
    program entrance: 'main' function
    @1: span-extraction
    @2: span-processing and tagging
    @3: augment data
    @4: recover aug-data
    @5: save aug-data
    '''
    random.seed(0)
    global cogs_ir_set
    global aug_set
    total_num = len(aug_set)
    print(total_num)
    for i in range(total_num):
        datum = aug_set[i]
        if datum["distribution"] == "primitive" and datum["category"][0] == "CNOUN":
            _datum = dict()
            _datum["tokens"] = ["A",datum["tokens"][0]]
            _datum["parent"] = [-1,-1]
            _datum["role"] = ["",""]
            _datum["category"] = ["","CNOUN"]
            _datum["noun_type"] = ["","INDEF"]
            _datum["verb_name"] = ["",""]
            _datum["distribution"] = "primitive"
            aug_set.append(_datum)
            _datum = dict()
            _datum["tokens"] = ["The",datum["tokens"][0]]
            _datum["parent"] = [-1,-1]
            _datum["role"] = ["",""]
            _datum["category"] = ["","CNOUN"]
            _datum["noun_type"] = ["","DEF"]
            _datum["verb_name"] = ["",""]
            _datum["distribution"] = "primitive"
            aug_set.append(_datum)
            print(aug_set.pop(i))

    total_num = len(cogs_ir_set)
    print(total_num)
    for i in range(total_num):
        datum = cogs_ir_set[i]
        if datum["distribution"] == "primitive" and datum["category"][0] == "CNOUN":
            _datum = dict()
            _datum["tokens"] = ["A",datum["tokens"][0]]
            _datum["parent"] = [-1,-1]
            _datum["role"] = ["",""]
            _datum["category"] = ["","CNOUN"]
            _datum["noun_type"] = ["","INDEF"]
            _datum["verb_name"] = ["",""]
            _datum["distribution"] = "primitive"
            cogs_ir_set.append(_datum)
            _datum = dict()
            _datum["tokens"] = ["The",datum["tokens"][0]]
            _datum["parent"] = [-1,-1]
            _datum["role"] = ["",""]
            _datum["category"] = ["","CNOUN"]
            _datum["noun_type"] = ["","DEF"]
            _datum["verb_name"] = ["",""]
            _datum["distribution"] = "primitive"
            cogs_ir_set.append(_datum)
            print(cogs_ir_set.pop(i))         
    '''
    construct datum_lists
    '''
    total_num = len(cogs_ir_set)
    print(total_num)
    for i in range(total_num):
        datum_lexs[i] = list()
        datum_spans[i] = list()
        datum_trees_[i] = list()
        datum_trees[i] = list()
        datum_strucs[i] = dict()
        datum = cogs_ir_set[i]
        if i == 1598:
            a = 1
        prim_flag, subcompons, nodes = subcompon_extract(datum)
        nodes_set[i] = nodes

        
        #(interval_start, interval_end, in_node, out_node, frag_type)
        if prim_flag == True:
            if len(datum["tokens"]) == 1:
                ori_span = span(0,0)
                proced_span = proc_span(ori_span, datum["noun_type"])
                category = cates_transform(datum, 0)
                frag = fragment(proced_span, 0, 0, category, i, 0, 0)
            else:
                ori_span = span(1,1)
                proced_span = proc_span(ori_span, datum["noun_type"])
                category = cates_transform(datum, 1)
                frag = fragment(proced_span, 1, 1, category, i, 0, 0)
            '''
            proced_span = proc_span(ori_span, datum["noun_type"])
            category = cates_transform(datum, 0)
            frag = fragment(proced_span, 0, 0, category, i, 0, 0)
            '''
            datum_lexs[i].append(frag)
            datum_trees_[i].append(frag)

        elif (len(nodes) == 0) and subcompons == None:
            continue
        else:
            for compon in subcompons:
                #(interval_start, interval_end, in_node, out_node, frag_type)
                sta, end, in_node, out_node, frag_type, c_flag = compon
                lc, rc = c_flag
                ori_span = span(sta, end)
                proced_span = proc_span(ori_span, datum["noun_type"])

                if frag_type == "lex":
                    category = cates_transform(datum, sta)
                elif frag_type == "tree" or frag_type == "tree_":
                    # only tag its root
                    category = cates_transform(datum, in_node)
                elif frag_type == "span":
                    category = (cates_transform(datum, in_node),cates_transform(datum, out_node))
                    if category == ('VERB', 'VERB'):
                        #print(111)
                        pass
                frag = fragment(proced_span, in_node, out_node, category, i, lc, rc)

                if frag_type == "lex":
                    datum_lexs[i].append(frag)

                elif frag_type == "tree":
                    datum_trees[i].append(frag)
                 
                elif frag_type == "tree_":
                    datum_trees_[i].append(frag)

                elif frag_type == "span":
                    datum_spans[i].append(frag)
                
                # v2-mod
                if frag_type == "tree_":
                    continue
                frag_encoding = tuple(encode_frag(datum, nodes, proced_span))
                if frag_type not in datum_strucs[i]:
                    datum_strucs[i][frag_type] = dict()
                if frag_encoding not in datum_strucs[i][frag_type]:
                    datum_strucs[i][frag_type][frag_encoding] = list()

                datum_strucs[i][frag_type][frag_encoding].append(frag)


    '''
    construct lists
    '''
    total_num = len(aug_set)
    for i in range(total_num):
        datum = aug_set[i]
        if i == 1598:
            a = 1
        prim_flag, subcompons, nodes = subcompon_extract(datum)
        aug_nodes_set[i] = nodes
        #(interval_start, interval_end, in_node, out_node, frag_type)
        if prim_flag == True:
            if len(datum["tokens"]) == 1:
                ori_span = span(0,0)
                proced_span = proc_span(ori_span, datum["noun_type"])
                category = cates_transform(datum, 0)
                frag = fragment(proced_span, 0, 0, category, i, 0, 0)
            else:
                ori_span = span(1,1)
                proced_span = proc_span(ori_span, datum["noun_type"])
                category = cates_transform(datum, 1)
                frag = fragment(proced_span, 1, 1, category, i, 0, 0)


            if category not in lexs.keys():
                lexs[category] = list()
            if category not in trees_.keys():
                # for subtrees we tentatively annotate its root only.
                trees_[category] = list()
            
            if len(datum["tokens"]) == 1:
                if datum["tokens"][0] not in lexs_dedup:
                    # dedup for lexs
                    lexs_dedup.add(datum["tokens"][0])
                    lexs[category].append(frag)
            else:
                cnoun_prim = (' ').join(datum["tokens"])
                #if datum["tokens"][1] not in lexs_dedup:
                if cnoun_prim not in lexs_dedup:
                    # dedup for lexs
                    lexs_dedup.add(cnoun_prim)
                    lexs[category].append(frag)

            frag_str = (' ').join(datum["tokens"])
            if frag_str not in _trees_dedup:
                _trees_dedup.add(frag_str)
                trees_[category].append(frag)
            


        elif (len(nodes) == 0) and subcompons == None:
            continue
        else:
            for compon in subcompons:
                #(interval_start, interval_end, in_node, out_node, frag_type)
                sta, end, in_node, out_node, frag_type, c_flag = compon
                lc, rc = c_flag
                ori_span = span(sta, end)
                proced_span = proc_span(ori_span, datum["noun_type"])

                dup_flag = 0

                if frag_type == "lex":
                    category = cates_transform(datum, sta)
                elif frag_type == "tree" or frag_type == "tree_":
                    # only tag its root
                    category = cates_transform(datum, in_node)
                elif frag_type == "span":
                    category = (cates_transform(datum, in_node),cates_transform(datum, out_node))
                    if category == ('VERB', 'VERB'):
                        #print(111)
                        pass
                frag = fragment(proced_span, in_node, out_node, category, i, lc, rc)
                if frag_type == "lex":
                    if category not in lexs.keys():
                        lexs[category] = list()
                    if datum["tokens"][sta] not in lexs_dedup:
                        # dedup for lexs
                        lexs_dedup.add(datum["tokens"][sta])
                        lexs[category].append(frag)
                    else:
                        dup_flag = 1
                    #lexs[category].append(frag)

                elif frag_type == "tree":
                    if category not in trees.keys():
                        trees[category] = list()

                    frag_str = (' ').join(datum["tokens"][frag.span.start : frag.span.end+1])
                    if frag_str not in trees_dedup:
                        that_cnt = frag_str.count("that")
                        if that_cnt >= 2:
                            a = 1
                        trees_dedup.add(frag_str)
                        trees[category].append(frag)
                    else:
                        dup_flag = 1

                elif frag_type == "tree_":
                    if category not in trees_.keys():
                        trees_[category] = list()

                    frag_str = (' ').join(datum["tokens"][frag.span.start : frag.span.end+1])
                    if frag_str not in _trees_dedup:
                        _trees_dedup.add(frag_str)
                        trees_[category].append(frag)

                elif frag_type == "span":
                    if category not in spans.keys():
                        spans[category] = list()

                    frag_str = (' ').join(datum["tokens"][frag.span.start : frag.span.end+1])
                    if frag_str not in spans_dedup:
                        spans_dedup.add(frag_str)                    
                        spans[category].append(frag)
                    else:
                        dup_flag = 1



                #v2-mod
                if frag_type == "tree_":
                    continue
                
                if dup_flag == 1:
                    continue
                
                frag_encoding = tuple(encode_frag(datum, nodes, proced_span))
                if frag_encoding not in strucs:
                    strucs[frag_encoding] = list()
                strucs[frag_encoding].append(frag)              
                if frag_type not in strucs_type:
                    strucs_type[frag_type] = dict()
                if category not in strucs_type[frag_type]:
                    strucs_type[frag_type][category] = list()
                if frag_encoding not in strucs_type[frag_type][category]:
                    strucs_type[frag_type][category].append(frag_encoding)
    
    aug_num = int(4e5)
    aug_data = augment(aug_num)
    that_cntt=0
    for aug_datum in aug_data:
        sent = ' '.join(aug_datum["tokens"])
        if (sent.count('that') >= 4):
            that_cntt += 1
    print(that_cntt)
    
    recovered_aug_data = recover(aug_data)
    print(len(recovered_aug_data))
    
    save_path_ = "/data2/home/zhaoyi/compsub/augs/data/cogs/iterative-aug/aug_cogs.spans.4.0.debug.txt"

    with open(save_path_, "w") as fw:
        for datum in recovered_aug_data:
            inp, out = datum
            fw.write(inp+'	'+out+'	augment'+'\n')
    
    def get_dataset(path):
        fr = open(path, "r")
        dataset = []
        for line in fr.readlines():
            inp_flag = 0
            inp = []
            out = []
            inp_line, out_line, _ = line.strip().split('\t')
            out_line = out_line.replace(" _ ", "_")
            # change x _ 1 towards x_1
            inp = inp_line.strip().split(' ')
            out = out_line.strip().split(' ')
            dataset.append((inp, out))
        return dataset
    
    aug_set = get_dataset(save_path_)
    save_path = "/data2/home/zhaoyi/compsub/augs/data/cogs/iterative-aug/aug_cogs.spans.4.0.debug.json"
    aug_format(aug_set, save_path)
    
if __name__ == '__main__':
    main()
    