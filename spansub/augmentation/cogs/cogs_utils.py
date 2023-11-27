import random

max_seqlen = 30
encode_tag = dict()
tags = ['VERB','NOUN','PREP','xcomp']
for i in range(len(tags)):
    encode_tag[tags[i]] = i

padding = -2

a = ['a', 'b', 'c', 'a']
#a = set(a)
a = tuple(a)
my_dict = dict()
my_dict[a] = 1

def encode_frag(datum, nodes, span):
    '''
    this function is going to transform a tree-frag
    into a seq encoding form.
    e.g., a `tree` category span: Lina knew that Tom was told that a cake on the table was ate.
    it's structure can be illustrated like:
    knew(VERB) -> told(VERB) -> ate(VERB)
    |               |           |
    v               v           v
    Lina(NOUN)     Tom(NOUN)   a cake(NOUN) -> on(PREP) -> the table(NOUN)
    when we encode ['VERB','NOUN','PREP'] into [0, 1, 2]:
    the above tree can directly expressed as:
    0->0->0
    |  |  |
    v  v  v
    1  1  1 ->2 ->1
    the inorder traverse of this tree is 
    inord_trav = [1, 0, 1, 0, 1, 2, 1, 0, -2, -2, -2, -2, -2, -2, -2]
    its thread list is 
    thread = [1, -1, 3, 1, 7, 4, 5, 3, -2, -2, -2, -2, -2, -2, -2]
    where `-2` is padding,
    the encode is : encode = [inord_trav : thread]
    '''
    span_tags = list()
    span_parents = list()
    span_nodes = list()
    nodes_map = dict()

    for node in nodes:
        if node >= span.start and node <= span.end:
            span_nodes.append(node)
            # nodes_map maps the node to the index of span_nodes
            nodes_map[node] = len(span_nodes) - 1

    for node in span_nodes:
       
        if datum["category"][node] == 'PNOUN' or datum["category"][node] == 'CNOUN':
            tag = 'NOUN'
        elif datum['category'][node] == 'VERB':
            tag = 'VERB'
        elif datum['role'][node] == 'PREP':
            tag = 'PREP'
        elif datum['role'][node] == 'xcomp':
            tag = 'xcomp'
        else:
            raise Exception("tag ERROR")

        span_tags.append(encode_tag[tag])

        if datum["parent"][node] >= span.start and datum["parent"][node] <= span.end:
            assert datum["parent"][node] in span_nodes
            span_parents.append(nodes_map[datum["parent"][node]])
        else:
            span_parents.append(-1)
        
    # padding
    while len(span_tags) < max_seqlen:
        span_tags.append(padding)
        span_parents.append(padding)
    encoding = span_tags + span_parents
    return encoding
    