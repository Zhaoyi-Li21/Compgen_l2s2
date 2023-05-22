# L2S2: Learning to Substitute Spans towards Improving Compositional Generalization
Implementation of the paper "Learning to Substitute Span towards Improving Compositional Generalization", Zhaoyi Li, Ying Wei and Defu Lian, ACL 2023 Main Conference.
### Introduction
This work introduces a novel composiitonal data augmentation method SpanSub to enable multi-grained sub-components recomposition, and a parameterized and differentiable data augmentation framework to encourage automatical recognition of hard compositions of elusive concepts and novel surroundings.
<div align="center">
<img src="./figures/l2s2.jpg" width="70%">
</div>
### Requirements

### Install the running environment

### Experiments
#### E.g.1: Run SpanSub on SCAN dataset

#### E.g.2: Run L2S2 on SCAN dataset

#### E.g.3: Run SpanSub on GeoQuery dataset

### Acknowledgement
The code in this repository is partly based on the following baseline implementations: \\
GECA(ACL'19) : https://github.com/jacobandreas/geca \\
SUBS(NAACL'22) : https://github.com/SALT-NLP/SUBS \\
MET-PRIM(EMNLP'22) : https://github.com/owenzx/met-primaug \\
Besides, some parsers are adapted from OpenNMT(https://github.com/OpenNMT/OpenNMT-py) and fairseq(https://github.com/pytorch/fairseq)

### Cite L2S2
If you find this repo/paper useful for your research, please consider citing the paper:
```
bibtex of l2s2 paper
```