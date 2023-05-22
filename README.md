# L2S2: Learning to Substitute Spans towards Improving Compositional Generalization
Implementation of the paper "Learning to Substitute Span towards Improving Compositional Generalization", Zhaoyi Li, Ying Wei and Defu Lian, ACL 2023 Main Conference.
### 1.Introduction
This work introduces a novel composiitonal data augmentation method SpanSub to enable multi-grained sub-components recomposition, and a parameterized and differentiable data augmentation framework to encourage automatical recognition of hard compositions of elusive concepts and novel surroundings.
<div align="center">
<img src="./figures/l2s2.jpg" width="60%">
</div>

### 2.Requirements

### 3.Install the running environment

### 4.Experiments
#### 4.1: Run SpanSub on SCAN dataset

#### 4.2: Run L2S2 on SCAN dataset

#### 4.3: Run SpanSub on GeoQuery dataset

### 5.Acknowledgement
The code in this repository is partly based on the following baseline implementations: \
1. GECA(ACL'19) : https://github.com/jacobandreas/geca \
2. SUBS(NAACL'22) : https://github.com/SALT-NLP/SUBS \
3. MET-PRIM(EMNLP'22) : https://github.com/owenzx/met-primaug \
4. Besides, some parsers are adapted from OpenNMT(https://github.com/OpenNMT/OpenNMT-py) and fairseq(https://github.com/pytorch/fairseq)

### 6.Cite this work
If you find this repo/paper useful for your research, please consider citing the paper:
```
bibtex of l2s2 paper
```