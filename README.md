# PyTorch-KPN
PyTorch implementation of Keypoint Proposal Network, with support for training, inference and evaluation.

## Installation

Clone this repo to local.

```
git clone https://github.com/SJ-Chuang/PyTorch-KPN.git
cd PyTorch-KPN
```

## Train on a custom dataset

We use the [hand keypoint detection dataset](http://domedb.perception.cs.cmu.edu/handdb.html) which has 21 hand joints as a demonstration of training.

### Prepare the dataset

```
wget -O hand_synth.zip https://github.com/SJ-Chuang/PyTorch-KPN/releases/download/v1.0/hand_synth.zip
unzip hand_synth.zip
```

### Start training

```
python train.py
```

## References

```
@inproceedings{simon2017hand,
author = {Tomas Simon and Hanbyul Joo and Iain Matthews and Yaser Sheikh},
booktitle = {CVPR},
title = {Hand Keypoint Detection in Single Images using Multiview Bootstrapping},
year = {2017}
}
```

