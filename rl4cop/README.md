# ReImplementation for POMO

This repository provides a reimplementation of **POMO-TSP** and saved trained models as described in the paper:
> POMO: Policy Optimization with Multiple Optima for Reinforcement Learning
> accepted at NeurIPS 2020
http://arxiv.org/abs/2010.16011

The code also provides a **GCN Encoder** which can also get the same result mentioned in paper.

### Basic Usage

To train the model, you can edit `config.yaml` to change the size of the problem or other hyper-parameters before training and then run the following command in the shell (you need to set the hyper-parameter *save_dir* whether in the yaml file or in the command line with absolute path):

```shell
python train.py save_dir="./"
```

Note that you can also see the training curve online with `wandb` if you have wandb's account and login wandb before run the command.


### Used Libraries
- torch >= 1.7
- pytorch-lighning (for training pipline)
- dgl (for GCN encoder)
- hydra-core (for conveniencely using hyper-parameters)
- wandb (for online logging)