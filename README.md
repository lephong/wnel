wnel: A weak supervised learning approach for entity linking
========

A Python implementation of the approach proposed in

[1] Phong Le and Ivan Titov (2019). [Boosting Entity Linking Performance by Leveraging Unlabeled Documents](https://arxiv.org/pdf/1906.01250.pdf).

Written and maintained by Phong Le (lephong.xyz [at] gmail.com)

### Installation

Requirements: Python 3.7, Pytorch 0.4, CUDA 8

### Usage

The following instruction is for replicating the experiments reported in [1]. 
Note that training and testing take lots of RAM (about 30GB) because 
some files related to Freebase have to be loaded. 


#### Data

Download data from [here](https://drive.google.com/...) 
and unzip to the main folder (i.e. your-path/wnel).


#### Train

To train, from the main folder run 
    
    python3 -u -m nel.main --mode train --inference star --multi_instance --n_negs 5 --margin 0.1 --n_rels 1  --eval_after_n_epochs 6 --n_epochs 6  --ent_top_n 30 --preranked_data data/generated/test_train_data/preranked_all_datasets_50kRCV1_large --n_not_inc 50 --n_docs 50000

Using a GTX 1080 Ti GPU it will take about 30 minutes. The output is a model saved in two files: 
`model.config` and `model.state_dict` . 

For more options, please have a look at `nel/main.py` 


