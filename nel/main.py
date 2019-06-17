import nel.dataset as D
from nel.mulrel_ranker import MulRelRanker
from nel.ed_ranker import EDRanker
import pickle

import nel.utils as utils

from random import shuffle
from pprint import pprint

import argparse

parser = argparse.ArgumentParser()
use_large = True

datadir = 'data/generated/test_train_data/'
conll_path = 'data/basic_data/test_datasets/'
person_path = 'data/basic_data/p_e_m_data/persons.txt'
voca_emb_dir = 'data/generated/embeddings/word_ent_embs/'
large_voca_emb_dir = 'data/generated/embeddings/large_word_ent_embs/'

if use_large:
    voca_emb_dir = large_voca_emb_dir


ModelClass = MulRelRanker


# general args
parser.add_argument("--mode", type=str,
                    help="train, eval, prerank, or ed",
                    default='train')
parser.add_argument("--model_path", type=str,
                    help="model path to save/load",
                    default="model")
parser.add_argument("--filelist", type=str,
                   help="filelist for ed (candidate filenames should be ended by .csv)",
                   default=None)
parser.add_argument("--preranked_data", type=str,
                    help="filelist for ed (candidate filenames should be ended by .csv)",
                    default=None)

# args for preranking (i.e. 2-step candidate selection)
parser.add_argument("--n_cands_before_rank", type=int,
                    help="number of candidates",
                    default=30)
parser.add_argument("--prerank_ctx_window", type=int,
                    help="size of context window for the preranking model",
                    default=50)
parser.add_argument("--keep_p_e_m", type=int,
                    help="number of top candidates to keep w.r.t p(e|m)",
                    default=4)
parser.add_argument("--keep_ctx_ent", type=int,
                    help="number of top candidates to keep w.r.t using context",
                    default=3)

# args for local model
parser.add_argument("--ctx_window", type=int,
                    help="size of context window for the local model",
                    default=100)
parser.add_argument("--tok_top_n", type=int,
                    help="number of top contextual words for the local model",
                    default=25)


# args for global model
parser.add_argument("--mulrel_type", type=str,
                    help="type for multi relation (rel-norm or ment-norm)",
                    default='ment-norm')
parser.add_argument("--n_rels", type=int,
                    help="number of relations",
                    default=5)
parser.add_argument("--hid_dims", type=int,
                    help="number of hidden neurons",
                    default=100)
parser.add_argument("--snd_local_ctx_window", type=int,
                    help="local ctx window size for relation scores",
                    default=6)
parser.add_argument("--dropout_rate", type=float,
                    help="dropout rate for relation scores",
                    default=0.3)
parser.add_argument('--uniform_att', action='store_true',
                    help='with uniform attention (equ. to G&H')


# args for training
parser.add_argument("--n_epochs", type=int,
                    help="max number of epochs",
                    default=200)
parser.add_argument("--dev_f1_change_lr", type=float,
                    help="dev f1 to change learning rate",
                    default=0.915)
parser.add_argument("--n_not_inc", type=int,
                    help="number of evals after dev f1 not increase",
                    default=10)
parser.add_argument("--eval_after_n_epochs", type=int,
                    help="number of epochs to eval",
                    default=5)
parser.add_argument("--learning_rate", type=float,
                    help="learning rate",
                    default=1e-4)
parser.add_argument("--margin", type=float,
                    help="margin",
                    default=0.01)

parser.add_argument("--n_negs", type=int,
                    help="number of negatives",
                    default=0)
parser.add_argument('--semisup', action='store_true',
                    help='using multi-instance learning and supervised learning (for semi supervised learning)')
parser.add_argument('--multi_instance', action='store_true',
                    help='using multi-instance learning')

# args for inference (LBP or star)
parser.add_argument("--inference", type=str,
                    help="inference method (LBP or star)",
                    default='LBP')
parser.add_argument("--df", type=float,  # for LBP
                    help="dampling factor (for LBP)",
                    default=0.5)
parser.add_argument("--n_loops", type=int,  # for LBP
                    help="number of LBP loops",
                    default=10)
parser.add_argument("--ent_top_n", type=int,  # for star
                    help="number of kept neighbours",
                    default=30)

parser.add_argument("--n_docs", type=int,
                    help="number of documents",
                    default=1000000)
parser.add_argument("--dev_ner", type=str,
                   help="dev net path",
                   default=None)


# args for debugging
parser.add_argument("--print_rel", action='store_true')
parser.add_argument("--print_incorrect", action='store_true')


args = parser.parse_args()
if (args.semisup or args.multi_instance) and args.n_negs < 1:
    raise Exception('multi instance requires at least 1 negative sample')

if __name__ == "__main__":
    print('create model')
    word_voca, word_embeddings = utils.load_voca_embs(voca_emb_dir + 'dict.word',
                                                      voca_emb_dir + 'word_embeddings.npy')
    print('word voca size', word_voca.size())
    snd_word_voca, snd_word_embeddings = utils.load_voca_embs(voca_emb_dir + '/glove/dict.word',
                                                              voca_emb_dir + '/glove/word_embeddings.npy')
    print('snd word voca size', snd_word_voca.size())

    entity_voca, entity_embeddings = utils.load_voca_embs(voca_emb_dir + 'dict.entity',
                                                          voca_emb_dir + 'entity_embeddings.npy')
    print('entity voca size', entity_voca.size())

    config = {'hid_dims': args.hid_dims,
              'emb_dims': entity_embeddings.shape[1],
              'freeze_embs': True,
              'tok_top_n': args.tok_top_n,
              'margin': args.margin,
              'word_voca': word_voca,
              'entity_voca': entity_voca,
              'word_embeddings': word_embeddings,
              'entity_embeddings': entity_embeddings,
              'snd_word_voca': snd_word_voca,
              'snd_word_embeddings': snd_word_embeddings,
              'dr': args.dropout_rate,
              'multi_instance': args.multi_instance,
              'semisup': args.semisup,
              'preranked_data': args.preranked_data,
              'uniform_att': args.uniform_att,
              'args': args}

    if args.mode == 'prerank':
        print('load entity net from', datadir + '/../entity_net.dat')
        entity_net = pickle.load(open(datadir + '/../entity_net.dat', 'rb'))
        config['ent_net'] = entity_net

    if args.multi_instance or args.semisup:
        config['n_negs'] = args.n_negs

    if ModelClass == MulRelRanker:
        config['inference'] = args.inference
        config['df'] = args.df
        config['n_loops'] = args.n_loops
        config['ent_top_n'] = args.ent_top_n

        config['n_rels'] = args.n_rels
        config['mulrel_type'] = args.mulrel_type
    else:
        raise Exception('unknown model class')

    # pprint(config)
    ranker = EDRanker(config=config)

    if args.mode == 'prerank':
        conll = D.CoNLLDataset(datadir, person_path, conll_path)

        if args.filelist is not None:
            if args.multi_instance:
                conll.train = {}
            with open(args.filelist, 'r') as flist:
                for fname in flist:
                    fname = fname.strip()
                    print('load file from', fname)
                    conll_path = fname
                    cands_path = conll_path + '.csv'
                    data = D.CoNLLDataset.load_file(conll_path, cands_path, person_path)
                    print('#docs', len(data))
                    conll.train = {**conll.train, **data}
                    if len(conll.train) > args.n_docs:
                        break

        conll.train = dict(list(conll.train.items())[:min(args.n_docs, len(conll.train))])
        print(len(conll.train))

        all_datasets = [['train', conll.train, None],
                        ['aida-A', conll.testA, None],
                        ['aida-B', conll.testB, None],
                        ['msnbc', conll.msnbc, None],
                        ['aquaint', conll.aquaint, None],
                        ['ace2004', conll.ace2004, None],
                        ['clueweb', conll.clueweb, None],
                        ['wikipedia', conll.wikipedia, None]]

        for dataset in all_datasets:
            name, data = dataset[0], dataset[1]
            print('prerank', name)
            data = ranker.get_data_items(data, predict=False if name=='train' else True)
            dataset[2] = data

        path = args.preranked_data
        print('save to all into', path)
        with open(path, 'wb') as f:
            pickle.dump(all_datasets, f)

    elif args.mode == 'train':
        path = args.preranked_data
        print('load all datasets from', path)
        with open(path, 'rb') as f:
            all_datasets = pickle.load(f)
            conll_train, preranked_train = all_datasets[0][1], all_datasets[0][2]
            preranked_train = preranked_train[:min(args.n_docs, len(preranked_train))]
            shuffle(preranked_train)

            dev_datasets = [(all_datasets[i][0], all_datasets[i][1]) for i in range(1, len(all_datasets))]
            preranked_dev = [(all_datasets[i][0], all_datasets[i][2]) for i in range(1, len(all_datasets))]

        print('training...')
        print('total training', len(preranked_train))
        config = {'lr': args.learning_rate,
                  'n_epochs': args.n_epochs,
                  'multi_instance': args.multi_instance,
                  'semisup': args.semisup}
        pprint(config)
        ranker.train(conll_train, dev_datasets, config,
            preranked_train=preranked_train, preranked_dev=preranked_dev)

    elif args.mode == 'eval':
        path = args.preranked_data
        print('load all datasets from', path)
        with open(path, 'rb') as f:
            all_datasets = pickle.load(f)
            conll_train, preranked_train = all_datasets[0][1], all_datasets[0][2]
            preranked_train = preranked_train[:min(args.n_docs, len(preranked_train))]
            org_dev_datasets = [(all_datasets[i][0], all_datasets[i][1]) for i in range(1, len(all_datasets))]
            dev_datasets = [(all_datasets[i][0], all_datasets[i][2]) for i in range(1, len(all_datasets))]
            dev_datasets = [(all_datasets[i][0], all_datasets[i][1], all_datasets[2]) for i in range(1, len(all_datasets))]

        for di, (dname, data, preranked) in enumerate(dev_datasets):
            ranker.model._coh_ctx_vecs = []
            predictions = ranker.predict(preranked)
            print(dname, utils.tokgreen('micro F1: ' + str(D.eval(data, predictions))))

    elif args.mode == 'ed':
        with open(args.filelist, 'r') as flist:
            for fname in flist:
                fname = fname.strip()
                print('load file from', fname)
                conll_path = fname
                cands_path = conll_path + '.csv'
                data = D.CoNLLDataset.load_file(conll_path, cands_path, person_path)

                data = ranker.get_data_items(data, predict=True)
                print('#docs', len(data))
                continue

                ranker.model._coh_ctx_vecs = []
                predictions = ranker.predict(data, rank=True)
                # pprint(predictions)

                with open(fname + '.ed', 'w') as fout:
                    for dname, links in predictions.items():
                        dname = dname.split()[0][:-1]
                        for l in links:
                            fout.write(dname + '\t' + '\t'.join([e[0] + ',' + str(e[1]) for e in l]) + '\n')

