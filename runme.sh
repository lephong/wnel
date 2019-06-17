python3 -u -m nel.main --mode train --inference star --multi_instance --n_negs 5 --margin 0.1 --n_rels 1  --eval_after_n_epochs 6 --n_epochs 6  --ent_top_n 30 --preranked_data data/generated/test_train_data/preranked_all_datasets_50kRCV1_large --n_not_inc 5 --n_docs 50000

