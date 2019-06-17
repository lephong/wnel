import torchfile
import numpy as np
import sys

mode = sys.argv[1]
path = sys.argv[2]

if mode == 'entity':
    embs_path = path + '/ent_vecs/ent_vecs__ep_177.t7'
    name_id_map_path = path + '/ent_name_id_map_RLTD.t7'
    thid_to_wikiid_path = path + '/all_candidate_ents_ed_rltd_datasets_RLTD.t7'
    freq_path = path + '/ent_wiki_freq.txt'
elif mode == 'word':
    embs_path = 'GoogleNews-vectors-negative300.t7'
    words_path = 'common_top_words_freq_vectors_w2v.t7'
    words_wiki_path = 'word_wiki_freq.txt'

def process_entity():
    ent_freq = {}
    with open(freq_path, 'r') as f:
        count = 0
        for line in f:
            eid, name, freq = line.strip().split('\t')
            name = name.replace('"', '%22').replace(' ', '_')
            ent_freq[name] = freq
            count += 1
            if count % 1000 == 0:
                print(count, end='\r')

    # embeddings
    print('load embeddings from', embs_path)
    embs = torchfile.load(embs_path)

    print('save embeddings to file entity_embeddings')
    np.save('entity_embeddings', embs)

    # dictionary
    print('load name_id_map from', name_id_map_path)
    wikiid2name = torchfile.load(name_id_map_path)[b'ent_wikiid2name']

    print('load thid_to_wikiid from', thid_to_wikiid_path)
    thid2wikiid = torchfile.load(thid_to_wikiid_path)[b'reltd_ents_rltdid_to_wikiid']

    name2id = {}
    id2name = []
    thid = 0

    with open('dict.entity', 'w') as f:
        for wikiid in thid2wikiid:
            try:
                name = wikiid2name[wikiid].decode('utf-8').replace('"', '%22').replace(' ', '_')
            except:
                name = 'UNK-wikiid-' + str(wikiid)
                # print(wikiid)
            f.write('en.wikipedia.org/wiki/' + name + '\t' + ent_freq.get(name, '100') + '\n')

            name2id[name] = thid
            id2name.append(name)
            thid += 1


    # sanity check
    targets = {'Bill_Clinton', 'Vietnam', 'Edinburgh'}
    for target in targets:
        print('---', target)
        emb = embs[name2id[target], :].reshape(-1, 1)
        scores = np.matmul(embs, emb).reshape(-1)

        for i in range(10):
            maxid = np.argmax(scores)
            scores[maxid] = -1e10
            print(id2name[maxid])


def process_word():
    # embeddings
    print('load embeddings from', embs_path)
    embs = torchfile.load(embs_path)

    print('save embeddings to file word_embeddings')
    np.save('word_embeddings', embs)

    # dictionary
    print('load words', words_path)
    words = set([w.decode('utf-8') for w in torchfile.load(words_path).keys()])

    name2id = {'#UNK#': 0}
    id2name = ['#UNK#']
    thid = 1

    with open(words_wiki_path, 'r') as fw:
        with open('dict.word', 'w') as f:
            f.write('#UNK#\t1000\n')

            for line in fw:
                w, _ = line.split('\t')
                if w in words:
                    f.write(w + '\t1000\n')

                    name2id[w] = thid
                    id2name.append(w)
                    thid += 1

    # sanity check
    targets = {'dog', 'Vietnam', 'Edinburgh'}
    for target in targets:
        print('---', target)
        emb = embs[name2id[target], :].reshape(-1, 1)
        scores = np.matmul(embs, emb).reshape(-1)

        for i in range(10):
            maxid = np.argmax(scores)
            scores[maxid] = -1e10
            print(id2name[maxid])


if __name__ == "__main__":
    if mode == "entity":
        process_entity()
    elif mode == "word":
        process_word()

