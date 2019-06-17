import sys
from nel.vocabulary import Vocabulary
import pickle

wiki_prefix = 'en.wikipedia.org/wiki/'


def read_ent2id(ent_dic_path):
    print('load ent dic from', ent_dic_path)
    ent_dic = Vocabulary.load(ent_dic_path)
    ent2id = ent_dic.word2id
    return ent2id

def read_exlude(exclude_path):
    exclude = []
    with open(exclude_path, 'r') as f:
        for line in f:
            title = wiki_prefix + line.split('title=')[-1][1:-2].replace(' ', '_').replace('"', '%22')
            title = ent2id.get(title, -1)
            if title != -1:
                exclude.append(title)
    exclude = set(exclude)
    return exclude

def read_ent_net(net_path, exclude, ent2id):
    network = {}
    with open(net_path, 'r') as f:
        count = 0
        for line in f:
            comps = line.strip().split('\t')
            if len(comps) <= 1:
                # print('sthing wrong with', comps)
                continue

            doc = [ent2id.get(wiki_prefix + e.replace(' ', '_').replace('"', '%22'), -1) for e in comps[1:]]
            try:
                title = wiki_prefix + comps[1].split('title=')[-1][1:-2].replace(' ', '_').replace('"', '%22')
                title = ent2id.get(title, -1)
            except:
                print(comps)
                print(comps[1].split('title='))

            if title in exclude:
                continue

            doc.append(title)

            for i, e in enumerate(doc):
                if e == -1:
                    continue

                l = 20
                if i == len(doc) - 1:  # title
                    l = 1000

                if e not in network:
                    network[e] = {}
                ne = network[e]
                for j in range(max(0, i-l), min(len(doc), i+l)):
                    d = doc[j]
                    if d == -1:
                        continue
                    ne[d] = ne.get(d, 0) + 1

            count += 1
            if count % 1000 == 0:
                print(count, end='\r')
            # if count > 100000:
            #     break

    print('pruning....')
    for e in network:
        ne = network[e]
        m = 0
        for d in ne:
            m = max(m, ne[d])
        for d in list(ne.keys()):
            ne[d] /= m
            if ne[d] < 1e-3:
                del ne[d]

    return network


def save(network, path):
    with open(path, 'wb') as f:
        pickle.dump(network, f)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('USE: python3 -m nel.read_entity_network ../data/generated/embeddings/ /tmp/x')

    data_dir = sys.argv[1]

    net_path = data_dir + '/wiki_entity_net.txt'
    exclude_path = data_dir + '/exclude.txt'
    ent_dic_path = data_dir + '/embeddings/large_word_ent_embs/dict.entity'
    out_path = sys.argv[2]

    print('------ load ent2id ---------')
    ent2id = read_ent2id(ent_dic_path)

    print('------ load exclude -------')
    exclude = read_exlude(exclude_path)

    print('load net from', net_path)
    network = read_ent_net(net_path, exclude, ent2id)
    print('print to file', out_path)
    save(network, out_path)
    print('done')

