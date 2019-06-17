import os, sys

datadir = '/disk/scratch1/ple/workspace/corpora/YAGO/'
yagoTaxonamy_path = datadir + 'yagoTaxonomy.ttl'
yagoTransitiveType_path = datadir + 'yagoTransitiveType.ttl'
ent2cat_path = datadir + 'ent2cat.txt'

# read taxonamy
print('read taxonamy from', yagoTaxonamy_path)
all_nodes = {}
with open(yagoTaxonamy_path, 'r', encoding='utf8') as f:
    for line in f:
        try:
            child, _, parent, _ = line.strip().split()
        except:
            continue

        child = child[1:-1] if child[0] == '<' else child
        parent = parent[1:-1] if parent[0] == '<' else parent
        if not child.startswith('wordnet'):
            continue

        child_node = all_nodes.get(child, None)
        if child_node is None:
            child_node = {'name': child, 'parent': None, 'children': []}
            all_nodes[child] = child_node

        parent_node = all_nodes.get(parent, None)
        if parent_node is None:
            parent_node = {'name': parent, 'parent': None, 'children': []}
            all_nodes[parent] = parent_node

        parent_node['children'].append(child_node)
        child_node['parent'] = parent_node


# find root
print('find roots')
roots = []
for node in all_nodes.values():
    if node['parent'] is None:
        roots.append(node)
        print(node['name'])

print('mark nodes')
def mark_node(node, route):
    node['route'] = route
    for i, c in enumerate(node['children']):
        mark_node(c, route + str(i) + '-')

for i, r in enumerate(roots):
    mark_node(r, str(i) + '-')

# get categories for all entities
def find_leaves(nodes):
    if len(nodes) == 1:
        return nodes

    nodes = sorted(nodes, key=lambda n: n['route'])
    ret = []

    for i in range(len(nodes) - 1):
        select = True
        for j in range(i+1, len(nodes)):
            if nodes[j]['route'].startswith(nodes[i]['route']):
                select = False
                break
        if select:
            ret.append(nodes[i])

    return ret


# collect all triples
print('collect all triples from ', yagoTransitiveType_path)
ent2cat = {}

with open(yagoTransitiveType_path, 'r', encoding='utf8') as f:
    count = 0
    for line in f:
        count += 1
        if count % 100000 == 0:
            #break
            print(count, end='\r')

        if line.startswith('<'):
            try:
                ent, _, cat, _ = line.strip().split()
            except:
                continue
            ent = ent[1:-1] if ent[0] == '<' else ent
            cat = cat[1:-1] if cat[0] == '<' else cat

            if cat.startswith('wordnet_'):
                if cat in all_nodes:
                    cats = ent2cat.get(ent, None)
                    if cats is None:
                        cats = []
                        ent2cat[ent] = cats
                    cats.append(all_nodes[cat])

print(len(ent2cat))

# print out
with open(ent2cat_path, 'w', encoding='utf8') as f:
    for ent, cats in ent2cat.items():
        cats = find_leaves(cats)
        for c in cats:
            f.write(ent + '\t' + c['name'] + '\n')
