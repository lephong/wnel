import spacy
import sys
from os import listdir, system
from os.path import isfile, join
from multiprocessing import Process
from math import ceil
import subprocess

indir_path = sys.argv[1]
outdir_path = sys.argv[2]
nthreads = int(sys.argv[3])

def process(filenames, tid):
    nlp = spacy.load('en')
    for fname in filenames:
        in_path = fname
        out_path = outdir_path + '/' + fname.split('/')[-1] + '.ner'
        print('tagging', fname, 'and write to', out_path)
        cmd = r"""awk '{if (NF > 2) printf("%s ", $2); else printf("\n")}' """ + fname + r' > /tmp/spacy' + str(tid)
        print(cmd)
        system(cmd)
        ner(nlp, '/tmp/spacy' + str(tid), out_path)
        print('done', fname)

def ner(nlp, in_path, out_path):
    with open(out_path, 'w') as fout:
        with open(in_path, 'r') as fin:
            count = 0
            for line in fin:
                line = line.strip().replace('-LRB-', '(').replace('-RRB-', ')')
                if line.startswith('- DOC'):
                    fout.write(line.replace('- DOCSTART', '-DOCSTART- (') + ')\n')
                    continue

                doc = nlp(line)
                start = 0
                string = str(doc)
                for ent in doc.ents:
                    if ent.label_ not in {'PERSON', 'NORP', 'FACILITY', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'WORK_OF_ART', 'EVENT', 'LAW', 'LANGUAGE'}:
                        continue

                    output = string[start:ent.start_char].strip()
                    if len(output) > 0:
                        fout.write(output.replace(' ', '\n') + '\n')

                    ent_name = string[ent.start_char:ent.end_char].strip()
                    toks = ent_name.split()
                    if len(toks) == 0:
                        continue

                    if toks[0] == 'the' or toks[0] == 'The':
                        fout.write(toks[0] + '\n')
                        toks = toks[1:]

                    if len(toks) == 0:
                        continue

                    print_s = False
                    if toks[-1] == "'s":
                        print_s = True
                        toks = toks[:-1]

                    ent_name = ' '.join(toks)
                    if len(ent_name) == 0:
                        continue

                    # print(ent, ent.label_, ent.start_char, ent.end_char)
                    fout.write(toks[0] + '\tB\t' + ent_name + '\tUNK\tUNK\tUNK\tUNK\n')
                    for i in range(1,len(toks)):
                        fout.write(toks[i] + '\tI\t' + ent_name + '\tUNK\tUNK\tUNK\tUNK\n')

                    if print_s:
                        fout.write("'s\n")

                    start = ent.end_char

                output = string[start:].strip()
                if len(output) > 0:
                    fout.write(output.replace(' ', '\n') + '\n')
                fout.write('\n')

                # count += 1
                # if count % 100 == 0:
                #     print(count, end='\r')
                # if count > 1000:
                #     break


if __name__ == "__main__":
    filelist = [join(indir_path, f) for f in listdir(indir_path) if isfile(join(indir_path, f))]
    split = []
    step = ceil(len(filelist) / nthreads)
    for i in range(nthreads):
        split.append(filelist[i * step : min(len(filelist), (i + 1) * step)])

    for i, fl in enumerate(split):
        p = Process(target=process, args=(fl,i))
        p.start()

