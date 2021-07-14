#!python3

import types
import numpy as np

def extract_best(name, idcs=range(10, 16), rownum=7*4):
    p = dict()
    for i in idcs:
        p[i] = types.SimpleNamespace()
        path = f'out-nn/{name}-s{i}.{i}-measures.dat'
        rows = np.loadtxt(path)
        p[i].iters, rows = np.split(rows, [1], axis=1)
        (p[i].sdr_dict, p[i].sdr_dir,
         p[i].sir_dict, p[i].sir_dir,
         p[i].sar_dict, p[i].sar_dir) = \
            np.split(rows, 6, axis=1)
        try:
            p[i].sdrmean = np.mean(p[i].sdr_dir[rownum])
        except IndexError as e:
            print(f"seed {i}:", e)
            p[i].sdrmean = -np.infty
    bestidx = max(p.items(), key=lambda it: np.mean(it[1].sdrmean))[0]

    return p, bestidx

if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
    rownum = 7*4
    #p, bestidx = extract_best('mozart-cl/mozart-real', rownum=rownum)
    #p, bestidx = extract_best('mozart/mozart-0+10abs+1+10-mod1e-4', rownum=rownum)
    #p, bestidx = extract_best('urmp/03-0+10+1+10', rownum=rownum)
    #p, bestidx = extract_best('urmp/09-real', rownum=rownum)
    #p, bestidx = extract_best('urmp/10', rownum=rownum)
    #p, bestidx = extract_best('urmp/11-8+1+10', rownum=rownum)
    #p, bestidx = extract_best('duan/acous-mod', rownum=rownum)
    #p, bestidx = extract_best('duan/synth2-sd32', rownum=rownum)
    #p, bestidx = extract_best('duan/synth3', idcs=range(10, 12), rownum=7*4*3)
    #p, bestidx = extract_best('urmp/03-oracle', idcs=range(10, 14), rownum=rownum)
    #p, bestidx = extract_best('urmp/03-init', idcs=(10, 12,13), rownum=rownum)
    #p, bestidx = extract_best('urmp/11-oracle', idcs=range(10, 14), rownum=rownum)
    #p, bestidx = extract_best('urmp/11-init', idcs=range(10, 14), rownum=rownum)

    print("sdrmean", [p[i].sdrmean for i in p])
    print("bestidx", bestidx)
    print("performance")
    print(p[bestidx].sdr_dir[rownum],
          p[bestidx].sir_dir[rownum],
          p[bestidx].sar_dir[rownum])
    print(p[bestidx].sdr_dict[rownum],
          p[bestidx].sir_dict[rownum],
          p[bestidx].sar_dict[rownum])
