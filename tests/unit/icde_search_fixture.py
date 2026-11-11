"""Small deterministic fixture for search-only integration; no index construction."""
import math
import pathlib
import random
import struct
import sys
p = pathlib.Path(sys.argv[1])
p.mkdir(parents=True, exist_ok=True)
n = 8192
rng = random.Random(19)
def array(f, code, values):
    f.write(struct.pack('<Q', len(values)))
    f.write(struct.pack('<' + code * len(values), *values))
def vectors(path, values):
    with path.open('wb') as f:
        f.write(struct.pack('<II', len(values), 4))
        for row in values:
            f.write(struct.pack('<4f', *row))
def vec(i):
    return [i / n, math.sin(i * .1), math.cos(i * .1), (i % 17) / 17]
vectors(p/'base.bin', [vec(i) for i in range(n)])
vectors(p/'queries.bin', [vec(i) for i in (27, 2040, 52, 6000)])
with (p/'attributes.bin').open('wb') as f:
    for i in range(n):
        f.write(struct.pack('<3f', i, i % 8, i % 17))
(p/'queries.tsv').write_text('A0_Range\tA1_Categorical\tA2_Range\n0,60\t\t\n0,8192\t\t\n0,70\t\t\n4096,8192\t0,2,4,6\t\n')
node = [0]
offsets = [0]
edges = []
with (p/'attr_equidepth_root.txt').open('w') as f:
    for i in range(n):
        ns = [(i+j)%n for j in range(1,9)] + [(i-j)%n for j in range(1,9)]
        ns += rng.sample(range(n), 16)
        f.write(str(i)+' '+ ' '.join(map(str, ns))+'\n')
        edges.extend(ns)
        offsets.append(len(edges))
        node.append(i+1)
(p/'attr_equidepth_root_upper.txt').write_text('0\n0\n17\n'+str(n)+'\n'+' '.join(['0']*n)+'\n0\n')
with (p/'pathseer.search').open('wb') as f:
    f.write(struct.pack('<Q4i',0x3147524545534650,0,0,16,32))
    for code, values in [('Q',node),('Q',offsets),('i',[16]*n),('i',edges)]:
        array(f,code,values)
with (p/'cgif.search').open('wb') as f:
    f.write(struct.pack('<Q',0x3152464947434650))
    array(f,'Q',list(range(0,n+1,256)))
    array(f,'i',list(range(n)))
    array(f,'i',[i//256 for i in range(n)])
print(p)
