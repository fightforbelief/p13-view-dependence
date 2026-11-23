"""Post-design diagnostic. Clip manufacturing weight to [0, .4] at a .2 benchmark."""
import json
from pathlib import Path
from statistics import median


def capped(pair):
    return [[min(.2,max(-.2,x)) for x in end] for end in pair]


def width(pair):
    lo,hi=pair
    return hi[0]-lo[1],hi[1]-lo[0]


def summarize(rows):
    gaps=[]
    for row in rows:
        f=width(capped(row['full']))
        s=width(capped(row['signed']))
        gaps.append([f[0]-s[1],f[1]-s[0]])
    return { 'median_paired_extra_pp':[100*median(g[i] for g in gaps) for i in (0,1)],
             'extra_above_one_pp_lower_count':sum(g[0]>.01 for g in gaps),
             'extra_above_one_pp_upper_count':sum(g[1]>.01 for g in gaps),
             'same_point_count':sum(width(capped(r['full']))==(0.,0.) for r in rows)}


if __name__=='__main__':
    rows=json.loads(Path('historical_results.json').read_text())['origins']
    for label,group in [('early',rows[:20]),('late',rows[20:]),('all',rows)]:
        print(label,summarize(group))
