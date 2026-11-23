"""Reproduce the frozen historical audit and magnitude bounds on the printed example."""
import csv
import hashlib
import json
import platform
from pathlib import Path
from statistics import median

import numpy as np
import sympy as sp
from envelope import endpoints
from study import inputs

P = sp.Matrix([[1,-1,0,0,0],[0,-1,1,0,0],[0,-1,0,1,0]])
W = sp.ones(5,1)/5
RISK = 3


def moments(rows):
    mean = rows.T*sp.ones(rows.rows,1)/rows.rows
    cov = (rows.T*rows-rows.rows*mean*mean.T)/(rows.rows-1)
    return mean,cov


def dependence_share(cov):
    """Smallest hundredth h for which cov-(1-h)diag(cov) is PSD."""
    diagonal = sp.diag(*cov.diagonal())
    lo,hi = -1,100
    assert cov.is_positive_semidefinite and min(diagonal.diagonal())>0
    while hi-lo>1:
        mid=(hi+lo)//2
        if (cov-(1-sp.Rational(mid,100))*diagonal).is_positive_semidefinite:
            hi=mid
        else:
            lo=mid
    return sp.Rational(hi,100)


def cases():
    source = Path('data/french_five_monthly.csv')
    with source.open() as f:
        rows=list(csv.reader(f))[1:]
    dates=[row[0] for row in rows]
    assert dates==[f'{year}{month:02}' for year in range(1970,2025) for month in range(1,13)]
    assets=sp.Matrix([[sp.Rational(val)/100 for val in row[1:]] for row in rows])
    assert all(v > -sp.Rational(9999,10000) for v in assets)
    spreads=assets*P.T
    errors=[]
    forecasts={}
    for i in range(60,len(rows)+1):
        forecast=sp.ones(1,60)*spreads[i-60:i,:]/60
        forecasts[i]=forecast.T
        if i < len(rows):
            errors.append(list(spreads[i,:]-forecast))
    errors=sp.Matrix(errors)
    c=P[:,1]/RISK
    for year in range(1980,2026):
        end=12*(year-1970)
        _,sigma=moments(assets[end-60:end,:])
        _,cov=moments(errors[end-120:end-60,:])
        a=P*sigma*P.T
        assert a.is_positive_definite
        d=forecasts[end]-RISK*P*sigma*W
        h=dependence_share(cov)
        v=list(cov.diagonal())
        out={"year":year,"h":float(h),"sample_tilt":float(c.dot((a+cov).inv()*d)),
             "diagonal_tilt":float(c.dot((a+sp.diag(*v)).inv()*d))}
        for key,share,family in [('full',h,'full'),('signed',h,'signed'),('half',h/2,'full')]:
            out[key]=endpoints(a,d,v,c,share,family=family)
        assert out['full'][0][0]<=out['sample_tilt']<=out['full'][1][1]
        yield out


def width(row,key):
    low,high=row[key]
    return [high[0]-low[1],high[1]-low[0]]


def interval_status(row,key):
    low,high=row[key]
    if low[1]<0<high[0]: return "both"
    if low[0]>0 or high[1]<0: return "one"
    return "unresolved"


def summary(rows):
    ans={'n':len(rows),'median_h':median(r['h'] for r in rows)}
    for key in ('full','signed','half'):
        ans[key]={'median_width_pp':[100*median(width(r,key)[i] for r in rows) for i in (0,1)],
                  'sign_counts':{s:sum(interval_status(r,key)==s for r in rows)
                                 for s in ('both','one','unresolved')}}
    gaps=[[width(r,'full')[0]-width(r,'signed')[1],
           width(r,'full')[1]-width(r,'signed')[0]] for r in rows]
    ans['added_width_median_pp']=[100*median(g[i] for g in gaps) for i in (0,1)]
    ans['extra_crossings']=[r['year'] for r in rows if interval_status(r,'full')=='both'
                            and interval_status(r,'signed')=='one']
    return ans


def study():
    rows=[]
    for row in cases():
        rows.append(row)
        print(row['year'],row['h'],width(row,'full'),flush=True)
    raw,p,risk,a,d=inputs()
    examples={}
    for multiple in (1,2):
        v=[multiple*x for x in a.diagonal()]
        examples[multiple]={raw["assets"][j]:endpoints(a,d,v,p[:,j]/risk,sp.Integer(1))
                            for j in (6,1,2)}
    results={'examples':examples,'origins':rows,
             'summaries':{label:summary(group) for label,group in [
                 ('early',rows[:20]),('late',rows[20:]),('all',rows)]},
             'sha256':{path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
                       for path in ('data/french_five_monthly.csv','historical.py','envelope.py')},
             'python':platform.python_version(),'system':platform.system(),
             'numpy':np.__version__,'sympy':sp.__version__}
    path=Path('historical_results.json')
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(results,indent=2)+'\n')
    tmp.replace(path)
    print(json.dumps(results['summaries']),flush=True)


if __name__=='__main__':
    study()
