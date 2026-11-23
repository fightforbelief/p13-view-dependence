"""Exact single-ellipsoid support certificates for the four-spread extension."""
import csv
import json
from pathlib import Path
from math import sqrt

import sympy as s
from historical import moments, dependence_share
from envelope import Box, down, up, scalar


def candidate(a,d,v,c,i):
    """Maximizer over the i-th outer ellipsoid, given as m + gamma*u."""
    row=a[:,i]
    hess=v[i]*a+row*row.T
    g=row*d[i]+v[i]*d/2
    inv=hess.inv()
    m,u=inv*g,inv*c
    gamma2=(g.dot(m)-d[i]**2)/c.dot(u)
    return m,u,gamma2


def certificate(a,d,v,c):
    if len(v)!=4:
        raise ValueError("Four variances required")
    uncertain=[]
    for i in range(len(v)):
        m,u,gamma2=candidate(a,d,v,c,i)
        if gamma2<=0:
            continue
        gamma=scalar(s.sqrt(gamma2))
        def linear(p,q):
            return [scalar(x)+gamma*scalar(y) for x,y in zip(p,q)]
        x=linear(m,u)
        residual=linear(d-a*m,-a*u)
        energy=Box(0.,0.)
        for xi,ri in zip(x,residual):
            energy=energy+xi*ri
        if energy.hi<=0:
            continue
        if energy.lo<=0:
            uncertain.append(i+1)
            continue
        lengths=[]
        inconclusive=False
        for j in range(len(v)):
            if i==j: # Its residual is exactly zero by the support construction.
                lengths.append(Box(0.,0.))
                continue
            rad=scalar(v[j])*energy-residual[j]*residual[j]
            if rad.hi<=0:
                break
            if rad.lo<=0:
                inconclusive=True
                break
            mag=Box(max(0.,x[j].lo,-x[j].hi),max(abs(x[j].lo),abs(x[j].hi)))
            lengths.append(mag*Box(down(sqrt(rad.lo)),up(sqrt(rad.hi))))
        if len(lengths)!=len(v):
            if inconclusive:
                uncertain.append(i+1)
            continue
        total=Box(0.,0.)
        for length in lengths:
            total=total+length
        if 2*max(length.hi for length in lengths)<total.lo:
            bound=scalar(c.dot(m))+gamma*scalar(c.dot(u))
            return {'rank':3,'active_view':i+1,'enclosure':[float(bound.lo),float(bound.hi)]}
        # Prove failure of strict polygon closure, including equality when identified.
        for j in range(4):
            others=Box(0.,0.)
            for k,length in enumerate(lengths):
                if k!=j:
                    others=others+length
            if lengths[j].lo>=others.hi:
                break
        else:
            uncertain.append(i+1)
    if uncertain:
        return {'rank':None,'undecided_views':uncertain}
    return {'rank_at_most':2}


def run():
    with Path('data/french_five_monthly.csv').open() as f:
        rows=list(csv.reader(f))[1:]
    assets=s.Matrix([[s.Rational(x)/100 for x in r[1:]] for r in rows])
    p=s.Matrix([[int(i==j)-int(j==1) for j in range(5)] for i in (0,2,3,4)])
    spreads=assets*p.T
    fore=[s.ones(1,60)*spreads[i-60:i,:]/60 for i in range(60,assets.rows+1)]
    errors=s.Matrix([list(spreads[i,:]-fore[i-60]) for i in range(60,assets.rows)])
    report=[]
    for year in range(1980,2026):
        end=12*(year-1970)
        _,sig=moments(assets[end-60:end,:])
        _,cov=moments(errors[end-120:end-60,:])
        h=dependence_share(cov)
        v=list(cov.diagonal())
        a=p*sig*p.T+(1-h)*s.diag(*v)
        d=fore[end-60].T-3*p*sig*s.ones(5,1)/5
        c=p[:,1]/3
        record={'year':year}
        for label,direction in [('min',-1),('max',1)]:
            cert=certificate(a,d,[h*x for x in v],c*direction)
            if 'enclosure' in cert and direction==-1:
                l,r=cert['enclosure']
                cert['enclosure']=[-r,-l]
            record[label]=cert
        report.append(record)
    Path('four_view_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print({side:[sum(r[side].get('rank')==3 for r in report),sum('rank_at_most' in r[side] for r in report),sum('undecided_views' in r[side] for r in report)] for side in ['min','max']})

if __name__=='__main__':
    run()
