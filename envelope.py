"""Certified three-view exposure endpoints. All control coefficients are enclosed."""
from functools import lru_cache
from heapq import heappop, heappush
from itertools import count, product

import numpy as np
import sympy as sp

down = lambda x: np.nextafter(x, -np.inf)
up = lambda x: np.nextafter(x, np.inf)


class Box:
    """Intervals of doubles, with directed rounding after each operation."""

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def __add__(self, other):
        return Box(down(self.lo + other.lo), up(self.hi + other.hi))

    def __neg__(self):
        return Box(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        values = [self.lo*other.lo, self.lo*other.hi, self.hi*other.lo, self.hi*other.hi]
        return Box(down(np.minimum.reduce(values)), up(np.maximum.reduce(values)))


def scalar(value):
    """Enclose an exact rational or square root using an exact comparison."""
    value = sp.sympify(value)
    f = float(value)
    lo, hi = float(down(f)), float(up(f))
    if not (sp.Rational(lo) <= value <= sp.Rational(hi)):
        raise ArithmeticError("Failed conversion enclosure")
    return Box(lo, hi)


def exact_array(array):
    entries = [scalar(x) for x in array.flat]
    return Box(np.array([x.lo for x in entries]).reshape(array.shape),
               np.array([x.hi for x in entries]).reshape(array.shape))


@lru_cache(None)
def basis_inverse(degree):
    nodes = [sp.Rational(i,degree) for i in range(degree+1)]
    matrix = sp.Matrix([[sp.binomial(degree,j)*x**j*(1-x)**(degree-j)
                         for j in range(degree+1)] for x in nodes])
    return [[scalar(z) for z in matrix.inv().row(i)] for i in range(degree+1)]


def transform_axis(vals, axis):
    lo, hi = vals.lo.swapaxes(0,axis), vals.hi.swapaxes(0,axis)
    rows = []
    for coeff in basis_inverse(len(lo)-1):
        row = Box(0., 0.)
        for j, weight in enumerate(coeff):
            row = row + weight*Box(lo[j],hi[j])
        rows.append(row)
    return Box(np.stack([r.lo for r in rows]).swapaxes(0,axis),
               np.stack([r.hi for r in rows]).swapaxes(0,axis))


@lru_cache(None)
def chart(sign_b, sign_c):
    t = np.array([sp.Rational(i-3,3) for i in range(7)])
    u, w = np.meshgrid(t,t,indexing="ij")
    def circle(z, sign):
        return sign*(1-z*z)/(1+z*z), sign*2*z/(1+z*z)
    c1,s1 = circle(u,sign_b)
    c2,s2 = circle(w,sign_c)
    return [exact_array(m) for m in [c1,c2,c1*c2+s1*s2,((1+u*u)*(1+w*w))**3]]


def samples(a,d,v,c,correlations,scale):
    """Cramer's numerator and determinant, at the collocation nodes."""
    diag = [scalar(a[i,i]+v[i]) for i in range(3)]
    off = [scalar(a[i,j])+scalar(sp.sqrt(v[i]*v[j]))*r for (i,j),r in
           zip([(0,1),(0,2),(1,2)],correlations)]
    p,q,r = off
    u,b,w = diag
    j00,j11,j22 = b*w-r*r, u*w-q*q, u*b-p*p
    j01,j02,j12 = q*r-p*w, p*r-q*b, p*q-u*r
    det = u*j00+p*j01+q*j02
    adj = [[j00,j01,j02],[j01,j11,j12],[j02,j12,j22]]
    num = Box(0.,0.)
    for i in range(3):
        for j in range(3):
            if c[i] and d[j]:
                num = num+scalar(c[i]*d[j])*adj[i][j]
    return num*scale,det*scale


def controls(num,det):
    values = Box(np.stack([num.lo,det.lo],axis=-1),np.stack([num.hi,det.hi],axis=-1))
    for axis in range(values.lo.ndim-1):
        values = transform_axis(values,axis)
    return np.stack([values.lo[...,0],values.hi[...,0],values.lo[...,1],values.hi[...,1]],axis=-1)


def split(control,axis):
    work = control.swapaxes(0,axis)
    left,right = [],[]
    while len(work)>1:
        left.append(work[0])
        right.append(work[-1])
        avg = (work[1:]+work[:-1])*.5
        avg[...,::2] = down(avg[...,::2])
        avg[...,1::2] = up(avg[...,1::2])
        work = avg
    return (np.stack(left+[work[0]]).swapaxes(0,axis),
            np.stack([work[0]]+right[::-1]).swapaxes(0,axis))


def ratio_bounds(control):
    nl,nh,dl,dh = np.moveaxis(control,-1,0)
    if np.any(dl <= 0):
        return -np.inf,np.inf
    return down(np.min(np.where(nl>=0,nl/dh,nl/dl))), up(np.max(np.where(nh>=0,nh/dl,nh/dh)))


def maximize(patches,tol):
    heap,ids = [],count()
    best = -np.inf
    def add(patch,depth):
        nonlocal best
        # Corners of a Bernstein patch are values of its polynomial.
        for index in product(*[(0,-1) for _ in patch.shape[:-1]]):
            best = max(best,ratio_bounds(patch[index])[0])
        ceiling = ratio_bounds(patch)[1]
        if ceiling > best:
            heappush(heap,(-ceiling,next(ids),depth,patch))
    for patch in patches:
        add(patch,0)
    while heap and -heap[0][0] > best+tol:
        _,_,depth,patch = heappop(heap)
        if depth >= 60:
            raise ArithmeticError("Tolerance below enclosure resolution")
        for child in split(patch,depth%(patch.ndim-1)):
            add(child,depth+1)
    return [float(best),float(max(best,-heap[0][0] if heap else best))]


def endpoints(a,d,v,c,h,tol=0.00025,family="full"):
    """Bounds on each endpoint in units of c-transpose times the view coefficients."""
    if a.shape != (3,3) or len(v) != 3 or c.shape != (3,1) or d.shape != (3,1):
        raise ValueError("Expected three views and column vectors")
    if not 0 <= h <= 1 or tol <= 0:
        raise ValueError("Require 0 <= h <= 1 and positive tolerance")
    if not any(c) or not any(d):
        return [[0.,0.],[0.,0.]]
    if h == 0:
        value = scalar(c.dot((a+sp.diag(*v)).inv()*d))
        return [[value.lo,value.hi],[value.lo,value.hi]]
    patches = []
    for b,k in product([1,-1],repeat=2):
        if family == "full":
            r1,r2,r3,scale = chart(b,k)
            corr = [scalar(h)*r for r in (r1,r2,r3)]
        elif family == "signed":
            steps = np.array([h*sp.Rational(i-1,2) for i in range(4)],dtype=object)
            corr = [exact_array(sign*steps) for sign in (b,k,b*k)]
            scale = Box(1.,1.)
        else:
            raise ValueError(family)
        patches.append(controls(*samples(a,d,v,c,corr,scale)))
    high = maximize(patches,tol)
    negatives = []
    for patch in patches:
        neg = patch.copy()
        neg[...,0],neg[...,1] = -patch[...,1],-patch[...,0]
        negatives.append(neg)
    low = maximize(negatives,tol)
    return [[-low[1],-low[0]],high]
