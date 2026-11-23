"""Exact arithmetic throughout certification; floats only for reporting."""
import hashlib
import json
import platform
from pathlib import Path

import sympy as sp
from sympy import Matrix, Rational as Q

T = sp.Symbol("t", real=True)


def inputs():
    raw = json.loads(Path("benchmark.json").read_text())
    cov = Matrix(raw["covariance_in_millionths"]) / 10**6
    market = Matrix(raw["market_in_basis_points"]) / 10000
    p = Matrix(raw["P_in_tenths"]) / 10
    q = Matrix(raw["q_in_basis_points"]) / 10000
    risk = Q(raw["risk_aversion"])
    return raw, p, risk, p * cov * p.T, q - risk * p * cov * market


def feasible_gain(a, d, variances, x):
    """Exact candidate test with rational inputs (Theorem 1)."""
    r = d - a * x
    energy = x.dot(r)
    if energy < 0 or (energy == 0 and any(r)):
        return False
    squared = list(variances) if energy == 0 else [
        v - u**2 / energy for v, u in zip(variances, r)]
    if min(squared) < 0:
        return False
    lengths = [abs(w) * sp.sqrt(v) for w, v in zip(x, squared)]
    return bool(2 * max(lengths) <= sum(lengths))


def sign_polynomial(a, d, v, i, h):
    j, k = [n for n in range(3) if n != i]
    off = a[j, k] + T
    u, w = a[j, j] + v[j], a[k, k] + v[k]
    det = u*w - off**2
    n1, n2 = w*d[j] - off*d[k], u*d[k] - off*d[j]
    center = d[i]*det - a[i, j]*n1 - a[i, k]*n2
    spread2 = h*v[i]*(h*v[j]*n1**2 + 2*T*n1*n2 + h*v[k]*n2**2)
    return sp.Poly(center**2 - spread2, T), h*sp.sqrt(v[j]*v[k])


def zero_possible(a, d, v, i, h):
    poly, bound = sign_polynomial(a, d, v, i, h)
    if poly.eval(0) <= 0:
        return True
    # Isolate each real root by exact rational endpoints. Refine only
    # when a root's isolating interval overlaps the admissible boundary.
    for root in poly.real_roots(radicals=False):
        if -bound <= root and root <= bound:
            return True
    return False


def radius(a, d, v, i):
    if zero_possible(a, d, v, i, 0):
        return [0, 0]
    if not zero_possible(a, d, v, i, 1):
        return None
    low, high = 0, 10000
    while high - low > 1:
        mid = (low + high) // 2
        if zero_possible(a, d, v, i, Q(mid, 10000)):
            high = mid
        else:
            low = mid
    return [low, high]


def witness(a, d, v, i, h):
    """Find a rational grid location with strict opposite-sign feasibility."""
    j, k = [n for n in range(3) if n != i]
    poly, bound = sign_polynomial(a, d, v, i, h)
    for step in range(-100, 101):
        t = bound * Q(step, 100)
        if poly.eval(t) >= 0:
            continue
        idx = [j,k]
        b = Matrix([[h*v[j], t], [t, h*v[k]]])
        y = (a.extract(idx,idx) + sp.diag((1-h)*v[j], (1-h)*v[k]) + b).inv() * d.extract(idx,[0])
        base = (a + sp.diag(*v)).inv() * d
        cross = sp.sign(base[i]) * sp.sqrt(h*v[i] / y.dot(b*y)) * b*y
        omega = sp.diag(*v)
        omega[j,k] = omega[k,j] = t
        for n, val in zip(idx, cross):
            omega[n,i] = omega[i,n] = val
        # Evaluate an explicit completion with ample precision after its construction.
        gain = (a + omega.evalf(45)).inv() * d
        assert base[i] * gain[i] < 0
        return {"h": float(h), "nuisance_corr_fraction":step/100,
                "correlations_12_13_23":[float(omega[l,m]/sp.sqrt(v[l]*v[m])) for l,m in [(0,1),(0,2),(1,2)]],
                "gain":[float(n) for n in gain]}
    return None


def structured_zero(a, d, v, i, pairs, low):
    """Cramer numerator along an equicorrelation or single-pair path."""
    m = a + sp.diag(*v)
    for j,k in pairs:
        m[j,k] = m[k,j] = m[j,k] + T*sp.sqrt(v[j]*v[k])
    m[:,i] = d
    poly = sp.Poly(m.det(), T)
    locations = [low, 1]
    if poly.degree() == 2:
        vertex = -poly.nth(1)/(2*poly.nth(2))
        if low < vertex < 1:
            locations.append(vertex)
    values = [poly.eval(t) for t in locations]
    return bool(min(values) <= 0 and max(values) >= 0)


def run():
    raw, p, risk, a, d = inputs()
    v = list(a.diagonal())
    gain = (a + sp.diag(*v)).inv()*d
    tilts = 10000*p.T*gain/risk
    diff = max(abs(x-y) for x,y in zip(tilts,raw["table6_tilts_in_basis_points"]))
    assert diff < 5, diff
    results = {}
    for label, factors in {"half_error_variance":[Q(1,2)]*3, "table6":[1]*3,
                           "double_error_variance":[2]*3,
                           "view_specific":[3,1,Q(7,13)]}.items():
        var = [x*y for x,y in zip(v,factors)]
        brackets = [radius(a,d,var,i) for i in range(3)]
        results[label] = {"radius_units_0.0001":brackets}
        pairs = [(0,1),(0,2),(1,2)]
        results[label]["equicorrelation_zero"] = [
            structured_zero(a,d,var,i,pairs,Q(-1,2)) for i in range(3)]
        results[label]["any_single_pair_zero"] = [
            any(structured_zero(a,d,var,i,[pair],-1) for pair in pairs) for i in range(3)]
        results[label]["strict_witnesses"] = [
            witness(a,d,var,i,Q(1)) if b is not None else None for i,b in enumerate(brackets)]
        for bracket, example in zip(brackets, results[label]["strict_witnesses"]):
            assert bracket is None or example is not None
        # Independent reconstruction from the printed prior instead of reverse optimization.
        printed_prior = Matrix([8,67,641,408,743,370,480,660])/10000
        alternate_d = Matrix(raw["q_in_basis_points"])/10000 - p*printed_prior
        results[label]["rounded_prior_radius"] = [radius(a,alternate_d,var,i) for i in range(3)]
    return {"python":platform.python_version(), "platform":platform.system(), "sympy":sp.__version__,
            "sha256":{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ["benchmark.json","study.py","test_study.py"]},
            "diagonal_tilts_bp":[float(x) for x in tilts], "max_baseline_difference_bp":float(diff),
            "cases":results}

if __name__ == "__main__":
    output = json.dumps(run(), indent=2)
    Path("results.json").write_text(output+"\n")
    print(output)
