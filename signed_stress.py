"""Minimum cancellation share across sign conventions of equicorrelation."""
from itertools import product
from sympy import Matrix, Poly, Rational, diag, eye, solve, sqrt, symbols

from study import inputs


def signed_radii(prior, innovation, variances):
    rho = symbols("rho", real=True)
    diagonal = diag(*variances)
    radii = [None] * 3
    for second, third in product([1, -1], repeat=2):
        signs = (1, second, third)
        loading = Matrix([sign * sqrt(v) for sign, v in zip(signs, variances)])
        total = prior + (1-rho)*diagonal + rho*loading*loading.T
        for i in range(3):
            numerator = total.copy()
            numerator[:, i] = innovation
            poly = Poly(numerator.det(), rho)
            if poly.is_zero:
                radii[i] = 0
                continue
            for root in solve(poly.as_expr(), rho):
                if root.is_real and -Rational(1, 2) <= root <= 1:
                    # Eigenvalues are 1-rho (twice) and 1+2*rho.
                    share = root if root >= 0 else -2*root
                    if radii[i] is None or share < radii[i]:
                        radii[i] = share
    return radii


if __name__ == "__main__":
    _, _, _, prior, innovation = inputs()
    v = list(prior.diagonal())
    for label, scale in [("half", [Rational(1,2)]*3), ("reference", [1]*3),
                         ("double", [2]*3), ("view-specific", [3,1,Rational(7,13)])]:
        radii = signed_radii(prior, innovation, [a*b for a,b in zip(v, scale)])
        print(label, [None if r is None else round(float(r), 8) for r in radii])
    analytic = signed_radii(eye(3), Matrix([1, Rational(3,2), -Rational(3,2)]), [1]*3)
    assert analytic == [1, None, None], analytic
    print("analytic", analytic)
