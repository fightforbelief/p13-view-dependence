import unittest
import sympy as s
from envelope import endpoints

class ExposureEnvelope(unittest.TestCase):
    def test_known_range(self):
        # With A=I, d=c=e1 the inverse diagonal lies between 1/2
        # (diagonal covariance) and 3/4 (rank-one equal loadings).
        a=s.eye(3)
        c=s.Matrix([1,0,0])
        for family in ("full","signed"):
            lo,hi=endpoints(a,c,[1]*3,c,1,family=family)
            self.assertLessEqual(lo[0],.5)
            self.assertGreaterEqual(lo[1],.5)
            self.assertLessEqual(hi[0],.75)
            self.assertGreaterEqual(hi[1],.75)
            self.assertLessEqual(hi[1]-hi[0],.00025)

    def test_three_factor_support_point(self):
        from study import feasible_gain
        a = s.eye(4)
        x = s.ones(4,1)
        d = s.Matrix([2,1,1,1])
        g = s.Matrix([[1,0,0],[0,1,0],[0,-s.Rational(1,2),s.sqrt(3)/2],
                      [0,-s.Rational(1,2),-s.sqrt(3)/2]])
        k = g*g.T
        self.assertEqual(k.rank(),3)
        self.assertEqual((a+k)*x,d)
        self.assertTrue(feasible_gain(a,d,[1]*4,x))
        # Tangent quadratic identity: f(x+z)=c'z+z'z+z1**2.
        z = s.Matrix(s.symbols("z:4"))
        y = x+z
        f = (d[0]-y[0])**2-y.dot(d-y)
        self.assertEqual(s.expand(f-z.dot(s.Matrix([-2,1,1,1]))-z.dot(z)-z[0]**2),0)

    def test_interior_covariances_and_analytic_sign(self):
        a=s.eye(3)
        d=s.Matrix([1,s.Rational(3,2),-s.Rational(3,2)])
        c=s.Matrix([1,0,0])
        low,high=endpoints(a,d,[1]*3,c,s.Rational(16,17))
        self.assertLessEqual(low[0],0)
        self.assertGreaterEqual(low[1],0)
        c=s.Matrix([2,-1,1])
        h=s.Rational(3,4)
        low,high=endpoints(a,d,[1,2,3],c,h)
        for scale in (0,s.Rational(1,2),1):
            # A full-rank dependent component from an independent Gram construction.
            g=s.Matrix([[1,2,1],[1,-1,2],[2,1,1]])
            b=g*g.T
            norm=s.diag(*[s.sqrt(s.Rational(v,b[i,i])) for i,v in enumerate((1,2,3))])
            k=(1-h*scale)*s.diag(1,2,3)+h*scale*norm*b*norm
            value=float(c.dot((a+k).inv()*d))
            self.assertLessEqual(low[0],value)
            self.assertGreaterEqual(high[1],value)

    def test_closed_form_support_certificate(self):
        from four_views import certificate
        cert = certificate(s.eye(4),s.Matrix([2,1,1,1]),[1]*4,s.Matrix([-2,1,1,1]))
        self.assertEqual(cert['active_view'],1)
        self.assertLessEqual(cert['enclosure'][0],1)
        self.assertGreaterEqual(cert['enclosure'][1],1)
