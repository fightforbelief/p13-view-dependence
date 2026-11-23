import random
import unittest
import sympy as sp
from study import Matrix, feasible_gain, radius, zero_possible

class Certificates(unittest.TestCase):
    def test_joint_membership(self):
        gen = random.Random(1301)
        for k in (2, 3, 6, 10):
            f = Matrix(k, 4, lambda i,j:gen.randint(-3,3))
            cov = f*f.T
            x = Matrix([gen.randint(-4,4) for _ in range(k)])
            a = sp.eye(k)
            self.assertTrue(feasible_gain(a,(a+cov)*x,cov.diagonal(),x))

    def test_null_polygon(self):
        a = sp.eye(3)
        for x, expected in [([1,1,1],True),([3,1,1],False),([2,1,1],True),([0,0,0],True)]:
            self.assertEqual(feasible_gain(a,Matrix(x),[1]*3,Matrix(x)),expected)
        self.assertFalse(feasible_gain(a,Matrix([1,0,0]),[1]*3,sp.zeros(3,1)))

    def test_known_threshold(self):
        # A third view can transmit dependence even with zero innovation.
        # For independent nuisance errors at h=0 there is no cancellation.
        a = sp.eye(3)
        d = Matrix([1,2,0])
        self.assertFalse(zero_possible(a,d,[1]*3,0,0))
        self.assertTrue(zero_possible(a,d,[1]*3,0,1))
        self.assertIsNone(radius(a,Matrix([1,0,0]),[1]*3,0))
        self.assertEqual(radius(a,Matrix([0,1,0]),[1]*3,0),[0,0])

    def test_missed_by_both_restricted_sweeps(self):
        from study import Q, structured_zero
        a, v = sp.eye(3), [1,1,1]
        d = Matrix([1,Q(3,2),-Q(3,2)])
        pairs = [(0,1),(0,2),(1,2)]
        self.assertFalse(structured_zero(a,d,v,0,pairs,Q(-1,2)))
        for pair in pairs:
            self.assertFalse(structured_zero(a,d,v,0,[pair],-1))
        self.assertTrue(zero_possible(a,d,v,0,Q(16,17)))
        self.assertEqual(radius(a,d,v,0),[9411,9412])
        omega = Matrix([[1,Q(7,10),-Q(7,10)],[Q(7,10),1,0],[-Q(7,10),0,1]])
        self.assertTrue(omega.is_positive_definite)
        self.assertEqual(((a+omega).inv()*d)[0],-Q(5,151))
