import unittest
import numpy as np
from episodes import features,label_signal,spread,build
from core import kernels,gains,magnitude,select
class Tests(unittest.TestCase):
 def test_recency_and_exact_annotation_intervals(self):
  a=np.array([[1,0],[-1,0],[0,1.]])
  b=a*.7;c=a*.3
  f=features([a,b,c],.8)
  self.assertEqual(len(f['deltas']),2)
  self.assertAlmostEqual(f['deltas'][0],spread(b)-spread(a))
  self.assertAlmostEqual(f['deltas'][1],spread(c)-spread(b))
  self.assertAlmostEqual(f['trend'],sum(x*y for x,y in zip(f['beta'],f['deltas'])))
  self.assertGreater(f['beta'][1],f['beta'][0])
  self.assertLess(label_signal(['narrowing','narrowing'],f['beta']),0)
 def test_stable_exactly_zero(self):
  a=np.eye(3);self.assertEqual(features([a,a,a],.8)['trend'],0.)
 def test_magnitude_independent_solve(self):
  x=np.random.default_rng(3).normal(size=(8,4));x[7]=x[0]
  for k in kernels(x,1):
   s=[0,2];g=gains(k,s)
   for j in [1,3,4,5,6,7]:self.assertAlmostEqual(g[j],magnitude(k[np.ix_(s+[j],s+[j])])-magnitude(k[np.ix_(s,s)]),places=6)
 def test_budget(self):
  k=kernels(np.eye(5),1);s,t,_=select(k,np.array([10]*5),np.arange(5.),25,np.ones(6)/6,.75)
  self.assertLessEqual(t,25);self.assertEqual(len(s),len(set(s)))
if __name__=='__main__':unittest.main()
