import unittest,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from models import *
ROOT=Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(3);self.p=20+np.cumsum(rng.normal(0,.1,350));self.date=pd.date_range('2010-01-01',periods=350).values;self.d=make_data(self.p,self.date,200)
 def test_exact_source(self):
  b=(ROOT/'raw_private/GCP.csv').read_bytes();self.assertEqual(hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),'23b1ffc5c68ded3c7b3c03dd46242788f0db52f7')
 def test_maturity(self):
  for t in range(100,300):self.assertLessEqual(training_indices(t).max()+20,t)
 def test_causal_features(self):
  p=self.p.copy();p[251:]+=np.arange(99)*1000;d2=make_data(p,self.date,200);np.testing.assert_array_equal(self.d.x[:251],d2.x[:251]);np.testing.assert_array_equal(self.d.scales,d2.scales)
 def test_translation(self):
  d2=make_data(self.p+1000,self.date,200);np.testing.assert_allclose(self.d.x,d2.x,atol=1e-9);np.testing.assert_allclose(self.d.y,d2.y,atol=1e-9)
 def test_equal_capacity(self):
  a=ResponseNet('conditional');b=ResponseNet('free');self.assertEqual(sum(x.numel() for x in a.parameters()),sum(x.numel() for x in b.parameters()))
 def test_tau_bounds_and_zero_initial_forecast(self):
  m=ResponseNet();p,t=m(torch.ones(12,61),torch.ones(12),True);self.assertTrue(torch.all(t>.5) and torch.all(t<40));np.testing.assert_array_equal(p.detach(),np.zeros((12,3)))
 def test_gradient_tau(self):
  m=ResponseNet();m.head.bias.data[1]=1;out=m(torch.ones(3,61),torch.ones(3));out.sum().backward();self.assertGreater(abs(m.head.bias.grad[2].item()),0)
 def test_future_predictions_invariance(self):
  p=self.p.copy();p[261:]+=100;d2=make_data(p,self.date,200)
  a,_,_=neural_forecasts(self.d,'conditional',.001,.01,11,10,2);b,_,_=neural_forecasts(d2,'conditional',.001,.01,11,10,2);np.testing.assert_array_equal(a[200:261],b[200:261])
 def test_repeatability(self):
  a,_,_=neural_forecasts(self.d,'conditional',.001,.01,11,10,2);b,_,_=neural_forecasts(self.d,'conditional',.001,.01,11,10,2);np.testing.assert_array_equal(a,b)
 def test_rw(self):
  a=baseline_forecasts(self.d,'RW',{});np.testing.assert_array_equal(a[200:],0)
 def test_no_target_at_boundary(self):
  for h in H:self.assertTrue(np.isnan(self.d.y[len(self.p)-h:,list(H).index(h)]).all())
 def test_gap_is_past_only(self):
  dates=self.date.copy();dates[261:]+=np.timedelta64(5,'D');d2=make_data(self.p,dates,200);np.testing.assert_array_equal(self.d.x[:261],d2.x[:261])
if __name__=='__main__':unittest.main()
