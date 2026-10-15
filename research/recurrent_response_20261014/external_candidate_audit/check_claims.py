"""Independent counterexamples; not market forecasting or agent-result reproduction."""
import json,sys,subprocess
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent
ROOT=R.parents[2]
sys.path.insert(0,str(R.parent));import run as base
from prefix_loader import read_prefix

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 cited='c2a9d78a745128ced71f02c411df75ba585bb9fd'
 path='research/recurrent_response_20261014/predictive_support_gate'
 tree=subprocess.check_output(['git','-C',str(ROOT),'ls-tree','-r','--name-only',cited,'--',path],text=True).strip()
 if tree:raise ValueError('version premise changed')
 # Exact sign reversal for a state direction need not make it an eigenvector of M.
 a=np.eye(2);b=np.array([[-1.,0.],[3.,2.]]);u=np.array([1.,0.]);m=(a@b.T+b@a.T)/2
 if not np.array_equal(b.T@u,-a.T@u):raise ValueError('sign reversal fixture')
 vals,vecs=np.linalg.eigh(m);p=(vecs*np.where(vals>0,vals/vals[-1],0))@vecs.T
 retained=float(np.linalg.norm(p@u)**2);rayleigh=float(u@m@u)
 if rayleigh>=0 or retained<=0.1:raise ValueError('counterexample not established')
 # PyTorch retention test: candidate n=0, reset irrelevant, gate bias only.
 gate=[]
 for bias in [.8,-1.2]:
  g=torch.nn.GRU(1,1,batch_first=True).double()
  with torch.no_grad():
   for param in g.parameters():param.zero_()
   g.bias_ih_l0[1]=bias
  _,h=g(torch.zeros(1,1,1,dtype=torch.float64),torch.ones(1,1,1,dtype=torch.float64))
  gate.append(float(h.detach()[0,0,0]))
 if not gate[0]>gate[1]:raise ValueError('GRU convention test')
 # Dense recurrence allows short-loss gradients into slow-row parameters even
 # when final slow-state gradients are zero. This is independent of gate priors.
 torch.manual_seed(17);g=torch.nn.GRU(1,16,batch_first=True).double();x=torch.randn(4,6,1,dtype=torch.float64)
 z=g(x)[0][:,-1];z.retain_grad();z.register_hook(lambda v:torch.cat([v[:,:8],torch.zeros_like(v[:,8:])],dim=1))
 (z[:,:8]**2).mean().backward()
 rows=torch.cat([torch.arange(k*16+8,k*16+16) for k in range(3)])
 slow=float(g.weight_hh_l0.grad[rows].norm());last=float(z.grad[:,8:].norm())
 if last!=0 or slow<=1e-8:raise ValueError('recurrent gradient counterexample')
 # Differentiating the mean preserves a bypass, not simply g_i P.
 h=torch.arange(8,dtype=torch.float64).reshape(4,2).clone().requires_grad_();pm=torch.diag(torch.tensor([1.,0.],dtype=torch.float64));mean=h.mean(0);out=mean+(h-mean)@pm
 out[:,1].sum().backward();mean_bypass=float(h.grad[:,1].norm())
 if mean_bypass!=2.:raise ValueError('center gradient fixture')
 # The literal formula max(0, lambda/lambda_max) requires a positive-max guard.
 neg=np.array([-2.,-1.]);literal=np.maximum(0,neg/neg.max())
 if literal.max()<=1:raise ValueError('negative spectrum formula fixture')
 # Train-only empirical horizon moments, no forbidden suffix or price release.
 rows,audit=read_prefix(ROOT/'data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31')
 prices=np.array([float(v['author_reported_price']) for v in rows]);ratios=[]
 for block in range(12):
  end=459+40*block;ids,_,y,tr,va,_=base.arrays(prices,end)
  ratios.append(dict(end=end,second_moment_ratio_20_to_1=float(np.mean(y[tr,19]**2)/np.mean(y[tr,0]**2)),variance_ratio_20_to_1=float(np.var(y[tr,19])/np.var(y[tr,0]))))
 result=dict(scope='independent mathematical examples and train-prefix diagnostics; not supplied sandbox reproduction',cited_commit_lacks_predictive_support_directory=True,cb_spg_counterexample=dict(a=a.tolist(),b=b.tolist(),state_direction=u.tolist(),eigenvalues=vals.tolist(),negative_rayleigh=rayleigh,positive_part_retained_energy=retained),pytorch_gate_retention=dict(positive_point8=gate[0],negative_point1point2=gate[1],larger_positive_means_more_retention=True),routing_counterexample=dict(final_slow_gradient_norm=last,slow_recurrent_row_gradient_norm=slow,slow_rows_not_isolated=True),mean_bypass_gradient_norm=mean_bypass,literal_negative_spectrum_weights=literal.tolist(),train_moment_ratios=ratios,prefix=audit,passed=True)
 (R/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
