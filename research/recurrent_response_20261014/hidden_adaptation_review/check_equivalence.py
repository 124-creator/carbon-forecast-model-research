from pathlib import Path
import json
import numpy as np
import torch
R=Path(__file__).resolve().parent
torch.manual_seed(17)
w=torch.randn(20,16,dtype=torch.float64);b=torch.randn(20,dtype=torch.float64)
x=torch.randn(7,16,dtype=torch.float64);g=torch.randn(7,16,dtype=torch.float64)
a=torch.randn(16,16,dtype=torch.float64,requires_grad=True);c=torch.randn(16,16,dtype=torch.float64,requires_grad=True)
z=torch.tanh(x@a+g@c)
inside=(x+z)@w.T+b
outside=x@w.T+b+z@w.T
error=float((inside-outside).abs().max())
if error>1e-12:raise ValueError('forward equivalence')
y=torch.randn(7,20,dtype=torch.float64)
g1=torch.autograd.grad(((inside-y)**2).mean(),(a,c),retain_graph=True)
g2=torch.autograd.grad(((outside-y)**2).mean(),(a,c))
gerror=max(float((v-u).abs().max()) for v,u in zip(g1,g2))
if gerror>1e-12:raise ValueError('gradient equivalence')
rank=int(torch.linalg.matrix_rank(w))
if rank>16:raise ValueError('rank bound')
result=dict(passed=True,forward_max_error=error,gradient_max_error=gerror,output_increment_rank_bound=rank,scope='algebra counterexample with random tensors; no market model trained; not ADAPT-Z reproduction')
(R/'EQUIVALENCE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
