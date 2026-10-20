import torch
from torch import nn
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
H=torch.tensor([1,5,10])
def features(d,cap):
    n=d.shape[1]; z=[]
    for j in range(n):
        m=min(12,n-1-j); future=d[:,j+1:j+1+m]; S=future.sum(1);A=future.abs().sum(1);x=d[:,j]
        z.append(torch.stack([x.abs().clamp(max=cap),torch.full_like(x,m/12), (x*S/(x*x+1)).clamp(-4,4),(x*S/(x.abs()*A+1)).clamp(-4,4)],1))
    return torch.stack(z,1)
class Route(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode;self.v=nn.Parameter(torch.zeros(5))
        if mode=='constant':self.b=nn.Parameter(torch.zeros(1))
        else:self.g=nn.Sequential(nn.Linear(4,4),nn.Tanh(),nn.Linear(4,1))
    def kernel(self):
        w=torch.softmax(torch.cat([self.v,self.v.new_zeros(1)]),0)
        nodes=torch.cat([w.new_ones(1),1-w.cumsum(0)[:-1],w.new_zeros(1)])
        return torch.stack([nodes[a//2] if a%2==0 else (nodes[a//2]+nodes[a//2+1])/2 for a in range(13)])
    def forward(self,d,f,detail=False):
        k=self.kernel();n=d.shape[1]
        if self.mode=='constant':q=self.b.sigmoid().expand_as(d)
        else:
            mask=f.new_ones(4)
            if self.mode=='age':mask[2:]=0
            if self.mode=='evidence':mask[1]=0
            q=self.g(f*mask).squeeze(-1).sigmoid()
        a=torch.arange(n);lag=a[:,None]-a[None,:]; valid=(lag>0)&(lag<=12)
        dk=k[lag.clamp(0,12)]-k[(lag-1).clamp(0,12)]
        T=torch.eye(n).unsqueeze(0)+(1-q).unsqueeze(1)*dk.unsqueeze(0)*valid.unsqueeze(0)
        e=torch.linalg.solve_triangular(T,d.unsqueeze(-1),upper=False).squeeze(-1)
        age=n-1-a;now=k[age.clamp(max=12)]
        future=k[(age[:,None]+H).clamp(max=12)]
        pred=((1-q)*e)@(future-now[:,None])
        return (pred,e,T,q) if detail else pred
class Small(nn.Module):
    def __init__(self):
        super().__init__();self.g=nn.Sequential(nn.Linear(4,4),nn.Tanh(),nn.Linear(4,3))
    def forward(self,d,f):
        age=torch.arange(d.shape[1]-1,-1,-1);weights=(age<12).to(d.dtype)
        return (self.g(f)*d.unsqueeze(-1)*weights[None,:,None]).sum(1)/12

def covariance(k,v,n):
    # Gaussian permanent innovations + Gaussian finite transient + iid measurement.
    b=torch.cat([k[:1],k[1:]-k[:-1]])
    cov=[v[1]*(b[:13-l]*b[l:]).sum()+(v[0]+2*v[2] if l==0 else -v[2] if l==1 else 0) for l in range(13)]
    c=torch.stack(cov);a=torch.arange(n);lag=(a[:,None]-a[None,:]).abs()
    return c[lag.clamp(max=12)]*(lag<=12)
def gaussian_predict(d,k,v):
    n=d.shape[1];C=covariance(k,v,n)
    # Cov(sum future differences, past differences) from stationary diff covariance.
    big=covariance(k,v,n+10);B=torch.stack([big[n:n+int(h),:n].sum(0) for h in H])
    return d@torch.linalg.solve(C,B.T)
