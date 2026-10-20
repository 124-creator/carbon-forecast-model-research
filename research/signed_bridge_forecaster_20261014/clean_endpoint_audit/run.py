import sys,json,time,hashlib,inspect
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from model import Forecaster,ordinary,bridge_matrix
from experiment import generate,prepare,metrics,REGIMES
from neuralforecast.models.nhits import NHITSBlock

def outputs(m,z):
    # Both views have exactly the same ordinary DST observations.
    _,inv=ordinary(8,'dst')
    raw=z@inv.float().T
    c=raw@m.E.T
    q=torch.stack([b(c[:,:,r])[1] for r,b in enumerate(m.blocks)],-1)
    routed=[]; full=[]
    for j,n in enumerate((8,8,4)):
        d=getattr(m,'D'+str(n)); v=q[:,j,:n]
        routed.append(v[:,:1]@d[:,:1].T+v[:,1:]@d[:,1:].T)
        full.append(v@d.T)
    return torch.cat(routed,-1).cumsum(-1),torch.cat(full,-1).cumsum(-1)

def run():
    torch.set_num_threads(1); start=time.monotonic()
    official=Path(inspect.getfile(NHITSBlock)); sha=hashlib.sha256(official.read_bytes()).hexdigest()
    result={'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),'fit_count':0,'independent_two_model_fits':False,'runs':[],'math':[],'official_sha256_before':sha}
    for n in (8,4):
        e,d=bridge_matrix(n); endpoint=d.sum(0)
        # Any detail-only change is zero at the endpoint; an added constant increment is not.
        witness=torch.ones(n,dtype=torch.float64)
        residual=witness-d[:,1:]@torch.linalg.lstsq(d[:,1:],witness).solution
        result['math'].append({'n':n,'D_rank':int(torch.linalg.matrix_rank(d)),'detail_rank':int(torch.linalg.matrix_rank(d[:,1:])),'endpoint_row':endpoint.tolist(),'free_endpoint_witness_residual_norm':float(residual.norm()),'decoder_identity_error':float((d@e-torch.eye(n,dtype=torch.float64)).abs().max())})
    torch.manual_seed(17); m=Forecaster('bridge'); z=torch.randn(7,8,8)
    a,b=outputs(m,z)
    ga=torch.autograd.grad(a.square().mean(),tuple(m.parameters()),allow_unused=True,retain_graph=True)
    gb=torch.autograd.grad(b.square().mean(),tuple(m.parameters()),allow_unused=True)
    result['pretraining_forward_max_error']=float((a-b).abs().max())
    result['pretraining_gradient_max_error']=max(float((x-y).abs().max()) for x,y in zip(ga,gb) if x is not None)
    assert result['pretraining_forward_max_error']<1e-5 and result['pretraining_gradient_max_error']<1e-5
    old=json.loads((ROOT.parent/'results.json').read_text(encoding='utf8'))
    result['prior_candidate_results']=[r for r in old['runs'] if r['kind']=='bridge']
    result['prior_results_sha256']=hashlib.sha256((ROOT.parent/'results.json').read_bytes()).hexdigest()
    local=ROOT/'local'; local.mkdir(exist_ok=True)
    enc,_=ordinary(8,'dst')
    for regime in REGIMES:
      for seed in (17,43):
        assert time.monotonic()-start<450
        sets,scale,origins=prepare(generate(seed,regime))
        zs=[x.reshape(-1,8,8)@enc.float().T for x,y in sets]
        ys=[y for x,y in sets]
        torch.manual_seed(seed); m=Forecaster('bridge'); opt=torch.optim.Adam(m.parameters(),lr=.001); g=torch.Generator().manual_seed(seed)
        trace=[]; maxdiff=0.
        with torch.no_grad(): initial=float(((outputs(m,zs[0])[1]*scale-ys[0])**2).mean())
        for step in range(160):
            idx=torch.randint(len(zs[0]),(64,),generator=g); opt.zero_grad()
            a,b=outputs(m,zs[0][idx]); loss=((b*scale-ys[0][idx])**2).mean(); loss.backward(); opt.step()
            with torch.no_grad():
                aa,bb=outputs(m,zs[0]); av,bv=outputs(m,zs[1]); maxdiff=max(maxdiff,float((aa-bb).abs().max()),float((av-bv).abs().max()))
                trace.append({'update':step+1,'minibatch_loss':float(loss),'train_full_loss':float(((bb*scale-ys[0])**2).mean()),'validation_full_loss':float(((bv*scale-ys[1])**2).mean()),'candidate_control_max_output_error':maxdiff})
        with torch.no_grad():
            a,b=outputs(m,zs[2]); sa=metrics(a*scale,ys[2]); sb=metrics(b*scale,ys[2]); zero=metrics(torch.zeros_like(ys[2]),ys[2])
        row={'regime':regime,'seed':seed,'steps':160,'parameters':sum(p.numel() for p in m.parameters()),'initial_train_loss':initial,'final_train_loss':trace[-1]['train_full_loss'],'candidate':sa,'same_space_control':sb,'endpoint_increment_mse':sb['mse20']-sa['mse20'],'zero':zero,'ratio_to_zero':sa['mse20']/zero['mse20'],'max_training_output_error':maxdiff,'loss_trace':trace}
        result['runs'].append(row); result['fit_count']+=1
        np.savez_compressed(local/f'{regime}_{seed}.npz',candidate_prediction=a.numpy()*scale,control_prediction=b.numpy()*scale,target=ys[2].numpy(),**{k:v.detach().numpy() for k,v in m.state_dict().items()})
        (ROOT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        print(regime,seed,sa['mse20'],sb['mse20'],maxdiff,flush=True)
    result['official_sha256_after']=hashlib.sha256(official.read_bytes()).hexdigest(); assert sha==result['official_sha256_after']
    assert result['fit_count']==8 and all(len(r['loss_trace'])==160 for r in result['runs'])
    assert all(abs(r['endpoint_increment_mse'])<1e-4 for r in result['runs'])
    result['elapsed_seconds']=time.monotonic()-start
    result['decision']='STOP signed-bridge new-constraint route: strict fixed-space route removal is algebraically identical. No independent relaxed endpoint function class was fitted.'
    (ROOT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('COMPLETE',result['fit_count'],result['elapsed_seconds'],flush=True)
if __name__=='__main__': run()
