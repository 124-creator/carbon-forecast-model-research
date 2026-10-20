from pathlib import Path
import json,hashlib
import numpy as np
import torch
from core import Route,features
from run import dataset,masks,loss,generate
ROOT=Path(__file__).parent

def engineering():
    torch.manual_seed(19);model=Route('full');d=torch.randn(4,59);f=features(d,4);t=torch.randn(4,3)
    def objective():return ((model(d,f)-t)**2).mean()
    model.zero_grad();objective().backward();pars=list(model.parameters());analytic=torch.cat([p.grad.flatten() for p in pars]).detach();errors=[];eps=1e-5
    for p in pars:
        for j in range(p.numel()):
            with torch.no_grad():p.flatten()[j]+=eps
            plus=float(objective())
            with torch.no_grad():p.flatten()[j]-=2*eps
            minus=float(objective())
            with torch.no_grad():p.flatten()[j]+=eps
            errors.append((plus-minus)/(2*eps))
    direction=torch.randn_like(analytic);direction/=direction.norm();offset=0
    for p in pars:
        with torch.no_grad():p.add_(eps*direction[offset:offset+p.numel()].reshape(p.shape))
        offset+=p.numel()
    plus=float(objective());offset=0
    for p in pars:
        with torch.no_grad():p.sub_(2*eps*direction[offset:offset+p.numel()].reshape(p.shape))
        offset+=p.numel()
    minus=float(objective());offset=0
    for p in pars:
        with torch.no_grad():p.add_(eps*direction[offset:offset+p.numel()].reshape(p.shape))
        offset+=p.numel()
    y,_=generate('G2',7);scale=np.std(np.diff(y[:1200]));cap=np.quantile(np.abs(np.diff(y[:1200]))/scale,.995)
    o,dd,ff,tt,mm=dataset(y,scale,cap);ms=masks(o,mm);y2=y.copy();y2[1200:]+=100*np.sin(np.arange(1200));o2,d2,f2,t2,m2=dataset(y2,scale,cap)
    trainrows=ms['train'].any(1)
    with torch.no_grad():future_delta=float((model(dd[trainrows],ff[trainrows])-model(d2[trainrows],f2[trainrows])).abs().max())
    # Literal objective-gradient invariance when every immature label is changed.
    altered=tt.clone();altered[~ms['train']]+=1e6
    model.zero_grad();loss(model(dd[trainrows],ff[trainrows]),tt[trainrows],ms['train'][trainrows]).backward();g1=torch.cat([p.grad.flatten() for p in pars]).clone()
    model.zero_grad();loss(model(dd[trainrows],ff[trainrows]),altered[trainrows],ms['train'][trainrows]).backward();g2=torch.cat([p.grad.flatten() for p in pars]).clone()
    constant=Route('constant');a=torch.randn(4,59);b=torch.randn(4,59)
    with torch.no_grad():superposition=float((constant(a+b,features(a+b,4))-constant(a,features(a,4))-constant(b,features(b,4))).abs().max())
    p,e,T,q=model(d,f,True);k=model.kernel();age=torch.arange(58,-1,-1)
    correction=lambda h:((1-q)*e*(k[(age+h).clamp(max=12)]-k[age.clamp(max=12)])).sum(1)
    dangerous=Route('constant')
    with torch.no_grad():dangerous.b.fill_(-30);dangerous.v.zero_();dangerous.v[0]=30
    _,_,badT,_=dangerous(d[:1],f[:1],True);badinv=torch.linalg.inv(badT[0]);bn=float(torch.linalg.matrix_norm(badinv,2))
    # Boundary initial state diagnosis: inject impulse before prefix into forward convolution.
    latent=torch.randn(80);truek=torch.tensor(np.interp(np.arange(13),np.arange(0,13,2),[1,.95,.85,.45,.2,.05,0]));latent[20]=8
    levels=torch.tensor(np.convolve(latent.numpy(),truek.numpy())[:80]);prefixdiff=torch.diff(levels[22:]);_,recovered,_,_=constant(prefixdiff[None,:],features(prefixdiff[None,:],4),True)
    return {'coordinate_gradient_max_error':float((analytic-torch.tensor(errors)).abs().max()),'direction_gradient_error':abs((plus-minus)/(2*eps)-float(analytic@direction)),'future_suffix_prediction_delta':future_delta,'immature_labels_fit_input_delta':float((g1-g2).abs().max()),'constant_route_superposition_error':superposition,'nonlinear_route_superposition_delta':float((model(a+b,features(a+b,4))-model(a,features(a,4))-model(b,features(b,4))).abs().max()),'h12_h24_platform_delta':float((correction(12)-correction(24)).abs().max()),'q0_control_inverse_norm':bn,'boundary_zero_initial_innovation_error_DIAGNOSTIC':float((recovered[0]-latent[23:]).norm()),'reconstruction_is_identification':False}

def verify_saved():
    results=json.loads((ROOT/'RESULTS.json').read_text(encoding='utf-8'));errors=[];computed={}
    for fam in ['G0','G1','G2','G5']:
      for seed in [7,17]:
        data=np.load(ROOT/f'{fam}_{seed}_synthetic.npz');pred=np.load(ROOT/f'{fam}_{seed}_predictions.npz');generated,_=generate(fam,seed)
        if not np.array_equal(generated,data['y']):errors.append([fam,seed,'generator'])
        for name in pred.files:
            h=[float(np.mean((pred[name][:,j][data['test'][:,j]]-data['targets'][:,j][data['test'][:,j]])**2)) for j in range(3)]
            method='linear_oracle_DIAGNOSTIC' if name=='linear_oracle' else name
            row=next(x for x in results if (x['family'],x['seed'],x['method'])==(fam,seed,method))
            if not np.allclose(h,row['test_h'],atol=1e-12):errors.append([fam,seed,name,'metrics'])
            computed[f'{fam}_{seed}_{method}']=float(np.mean(h))
    get=lambda fam,seed,name:computed[f'{fam}_{seed}_{name}']
    gate={}
    for seed in [7,17]:
        full=get('G2',seed,'full');strong=min(get('G2',seed,name) for name in ['constant','age','evidence','small','mle','rw'])
        gate[str(seed)]={'full_vs_strongest_gain':1-full/strong,'age_increment_over_evidence':1-full/get('G2',seed,'evidence'),'evidence_increment_over_age':1-full/get('G2',seed,'age'),'null_rw_ratio':get('G0',seed,'full')/get('G0',seed,'rw'),'linear_oracle_ratio':get('G1',seed,'full')/get('G1',seed,'linear_oracle_DIAGNOSTIC')}
    mechanism=all(x['full_vs_strongest_gain']>=.03 and x['age_increment_over_evidence']>=.02 and x['evidence_increment_over_age']>=.02 for x in gate.values())
    stable=all(x.get('stable_learning',False) for x in results if x['method']=='full')
    inverse=all(x['diagnostics']['inverse_gate'] for x in results if x['method']=='full')
    positive=mechanism and stable and inverse and all(x['null_rw_ratio']<=1.01 and x['linear_oracle_ratio']<=1.05 for x in gate.values())
    if errors:raise RuntimeError(errors)
    return {'verified_rows':len(results),'metric_recomputed_count':len(computed),'errors':errors,'seed_gates':gate,'mechanism_pass':mechanism,'stable_learning_all_full':stable,'inverse_all_full':inverse,'overall_pass':positive,'decision':'continue research; STOP current implementation, no market access' if not positive else 'synthetic preliminary gate only; no market authorization','two_seeds_not_original_five_seed_confirmation':True}
if __name__=='__main__':
    (ROOT/'ENGINEERING.json').write_text(json.dumps(engineering(),indent=2),encoding='utf-8')
    if (ROOT/'COMPLETE.json').exists():(ROOT/'VERIFICATION.json').write_text(json.dumps(verify_saved(),indent=2),encoding='utf-8')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.is_file() and p.name!='MANIFEST.json'}
    (ROOT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
