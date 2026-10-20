#!/usr/bin/env python3
"""Single-step carbon price forecasting: a prespecified finite development pilot.

This script is intentionally self-contained and takes its raw data from the
user's existing private archives plus source-identified public CEA/CCER series.
No raw market observations or per-origin predictions should be publicly pushed.
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, random, time, zipfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

LOOKBACK=32
HORIZON=1
LRS=[.001,.003]
SEEDS=[17,29,43]
MAX_EPOCH=45
PATIENCE=8
ALPHA_GRID=[.0001,.001,.01,.1,1,10]
SPECIAL_SOURCE_SHA={
"CEA_REPORTED_CLOSE":"287c72ca277233005d54b9e04805d49762a439ad",
"CCER_DAILY_AVERAGE":"1e6ed6269612b529a833e4ae84af5901e5080d6d"}
MODELS=("raw_free","raw_zero","comp_free","comp_zero","gru_residual")


def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def safe_json(obj):
    def conv(v):
        if isinstance(v, np.generic): return v.item()
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, dict):return {k:conv(x) for k,x in v.items()}
        if isinstance(v, (list,tuple)):return [conv(x) for x in v]
        return v
    return conv(obj)
def writej(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(safe_json(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def get_markets(data_dir, new_data_dir):
    data_dir=Path(data_dir)
    main=data_dir/'carbon_distribution_update_private_run_20261019.zip'
    aux=data_dir/'carbon_research_prior_transport_20261018.zip'
    out={}
    with zipfile.ZipFile(main) as z:
        for key in ('EUA_LONG','GUANGZHOU','HUBEI','SHANGHAI'):
            name=f'inputs/development/{key}_development.csv'
            b=z.read(name)
            df=pd.read_csv(io.BytesIO(b))
            out[key]=(df,{'container_archive':main.name,'inner_path':name,'bytes_sha256':sha_bytes(b),'data_identity':'archived_author_reported_price','market_group':'EU' if key=='EUA_LONG' else key})
    with zipfile.ZipFile(aux) as z:
        name='research/cross_market_innovation_20261017/data/EU_REPORTED_development.csv'
        b=z.read(name)
        out['EU_REPORTED_SOURCE_SENSITIVITY']=(pd.read_csv(io.BytesIO(b)),{'container_archive':aux.name,'inner_path':name,'bytes_sha256':sha_bytes(b),'data_identity':'author_reported_secondary_EU_source','market_group':'EU','do_not_count_as_new_independent_market':True})
    for key,file,name in [ ('CCER_DAILY_AVERAGE','ccer_from_public_repository.csv','volume_weighted_average'), ('CEA_REPORTED_CLOSE','cea_from_public_repository.csv','reported_close_not_official_average')]:
        p=Path(new_data_dir)/file
        out[key]=(pd.read_csv(p),{'container_archive':None,'inner_path':file,'bytes_sha256':sha_bytes(p.read_bytes()),'data_identity':name,'market_group':key,'github_source_blob_sha':SPECIAL_SOURCE_SHA[key], 'public_reconstruction':True})
    return out

def validate_data(df,market):
    df=df.copy();df['date']=pd.to_datetime(df['date'],errors='raise');df['price']=pd.to_numeric(df['price'],errors='raise')
    if df.date.isna().any() or df.date.duplicated().any() or not df.date.is_monotonic_increasing:raise ValueError(f'Invalid dates: {market}')
    if not np.isfinite(df.price).all():raise ValueError(f'Nonfinite price: {market}')
    removed=[]
    # Source CCER begins with a one-off Jan 2024 row, then relaunch Mar 2025.
    # This is a market discontinuity, not a date to forward-fill.
    if market=='CCER_DAILY_AVERAGE':
        first=df.iloc[0].date
        gap_days=np.diff(df.date.values.astype('datetime64[D]')).astype(int)
        if int(gap_days.max())>180:
            pivot=int(np.argmax(gap_days))+1
            for _,r in df.iloc[:pivot].iterrows():removed.append({'date':str(r.date.date()),'reason':'early_isolated_pre_relaunch_record_after_gt180_day_gap'})
            df=df.iloc[pivot:].copy()
    for _,r in df.loc[df.price<=0].iterrows():removed.append({'date':str(r.date.date()),'reason':'nonpositive_reported_price'})
    df=df.loc[df.price>0].reset_index(drop=True)
    if df.empty:raise ValueError('No usable price series')
    return df,removed

def lag_weights():
    r=np.array([.5,.9])
    return ((1-r)[None,:]*r[None,:]**np.arange(LOOKBACK-1,-1,-1)[:,None]).astype(float)
W=lag_weights()
Q,_=np.linalg.qr(np.column_stack((W,np.eye(LOOKBACK)[:,-1])))
assert np.linalg.matrix_rank(np.column_stack((W,np.eye(LOOKBACK)[:,-1])))==3

def calc_context(X):
    st=X@W
    s5=np.sqrt(np.mean(X[:,-5:]**2,axis=1))
    s20=np.sqrt(np.mean(X[:,-20:]**2,axis=1))
    return np.column_stack((X[:,-1], st, np.log((s5+.05)/(s20+.05))))

def build_cell(df,year,min_train=80):
    p=df.price.to_numpy(float); dates=df.date.to_numpy(dtype='datetime64[D]')
    b=int(np.searchsorted(dates,np.datetime64(f'{year}-01-01')))
    e=int(np.searchsorted(dates,np.datetime64(f'{year+1}-01-01')))
    if e-b<35:return None,'insufficient_test_obs'
    origins=np.arange(LOOKBACK-1,len(p)-1)
    test=(origins>=b)&(origins+1<e)
    if test.sum()<30:return None,'too_few_test_origins'
    max_val=int(min(100,max(22,(b-LOOKBACK)//4)))
    a=b-max_val
    train=(origins+1<a)
    val=(origins>=a)&(origins+1<b)
    if train.sum()<min_train or val.sum()<20:return None,'insufficient_training_or_validation'
    diff=np.diff(p,prepend=p[0]);s=float(np.sqrt(np.mean(diff[1:a]**2)))
    if s<=1e-9:return None,'zero_train_variance'
    X=np.stack([diff[t-LOOKBACK+1:t+1]/s for t in origins])
    Y=diff[origins+1]/s
    if not np.isfinite(X).all():raise ValueError('bad design matrix')
    span=(Q@Q.T)
    U=X-X@span
    assert np.max(np.abs(U@Q))<1e-10
    C=calc_context(X)
    return {'X':X,'U':U,'C':C,'Y':Y,'price':p[origins+1],'anchor':p[origins], 'origins':origins,'dates':dates,
            'train':train,'val':val,'final':train|val,'test':test,'s':s,
            'info':{'year':year,'test_origins':int(test.sum()),'train_origins':int(train.sum()),'validation_origins':int(val.sum()),
                    'test_start':str(dates[origins[test][0]]),'test_end':str(dates[origins[test][-1]]),
                    'last_fitting_target':str(dates[origins[(train|val)][-1]+1]),
                    'min_training_date':str(dates[origins[train][0]]),'normalization_s':s,
                    'max_original_date_gap_days':int(np.max(np.diff(dates).astype(int)))}} , None

def ridge_select(x,y,tr,va,alpha_grid):
    best=None
    for a in alpha_grid:
        mod=Ridge(alpha=a*tr.sum()).fit(x[tr],y[tr])
        val=float(np.mean((mod.predict(x[va])-y[va])**2))
        if best is None or (val,a)<(best[0],best[1]):best=(val,a)
    return best[1],best[0]

def static_baseline(c):
    states=c['X']@W
    a,val=ridge_select(states,c['Y'],c['train'],c['val'],ALPHA_GRID)
    model_selection=Ridge(alpha=a*c['train'].sum()).fit(states[c['train']],c['Y'][c['train']]);selection=model_selection.predict(states)
    model_final=Ridge(alpha=a*c['final'].sum()).fit(states[c['final']],c['Y'][c['final']]);pred=model_final.predict(states)
    return selection,pred,{'alpha':a,'validation_mse':val,'final_coefficients':model_final.coef_.tolist(),'final_intercept':float(model_final.intercept_)}

class CNN(nn.Module):
    def __init__(self,width=16):
        super().__init__()
        self.conv1=nn.Conv1d(1,width,kernel_size=3,padding=1)
        self.conv2=nn.Conv1d(width,width,kernel_size=3,padding=2,dilation=2)
        self.act=nn.GELU()
        self.head=nn.Sequential(nn.Linear(width*5+4,24),nn.GELU(),nn.Linear(24,1,bias=False))
        nn.init.zeros_(self.head[-1].weight)
    def forward(self,x,c):
        v=self.act(self.conv1(x.unsqueeze(1)))
        v=self.act(self.conv2(v))
        pooled=nn.functional.adaptive_avg_pool1d(v,4).flatten(1)
        return self.head(torch.cat((pooled,v[:,:,-1],c),axis=1)).squeeze(-1)

class GRU(nn.Module):
    def __init__(self,width=16):
        super().__init__()
        self.gru=nn.GRU(1,width,batch_first=True)
        self.head=nn.Linear(width+4,1,bias=False)
        nn.init.zeros_(self.head.weight)
    def forward(self,x,c):
        o,_=self.gru(x.unsqueeze(-1))
        return self.head(torch.cat((o[:,-1],c),axis=1)).squeeze(-1)

class Residual(nn.Module):
    def __init__(self,kind,width=16):
        super().__init__();self.kind=kind
        self.encoder=GRU(width) if kind=='gru_residual' else CNN(width)
    def forward(self,x,c):
        z=self.encoder(x,c)
        if self.kind.endswith('_zero'):
            z=z-self.encoder(torch.zeros_like(x),c)
        return z


def train_nn(kind,c,back,lr,seed,phase='select',max_epochs=None):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.set_num_threads(1)
    fit_ids=c['train'] if phase=='select' else c['final']
    x=c['U'] if kind.startswith('comp') else c['X']
    xscale=max(float(np.sqrt(np.mean(x[fit_ids]**2))),.02)
    ctx=c['C']
    # Context normalization can use fitting prefix only (no validation/test statistics).
    mean=ctx[fit_ids].mean(axis=0);std=np.maximum(ctx[fit_ids].std(axis=0),.1)
    xt=torch.tensor(x/xscale,dtype=torch.float32)
    ct=torch.tensor((ctx-mean)/std,dtype=torch.float32)
    yt=torch.tensor(c['Y']-back,dtype=torch.float32)
    mdl=Residual(kind,16)
    opt=torch.optim.AdamW(mdl.parameters(),lr=lr,weight_decay=.01)
    n_epoch=MAX_EPOCH if max_epochs is None else int(max_epochs)
    best_val=float('inf');best_epoch=0;stale=0;updates=0;best_state=None
    ids=np.where(fit_ids)[0]
    rng=np.random.default_rng(seed)
    for epoch in range(n_epoch+1):
        mdl.eval()
        with torch.no_grad():
            if phase=='select':
                val=float(torch.mean((mdl(xt[c['val']],ct[c['val']])-yt[c['val']])**2).item())
                if val < best_val-1e-6:
                    best_val,best_epoch,stale,best_state=val,epoch,0,{k:v.clone() for k,v in mdl.state_dict().items()}
                else:stale+=1
            if epoch==n_epoch or (phase=='select' and epoch>=12 and stale>=PATIENCE):break
        mdl.train()
        # One epoch in batches. A new seeded permutation for each epoch.
        for chunk in np.array_split(rng.permutation(ids),max(1,int(np.ceil(len(ids)/128)))):
            opt.zero_grad(set_to_none=True)
            err=mdl(xt[chunk],ct[chunk])-yt[chunk]
            loss=(err*err).mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite training')
            loss.backward();nn.utils.clip_grad_norm_(mdl.parameters(),5.);opt.step();updates+=1
    if phase=='select':mdl.load_state_dict(best_state)
    mdl.eval()
    with torch.no_grad():pred=back+mdl(xt,ct).numpy()
    return pred,{'seed':seed,'lr':lr,'best_epoch':best_epoch if phase=='select' else n_epoch,'validation_mse':best_val if phase=='select' else None,'optimizer_steps':updates,'epochs_executed':epoch,'params':sum(v.numel() for v in mdl.parameters()),'kind':kind,'gradient_loss':float(np.mean((pred[fit_ids]-c['Y'][fit_ids])**2)),'xscale':xscale}

def score(pred,obs,anchor,rw_mse):
    err=pred-obs
    mse=float(np.mean(err*err));rmse=float(np.sqrt(mse))
    return {'RMSE':rmse,'MAE':float(np.mean(np.abs(err))),
            'MAPE_percent':float(np.mean(np.abs(err)/np.maximum(np.abs(obs),1e-8))*100),
            'R2_price_level':float(r2_score(obs,pred)),
            'Q_MSE_over_RW':mse/rw_mse,
            'MSE':mse}

def run_cell(market,df,source,year,out,kindset=MODELS):
    start=time.perf_counter();df,excluded=validate_data(df,market)
    cell,why=build_cell(df,year)
    if cell is None:return {'cell':market+'_'+str(year),'status':'SKIPPED','reason':why,'rows':len(df),'excluded':excluded}
    c=cell;c['info']['market']=market;c['info']['source']=source;c['info']['excluded']=excluded
    group=int(c['test'].sum());truth=c['price'][c['test']];anchor=c['anchor'][c['test']]
    rw_mse=float(np.mean((truth-anchor)**2));assert rw_mse>0
    static_sel,static_final,static_meta=static_baseline(c)
    prediction={'rw':anchor,'static':c['anchor'][c['test']]+c['s']*static_final[c['test']]}
    # Ridge32 exactly uses known price-change history, not levels.
    sc=StandardScaler().fit(c['X'][c['train']]);z=sc.transform(c['X']);alpha,vl=ridge_select(z,c['Y'],c['train'],c['val'],ALPHA_GRID)
    sc=StandardScaler().fit(c['X'][c['final']]);z=sc.transform(c['X']);rg=Ridge(alpha=alpha*c['final'].sum()).fit(z[c['final']],c['Y'][c['final']])
    prediction['ridge32']=c['anchor'][c['test']]+c['s']*rg.predict(z[c['test']])
    # HGB with four fixed configurations; select validation only.
    feat=np.column_stack((c['X'],c['C'][:,-1]))
    best=None
    for leaves,l2 in ((7,.1),(7,1.),(15,.1),(15,1.)):
        m=HistGradientBoostingRegressor(max_iter=100,learning_rate=.05,max_leaf_nodes=leaves,l2_regularization=l2,early_stopping=False,random_state=17).fit(feat[c['train']],c['Y'][c['train']])
        score_val=float(np.mean((m.predict(feat[c['val']])-c['Y'][c['val']])**2))
        if best is None or (score_val,leaves,l2)<best:best=(score_val,leaves,l2)
    hgb_validation_mse,leaves,l2=best
    m=HistGradientBoostingRegressor(max_iter=100,learning_rate=.05,max_leaf_nodes=leaves,l2_regularization=l2,early_stopping=False,random_state=17).fit(feat[c['final']],c['Y'][c['final']])
    prediction['hgb']=c['anchor'][c['test']]+c['s']*m.predict(feat[c['test']])
    # NN models validation per kind, seed17 for config; 3 seed refits at chosen epochs.
    nneval=[];trial_summary={};predseed={}
    for kind in kindset:
        choices=[]
        for lr in LRS:
            _,receipt=train_nn(kind,c,static_sel,lr,SEEDS[0])
            choices.append(receipt)
        best=min(choices,key=lambda t:(t['validation_mse'],t['lr'],t['best_epoch']))
        trial_summary[kind]={'validation_trials':choices,'selected':best}
        member_preds=[];members=[]
        for seed in SEEDS:
            pr,rr=train_nn(kind,c,static_final,best['lr'],seed,phase='final',max_epochs=best['best_epoch'])
            member_preds.append(c['anchor'][c['test']]+c['s']*pr[c['test']]);members.append(rr)
        predseed[kind]=np.array(member_preds)
        trial_summary[kind]['refits']=members
        prediction[kind]=predseed[kind].mean(axis=0)
        nneval.append(kind)
    scores={model:score(pr,truth,anchor,rw_mse) for model,pr in prediction.items()}
    for kind in kindset:
        ms=[score(pr,truth,anchor,rw_mse) for pr in predseed[kind]]
        scores[kind]['Q_seed_mean']=float(np.mean([z['Q_MSE_over_RW'] for z in ms]))
        scores[kind]['Q_seed_std']=float(np.std([z['Q_MSE_over_RW'] for z in ms],ddof=1))
        scores[kind]['Q_ensemble']=scores[kind]['Q_MSE_over_RW']
        scores[kind]['seed_mean_metrics']={k:float(np.mean([v[k] for v in ms])) for k in ('RMSE','MAE','MAPE_percent','R2_price_level','Q_MSE_over_RW','MSE')}
        scores[kind]['RMSE_seed_mean']=float(np.sqrt(np.mean([z['MSE'] for z in ms])))
    # Keep per-origin model predictions private; public only aggregates.
    pdir=out/'private';pdir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(pdir/(market+'_'+str(year)+'.npz'),anchor=anchor,observed=truth,origins=c['origins'][c['test']],**prediction,**{'seeds_'+k:ps for k,ps in predseed.items()})
    public={'cell':market+'_'+str(year),'status':'COMPLETE','info':c['info'],'static':static_meta,'ridge':{'alpha':alpha,'validation_mse':vl},'hgb':{'leaves':leaves,'l2':l2,'validation_mse':hgb_validation_mse},'neural':trial_summary,'score':scores,'time_seconds':time.perf_counter()-start}
    writej(out/'cells'/(market+'_'+str(year)+'.json'),public)
    return {'cell':public['cell'],'status':'COMPLETE','score':scores,'info':c['info'],'time_seconds':public['time_seconds']}

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--data-dir',default='/mnt/data/carbon_research/recovered_archives');pa.add_argument('--new-data-dir',default='/mnt/data/carbon_next_20261020/data');pa.add_argument('--out',required=True);pa.add_argument('--preflight',action='store_true');pa.add_argument('--short',action='store_true');args=pa.parse_args()
    src=get_markets(args.data_dir, args.new_data_dir)
    for market,filename in (('CCER_DAILY_AVERAGE','ccer_from_public_repository.csv'),('CEA_REPORTED_CLOSE','cea_from_public_repository.csv')):
        frame,pr=src[market];src[market]=(frame,pr)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    code={x.name:sha_bytes(x.read_bytes()) for x in Path(__file__).parent.glob('*.py')}
    cells=[(m,y) for m in ('EUA_LONG','GUANGZHOU','HUBEI','SHANGHAI') for y in (2021,2022,2023)]
    cells += [(m,y) for m in ('EU_REPORTED_SOURCE_SENSITIVITY',) for y in (2022,2023)]
    cells += [('CCER_DAILY_AVERAGE',2026),('CEA_REPORTED_CLOSE',2026)]
    protocol={'study':'one_step_path_complement_development','lookback':LOOKBACK,'horizon_records':HORIZON,
              'source_shas':{m:v[1] for m,v in src.items()}, 'epochs_max':MAX_EPOCH,'patience':PATIENCE,'seeds':SEEDS,'learning_rates':LRS,
              'alpha_grid':ALPHA_GRID,'candidate':'comp_zero','information':'historical prices only, full past-window representation; validation used for tuning, test-year data exposed from previous project development',
              'models':['rw','static','ridge32','hgb',*MODELS], 'cells_predeclared':cells,
              'main_nn_metric':'mean_loss_across_three_final_seeds; ensemble_pred_loss_retained_as_separate_diagnostic',
              'source_sha256':code,'input_convention':'price[t+1] at origin t: known price[t] plus forecast next arithmetic change; train-only scale for change and all other scalers',
              'evaluation':'chronological, market-year cell, in-year one-observation-ahead, equal cell mean Q; native RMSE MAE MAPE R2 for paper-compatible display',
              'publication':'all exposed development; no new independent confirmation; no future-data full-series decomposition'}
    writej(out/'PROTOCOL.json',protocol)
    if args.preflight:
        a=[]
        for m,y in cells:
            df,meta=src[m];ff,exc=validate_data(df,m);cc,why=build_cell(ff,y);a.append({'cell':m+'_'+str(y),'why':why,'counts':None if cc is None else cc['info'],'exclusions':exc})
        writej(out/'PREFLIGHT.json',a);print(json.dumps(a,ensure_ascii=False,indent=2));return
    results=[]
    for m,y in cells[:1] if args.short else cells:
        df,meta=src[m]
        rr=run_cell(m,df,meta,y,out)
        results.append(rr)
        if rr['status']=='COMPLETE': print('DONE',m,y,'Q',[round(rr['score'][a]['Q_MSE_over_RW'],4) for a in ('rw','static','raw_free','raw_zero','comp_zero','gru_residual')], 'time',round(rr['time_seconds'],1),flush=True)
        else:print('SKIP',m,y,rr['reason'],flush=True)
    if code!={x.name:sha_bytes(x.read_bytes()) for x in Path(__file__).parent.glob('*.py')}:raise RuntimeError('source code mutated during run')
    writej(out/'COMPLETE.json',{'status':'COMPLETE_PILOT' if args.short else 'COMPLETE_DECLARED_MATRIX','source_unchanged':True,'n_cells_attempted':len(results),'n_cells_completed':sum(z['status']=='COMPLETE' for z in results),'results':results,'code_sha256':code})

if __name__=='__main__':main()
