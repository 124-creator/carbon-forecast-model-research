# -*- coding: utf-8 -*-
"""Frozen round4 task A: four correctly trained sequence quantile baselines.
Only deep_seq_round4* files are written; old candidates remain read-only.
Fifteen quantiles identify MQS/WIS, not exact full-distribution CRPS.
"""
from __future__ import annotations
import os,sys
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
sys.dont_write_bytecode=True
import argparse,json,math,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits
import empirical_tilt_round3 as T

HERE=Path(__file__).resolve().parent
TAU=torch.tensor(T.B.TAUS,dtype=torch.float32)
MODELS=['lstm_qr','gru_qr','tcn_qr','transformer_qr']
SEEDS=[11,22,33,44,55]


def pinball_values(y,q):
    e=y[:,None]-q
    return torch.maximum(TAU.to(q.dtype)*e,(TAU.to(q.dtype)-1)*e)


def sanity():
    q=torch.zeros(2,len(TAU));v=pinball_values(torch.tensor([1.,-1.]),q)
    assert abs(float(v[0,2])-.1)<1e-6 and abs(float(v[1,2])-.9)<1e-6
    rng=np.random.default_rng(61);y=torch.tensor(rng.standard_normal(6000),dtype=torch.float32)
    free=nn.Parameter(torch.zeros(len(TAU)));opt=torch.optim.Adam([free],lr=.05)
    for _ in range(250):
        opt.zero_grad();loss=pinball_values(y,free[None].expand(len(y),-1)).mean();loss.backward();opt.step()
    error=float(np.max(np.abs(free.detach().numpy()-np.quantile(y.numpy(),T.B.TAUS))))
    assert error<.12
    return dict(asymmetric_positive_error_tau01=float(v[0,2]),asymmetric_negative_error_tau01=float(v[1,2]),independent_normal_quantile_max_error=error)


def build(d):
    D=T.add_context(d,T.E.build(d));lp=np.log(d.price.to_numpy(float));seq=[]
    for t,hi,scale in zip(D['t'],D['hi'],D['scale']):
        r=np.diff(lp[t-64:t+1]);sd=scale/np.sqrt(T.HOR[hi])
        assert len(r)==64 and sd>0
        seq.append(np.column_stack([r/sd,(np.abs(r)<=1e-12).astype(float)]))
    D['seq']=np.asarray(seq,np.float32)
    va_start=D['audit']['validation_start_index'];cut=D['audit']['cut_index']
    assert np.all(D['t'][D['tr']]+T.HOR[D['hi'][D['tr']]]<va_start)
    assert np.all(D['t'][D['va']]>=va_start) and np.all(D['t'][D['va']]+T.HOR[D['hi'][D['va']]]<cut)
    assert np.all(D['t'][D['rf']]+T.HOR[D['hi'][D['rf']]]<cut) and np.all(D['t'][D['te']]>=cut)
    assert D['x_aug'].shape[1]==36
    return D


def prepare(D,idx):
    mean=D['x_aug'][idx].mean(0);sd=D['x_aug'][idx].std(0)+1e-5
    ctx=torch.tensor(np.clip((D['x_aug']-mean)/sd,-8,8),dtype=torch.float32)
    base=np.stack([np.quantile(D['z'][idx[D['hi'][idx]==hi]],T.B.TAUS) for hi in range(4)]).astype(np.float32)
    return ctx,base,dict(n_fit_rows=len(idx),context_mean_sha256=hashlib.sha256(mean.tobytes()).hexdigest(),context_sd_sha256=hashlib.sha256(sd.tobytes()).hexdigest())


class CausalConv(nn.Module):
    def __init__(self,nin,nout,dilation):
        super().__init__();self.pad=2*dilation;self.conv=nn.Conv1d(nin,nout,3,dilation=dilation,padding=self.pad)
    def forward(self,x):
        return self.conv(x)[:,:,:-self.pad]


class QRNet(nn.Module):
    def __init__(self,model,base,width=16):
        super().__init__();self.model=model;self.width=width
        if model=='lstm_qr':self.encoder=nn.LSTM(2,width,batch_first=True)
        elif model=='gru_qr':self.encoder=nn.GRU(2,width,batch_first=True)
        elif model=='tcn_qr':
            self.encoder=nn.Sequential(CausalConv(2,width,1),nn.GELU(),CausalConv(width,width,2),nn.GELU(),CausalConv(width,width,4),nn.GELU())
        else:
            self.patch=nn.Linear(8,width)
            layer=nn.TransformerEncoderLayer(width,2,dim_feedforward=32,dropout=.05,batch_first=True,norm_first=False)
            self.encoder=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
            pos=torch.arange(16,dtype=torch.float32)[:,None];div=torch.exp(torch.arange(0,width,2,dtype=torch.float32)*(-math.log(10000)/width))
            pe=torch.zeros(16,width);pe[:,0::2]=torch.sin(pos*div);pe[:,1::2]=torch.cos(pos*div)
            self.register_buffer('position',pe[None])
        self.context=nn.Sequential(nn.Linear(36,16),nn.Tanh())
        self.head=nn.Sequential(nn.Linear(width*2+16,24),nn.Tanh(),nn.Dropout(.05),nn.Linear(24,len(TAU)))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('base',torch.tensor(base))

    def encode(self,seq,position=True):
        if self.model in ['lstm_qr','gru_qr']:out,_=self.encoder(seq)
        elif self.model=='tcn_qr':out=self.encoder(seq.transpose(1,2)).transpose(1,2)
        else:
            tokens=self.patch(seq.reshape(len(seq),16,8))
            if position:tokens=tokens+self.position
            out=self.encoder(tokens)
        return torch.cat([out.mean(1),out[:,-1]],1)

    def forward(self,seq,ctx,hi):
        out=self.head(torch.cat([self.encode(seq),self.context(ctx)],1))
        return torch.sort(self.base[hi]+out,dim=1).values

    def encoder_parameters(self):
        out=list(self.encoder.parameters())
        if self.model=='transformer_qr':out+=list(self.patch.parameters())
        return out


def fit_predict(D,model,seed,args):
    seq=torch.tensor(D['seq']);hi=torch.tensor(D['hi'],dtype=torch.long);y=torch.tensor(D['z'],dtype=torch.float32)
    tr,va,rf,te=[torch.tensor(D[k]) for k in ['tr','va','rf','te']]
    def setup(idx):
        ctx,base,meta=prepare(D,np.asarray(idx));torch.manual_seed(seed);net=QRNet(model,base);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.003)
        original=[p.detach().clone() for p in net.encoder_parameters()]
        return ctx,net,opt,meta,original
    def delta(net,original):return float(sum(((p.detach()-o)**2).sum() for p,o in zip(net.encoder_parameters(),original)).sqrt())
    def train_epoch(ctx,net,opt,idx,seen):
        net.train();order=idx[torch.randperm(len(idx))]
        count=torch.bincount(hi[idx],minlength=4);weights=len(idx)/(4*count.float())
        for b in order.split(512):
            opt.zero_grad();q=net(seq[b],ctx[b],hi[b]);loss=(pinball_values(y[b],q).mean(1)*weights[hi[b]]).mean();assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            if not seen:seen=any(p.grad is not None and bool(torch.any(p.grad!=0)) for p in net.encoder_parameters())
            opt.step()
        return seen
    def val(ctx,net):
        net.eval()
        with torch.no_grad():
            parts=[]
            for h in range(4):
                ids=va[hi[va]==h];losses=[]
                for b in ids.split(512):losses.append(pinball_values(y[b],net(seq[b],ctx[b],hi[b])).mean(1))
                parts.append(torch.cat(losses).mean())
        return float(torch.stack(parts).mean())
    ctx,net,opt,inner_meta,orig=setup(tr)
    net.eval()
    with torch.no_grad():
        check=net(seq[tr[:8]],ctx[tr[:8]],hi[tr[:8]])
        initial_error=float((check-net.base[hi[tr[:8]]]).abs().max())
        assert initial_error<1e-6
        pe_effect=float((net.encode(seq[tr[:4]],True)-net.encode(seq[tr[:4]],False)).abs().sum()) if model=='transformer_qr' else None
        if pe_effect is not None:assert pe_effect>1e-5
    best=val(ctx,net);best_ep=0;bad=0;history=[[0,best]];seen=False
    torch.manual_seed(seed+731)
    for ep in range(1,args.epochs+1):
        seen=train_epoch(ctx,net,opt,tr,seen);value=val(ctx,net);history.append([ep,value])
        if value<best-1e-6:best,best_ep,bad=value,ep,0
        else:bad+=1
        if ep>=args.min_epochs and bad>=args.patience:break
    inner_delta=delta(net,orig);assert seen and inner_delta>0
    ctx,net,opt,refit_meta,orig=setup(rf);torch.manual_seed(seed+731);refit_seen=False
    for _ in range(best_ep):refit_seen=train_epoch(ctx,net,opt,rf,refit_seen)
    net.eval()
    with torch.no_grad():q=torch.cat([net(seq[b],ctx[b],hi[b]) for b in te.split(512)]).numpy()*D['scale'][D['te'],None]
    assert np.isfinite(q).all() and np.all(np.diff(q,axis=1)>=-1e-8)
    fd=delta(net,orig)
    if best_ep:assert refit_seen and fd>0
    return q,dict(seed=seed,best_epoch=best_ep,inner_epochs=ep,validation_normalized_macro_pinball=best,validation_history=history,refit_epochs=best_ep,inner_encoder_nonzero_gradient=seen,inner_encoder_parameter_L2_change=inner_delta,refit_encoder_nonzero_gradient=refit_seen,refit_encoder_parameter_L2_change=fd,zerohead_anchor_max_error=initial_error,position_encoding_effect_L1=pe_effect,allocated_parameters=sum(p.numel() for p in net.parameters()),inner_normalization=inner_meta,refit_normalization=refit_meta)


def metrics(y,q):
    return dict(MQS=float(T.B.score(y,q).mean()),WIS=float(T.B.wis(y,q).mean()),MAE_median=float(np.abs(y-q[:,7]).mean()),coverage80=float(((y>=q[:,2])&(y<=q[:,12])).mean()),coverage95=float(((y>=q[:,0])&(y<=q[:,14])).mean()))


def summary(markets):
    cells=[c['scores'] for m in markets.values() for c in m['by_horizon'].values()];out={}
    for model in MODELS:
        if not cells or model not in cells[0]:continue
        for base in ['candidate_probability_mixture','candidate_quantile_average','legacy_empirical','uniform_static','gaussian']:
            vals={}
            for met in ['MQS','WIS','MAE_median']:
                rr=[c[model][met]/c[base][met] for c in cells];vals[met]=dict(geometric_gain_percent=float(100*(1-np.exp(np.log(rr).mean()))),wins=int(np.sum(np.array(rr)<1)),cells=len(rr))
            out[model+'_vs_'+base]=vals
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--markets',nargs='+',default=list(T.B.CUT));ap.add_argument('--models',nargs='+',default=MODELS);ap.add_argument('--seeds',nargs='+',type=int,default=SEEDS);ap.add_argument('--epochs',type=int,default=80);ap.add_argument('--min-epochs',type=int,default=30);ap.add_argument('--patience',type=int,default=16);ap.add_argument('--tag',default='deep_seq_round4');args=ap.parse_args();assert args.tag.startswith('deep_seq_round4')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);san=sanity();print('loss_sanity',json.dumps(san),flush=True)
    paths=[p for p in HERE.glob('empirical_tilt_round3*') if p.is_file()]
    paths+=[HERE/'marked_event_round2.py',HERE/'marked_event_round1.py',HERE/'neural_restart.py']
    frozen={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    prev=np.load(HERE/'empirical_tilt_round3_followup_predictions.npz');panel=pd.read_csv(T.B.PANEL,parse_dates=['date']);start=time.perf_counter()
    r=dict(status='exploratory frozen-cut model experiment; all seeds/models retained',config=vars(args),loss_sanity=san,frozen_sha256=frozen,training='15-tau correct pinball from first epoch on all4h, equal-horizon normalized validation, epoch0 eligible, same-seed full-precut refit',sequence='last64 observed daily returns divided by known origin128-day sd, plus unchanged indicator',context='exact T.add_context(E.build(d)),36dim train-only mean/sd and clip[-8,8]',architectures=dict(width=16,context_width=16,head_width=24,RNN_layers=1,TCN='3 causal dilated convs d1/2/4; concatenate last/mean outputs so all64 origins contribute',Transformer='patch4 retains all64 steps,16 ordered tokens,explicit sinusoidal PE,1 layer2 heads FF32,dropout.05'),ensemble='deep QR quantile average; frozen candidate full probability mixture and separate same-five-seed quantile average',metrics='MQS/WIS/median MAE/coverage only; no exact CRPS inferred from15quantiles',markets={})
    arrays={'taus':T.B.TAUS};rows=[]
    for market in args.markets:
        data=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=build(data);te=D['te'];y=D['y'][te];hi=D['hi'][te]
        for suffix,val in [('origin',D['t'][te]),('horizon',T.HOR[hi]),('target',y),('scale',D['scale'][te])]:
            assert np.array_equal(val,prev[market+'_'+suffix]);arrays[market+'_'+suffix]=val
        arrays[market+'_origin_date']=prev[market+'_origin_date']
        pred={'candidate_probability_mixture':prev[market+'_full_ensemble_Q'],'candidate_quantile_average':np.mean([prev[f'{market}_full_seed{s}_Q'] for s in SEEDS],axis=0),'legacy_empirical':prev[market+'_legacy_empirical_Q'],'uniform_static':prev[market+'_uniform_static_Q'],'gaussian':prev[market+'_gaussian_Q']}
        fits={};single={}
        for model in args.models:
            qs=[];fs=[]
            for seed in args.seeds:
                q,f=fit_predict(D,model,seed,args);qs.append(q);fs.append(f);arrays[f'{market}_{model}_seed{seed}_Q']=q;single[(model,seed)]=q
                print(market,model,seed,'best',f['best_epoch'],'run',f['inner_epochs'],'encoder_grad',f['inner_encoder_nonzero_gradient'],'elapsed',round(time.perf_counter()-start,1),flush=True)
            pred[model]=np.mean(qs,axis=0);fits[model]=fs
        res=dict(audit=D['audit'],fits=fits,alignment='origin,horizon,target,scale exactly equal frozen5seed followup',by_horizon={})
        for j,h in enumerate(T.HOR):
            mask=hi==j;ss={m:metrics(y[mask],q[mask]) for m,q in pred.items()};res['by_horizon'][str(h)]=dict(n=int(mask.sum()),scores=ss)
            for (model,seed),q in single.items():rows.append(dict(market=market,horizon=int(h),model=model,seed=seed,representation='single_seed_quantile',n=int(mask.sum()),**metrics(y[mask],q[mask])))
            for m,q in pred.items():rows.append(dict(market=market,horizon=int(h),model=m,seed='ensemble' if m in args.models or m.startswith('candidate') else 'fixed',representation='probability_mixture' if m=='candidate_probability_mixture' else 'quantile_average' if m=='candidate_quantile_average' or m in args.models else 'fixed_quantile',n=int(mask.sum()),**ss[m]))
        for model,q in pred.items():arrays[market+'_'+model+'_Q']=q
        r['markets'][market]=res;r['summary']=summary(r['markets']);r['elapsed_seconds']=time.perf_counter()-start
        (HERE/(args.tag+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');np.savez_compressed(HERE/(args.tag+'_predictions.npz'),**arrays);pd.DataFrame(rows).to_csv(HERE/(args.tag+'_cells.csv'),index=False)
    assert all(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha for name,sha in frozen.items());r['frozen_sha_unchanged']=True;r['elapsed_seconds']=time.perf_counter()-start
    (HERE/(args.tag+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 第四轮任务A：正确训练的四个深度序列分位基线','',f'完成{sum(len(fs) for m in r["markets"].values() for fs in m["fits"].values())}次inner训练与固定所选epoch的全precut重拟合，运行{r["elapsed_seconds"]:.2f}秒。所有模型/seed保留，不按test挑选。','',
      '历史64步r/原点已知sd及不变价indicator，加入与冻结候选完全相同36context。上下文scaler与经验15tau底座只fit当前训练集；final重新估计全部precut统计。原cut/h/128lookback/step2与目标保持完全一致。每tau使用正确不对称pinball，第一epoch即联合训练四个h。zerohead恢复经验底座，sort保持单调并允许零ties。','',
      'LSTM/GRU为单层hidden16；TCN为d1/2/4因果卷积，全64输出平均与末输出拼接；Transformer使用patch4保留64步内顺序，16tokens显式sin/cos位置编码、1层2头。四模型共享16维context与24维head。记录了inner encoder非零梯度/参数变化，以及每个非零epoch final encoder的更新；epoch0 final不训练是允许的验证选择。','',
      '统一AdamW lr=.001 wd=.003，max80/min30/patience16（smoke参数另存），clip1，torch1thread。QR五seed采用quantile平均，这与冻结候选的五seed完整概率混合不同；同时给候选五seed quantile平均的无重训对比。只有15q，未构造或声称完整CRPS。','',
      '| 深度模型 / 参照 | MQS几何改善 | WIS几何改善 |','|---|---:|---:|']
    for key,v in r['summary'].items():lines.append(f"| {key} | {v['MQS']['geometric_gain_percent']:+.4f}% ({v['MQS']['wins']}/{v['MQS']['cells']}) | {v['WIS']['geometric_gain_percent']:+.4f}% |")
    lines+=['','完整逐seed/市场/h的MQS、WIS、中位数MAE及80/95覆盖在cells CSV与JSON，逐原点各seed与ensemble15q在NPZ；原点日期与target供合并。所有冻结源/结果SHA保持不变。历史测试已开发复用，本实验不证明新颖性或发表成功。']
    (HERE/(args.tag+'_report_zh.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(r['summary'],indent=2),flush=True);print('elapsed',r['elapsed_seconds'],'frozen unchanged',True,flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=1):main()
