from runner import *
import networkx as nx
sys.path.insert(0,str(R.parents[1]/'torch_segmented_backbone_20261014'))
from prefix_loader import read_prefix
if not all(json.loads((R/f'FIXTURE_{k}.json').read_text())['passed'] for k in ['NORMAL','O']):raise RuntimeError('fixture gate')
local=R/'local_only';local.mkdir(exist_ok=True)
rows,audit=read_prefix(R.parents[2]/'data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31')
p=np.array([float(r['author_reported_price']) for r in rows]);dates=np.array([r['date'] for r in rows])
initial=int(np.sum(dates<'2020-01-01'))
origins=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]])
if initial!=459 or len(origins)!=451:raise RuntimeError('protocol count')
truth=np.array([p[t+1:t+21] for t in origins]);rw=np.repeat(p[origins,None],20,axis=1);years=np.array([d[:4] for d in dates[origins]])
saved=dict(origins=origins,truth_price=truth,rw_price=rw,years=years,issuance_price=p[origins],dates=dates[origins])
for kind in ['MBTR','HGBT']:saved[kind]=np.full((451,20),np.nan)
receipt=dict(audit=audit,initial=initial,origins=451,fit_seconds=0.,budget_seconds=900,blocks=[],perturbation=[],deterministic_MBTR=True,replications=1,scale='none raw prices',output_weighting='identity unweighted raw MSE',validation='val_ratio=0; random validation disabled; no external selection',innovation_trained=False)
if (R/'RUN_RECEIPT.json').exists() and (local/'MARKET.npz').exists():
    receipt=json.loads((R/'RUN_RECEIPT.json').read_text())
    with np.load(local/'MARKET.npz',allow_pickle=False) as previous:
        for key in ['origins','truth_price','rw_price','years','issuance_price','dates']:
            if not np.array_equal(saved[key],previous[key]):raise RuntimeError('resume input mismatch')
        for kind in ['MBTR','HGBT']:saved[kind]=previous[kind].copy()
    receipt.setdefault('resume_events',[]).append(dict(saved_blocks=len(receipt['blocks']),saved_fit_seconds=receipt['fit_seconds']))
remaining=True
for block in sorted(set(((origins-initial)//40).tolist())):
    fit_end=initial+40*block
    ids=np.where((origins-initial)//40==block)[0]
    x,y,s=training_data(p,fit_end)
    xp=np.array([np.diff(p[t-60:t+1]) for t in origins[ids]])
    for kind in ['HGBT','MBTR']:
        prior=[row for row in receipt['blocks'] if row['model']==kind and row['block']==block]
        if prior:
            if len(prior)!=1 or not np.isfinite(saved[kind][ids]).all():raise RuntimeError('resume block mismatch')
            continue
        if receipt['fit_seconds']>=900:remaining=False;break
        start=time.perf_counter();m=fit_model(kind,x,y);elapsed=time.perf_counter()-start;receipt['fit_seconds']+=elapsed
        with threadpool_limits(limits=1):pred=m.predict(xp)+p[origins[ids],None]
        if not np.isfinite(pred).all():raise RuntimeError('nonfinite')
        saved[kind][ids]=pred
        signature=model_signature(m)
        caps=[];state={}
        if kind=='MBTR':
            state['y0']=m.y_0
            for k,tree in enumerate(m.trees):
                supports=[int(z.sum()) for z in tree._find_regions(x,y)]
                depths=nx.single_source_shortest_path_length(tree.g,'origin')
                caps.append(dict(leaves=len(supports),min_support=min(supports),max_depth=max(depths.values()),nodes=len(tree.g)))
                for j,v in tree.g.nodes(data=True):
                    for name,val in v.items():state[f't{k}_node{j}_{name}']=np.asarray(val)
            counts=[len(m.trees)]
        else:
            counts=[]
            for h,e in enumerate(m.estimators_):
                state[f'h{h}_base']=e._baseline_prediction;counts.append(e.n_iter_)
                for k,stage in enumerate(e._predictors):
                    a=stage[0].nodes;state[f'h{h}_t{k}']=a
                    leaves=a[a['is_leaf']==1]
                    caps.append(dict(leaves=len(leaves),min_support=int(leaves['count'].min()),max_depth=int(a['depth'].max()),nodes=len(a)))
        np.savez_compressed(local/f'{kind}_block{block}_parameters.npz',**state)
        row=dict(model=kind,block=int(block),fit_end=fit_end,train_n=len(s),train_first=int(s[0]),train_last=int(s[-1]),latest_label=int(s[-1]+20),prediction_n=len(ids),attempted_boosts=100,retained_tree_counts=counts,fit_seconds=elapsed,signature=signature,capacity=caps)
        receipt['blocks'].append(row)
        # Two representative refreshes; genuinely perturb ALL post-fit prices and re-fit full budget.
        if block in [0,6] and receipt['fit_seconds']<900:
            q=p.copy();q[fit_end:]+=10000+np.arange(len(q)-fit_end)*123.
            a,b,c=training_data(q,fit_end)
            start=time.perf_counter();n=fit_model(kind,a,b);receipt['fit_seconds']+=time.perf_counter()-start
            # Predict on pre-boundary last historical feature, not future-perturbed issuance feature.
            check=x[-3:]
            with threadpool_limits(limits=1):d1=m.predict(check);d2=n.predict(check)
            same=model_signature(n)==signature;equal=np.array_equal(d1,d2)
            receipt['perturbation'].append(dict(model=kind,block=int(block),fit_end=fit_end,signature_equal=same,prediction_equal=equal,max_abs=float(np.max(abs(d1-d2))),actual_refit=True,attempted_boosts=100))
            if not same or not equal:raise RuntimeError('causality failure')
        np.savez_compressed(local/'MARKET.npz',**saved)
        (R/'RUN_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k!='capacity'}),flush=True)
    if not remaining:break
receipt['completed']={kind:int(np.isfinite(saved[kind]).all(1).sum()) for kind in ['MBTR','HGBT']}
receipt['status']='complete' if all(v==451 for v in receipt['completed'].values()) else 'budget_partial'
(R/'RUN_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(receipt['status'],receipt['completed'],receipt['fit_seconds'],flush=True)
