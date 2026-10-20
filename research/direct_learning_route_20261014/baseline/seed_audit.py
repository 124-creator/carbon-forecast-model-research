from runner import *
sys.path.insert(0,str(R.parents[1]/'torch_segmented_backbone_20261014'))
from prefix_loader import read_prefix
rows,audit=read_prefix(R.parents[2]/'data/parsed/figshare_28682348/guangzhou_reported.csv',459,'2019-12-31')
p=np.array([float(z['author_reported_price']) for z in rows]);x,y,s=training_data(p,459)
start=time.perf_counter();a=fit_model('HGBT',x,y,seed=17);b=fit_model('HGBT',x,y,seed=29);seconds=time.perf_counter()-start
with threadpool_limits(limits=1):u=a.predict(x);v=b.predict(x)
same=model_signature(a)==model_signature(b);equal=np.array_equal(u,v)
result=dict(seeds=[17,29],actual_fits=2,seconds=seconds,parameters_identical=same,predictions_identical=equal,n_training=len(s),random_validation_disabled=True,claim='seed-invariant first full market fit; not two independent repetitions')
(R/'SEED_AUDIT.json').write_text(json.dumps(result,indent=2));print(result)
if not same or not equal:raise RuntimeError('HGBT random: requires second full seed run')
