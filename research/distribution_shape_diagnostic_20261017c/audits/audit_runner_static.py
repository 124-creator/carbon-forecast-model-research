"""Narrow before-freeze static/sampler audit; model initialization only, no fitting."""
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-dir',required=True)
    ap.add_argument('--protocol',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    src=Path(args.source_dir).resolve()
    run_text=(src/'run.py').read_text()
    tree=ast.parse(run_text)
    protocol=json.loads(Path(args.protocol).read_text())
    checks=[]
    def check(name,value):
        if not value:
            raise AssertionError(name)
        checks.append({'name':name,'pass':True})
    neural=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='neural_fit')
    randperms=[n for n in ast.walk(neural) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='randperm']
    check('exactly one train randperm site uses explicit sampler',len(randperms)==1 and any(k.arg=='generator' and isinstance(k.value,ast.Name) and k.value.id=='sampler' for k in randperms[0].keywords))
    check('sampler constructed with fixed seed independently of model RNG',"sampler = torch.Generator().manual_seed(config['seed'])" in run_text)
    check('public fit receipt records sampler seed',"sampler_seed=config['seed']" in run_text)
    keys=[n for n in ast.walk(neural) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='keys' for t in n.targets)]
    check('test tensor receives exactly x and sigma',len(keys)==1 and isinstance(keys[0].value,ast.IfExp) and ast.literal_eval(keys[0].value.body)==('x','sigma') and ast.unparse(keys[0].value.test)=='i == 2')
    predict=next(n for n in neural.body if isinstance(n,ast.FunctionDef) and n.name=='predict')
    bkeys=[n.slice.value for n in ast.walk(predict) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='b' and isinstance(n.slice,ast.Constant)]
    check('actual prediction body reads only x and sigma',set(bkeys)=={'x','sigma'})
    validation=next(n for n in neural.body if isinstance(n,ast.FunctionDef) and n.name=='validation')
    check('checkpoint function contains no test dictionary or te reference',all(not isinstance(n,ast.Name) or n.id not in ('test','te') for n in ast.walk(validation)))
    check('all-H raw-unit loss and frozen train denominator are explicit',"native_a = a*tr['sigma'][idx][..., None]" in run_text and 'loss = (loss_cells/td).mean()' in run_text)
    check('same-year validation chooses own distribution checkpoint',"return float((crps_numpy(a,p,val['delta']).mean(0)/val_denom)[HS].mean())" in run_text)
    check('data relocation keeps exact frozen source names',"set(mapping) != {s['name'] for s in protocol['markets']}" in run_text)
    check('data relocation edits only source paths',"spec['path'] = mapping[spec['name']]" in run_text and "effective_sources=protocol['markets']" in run_text)
    data_text=(src/'data.py').read_text()
    check('relocated bytes still checked against frozen source SHA',"if self.sha256 != spec['sha256']:" in data_text)
    check('data reader rejects any2024-or-later row',"np.any(dates >= '2024-01-01')" in data_text)
    check('main verifies every pinned executable before jobs',"for rel, expected in protocol['source_sha256'].items():" in run_text and "raise ValueError('Frozen executable changed: '+rel)" in run_text)
    check('fixed budget30NN10stat65flows has zero retries',protocol['counts']=={'neural_fits':30,'statistical_fits':10,'forecast_jobs':65,'market_retries':0})
    check('two CPU workers are explicitly bounded',protocol['cpu_processes']==2 and protocol['threads_per_process']==1 and protocol['deadline_seconds']==2400)

    # No optimizer construction or call. Instantiate the actual heads only to
    # demonstrate that their RNG consumption cannot alter the dedicated stream.
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    spec=importlib.util.spec_from_file_location('independent_sampler_models',src/'models.py')
    models=importlib.util.module_from_spec(spec);spec.loader.exec_module(models)
    permutations={};encoder_states={}
    for family in ('global_shape','conditional_shape','scale_only','shape_scale'):
        torch.manual_seed(17)
        model=models.Model(family,3,20,31).double()
        sampler=torch.Generator().manual_seed(17)
        permutations[family]=torch.stack([torch.randperm(67,generator=sampler) for _ in range(4)])
        if hasattr(model,'encoder'):
            encoder_states[family]={k:v.detach().clone() for k,v in model.encoder.state_dict().items()}
    ref=permutations['global_shape']
    for family,perm in permutations.items():
        check('four-epoch exact permutation equality '+family,torch.equal(ref,perm))
    ref_encoder=encoder_states['conditional_shape']
    for family,weights in encoder_states.items():
        check('shared GRU exact initial-state equality '+family,all(torch.equal(ref_encoder[k],v) for k,v in weights.items()))
    # This is metadata of artificial integer permutations, never market state.
    permutation_hash=hashlib.sha256(ref.numpy().tobytes()).hexdigest()
    receipt={'status':'PASS','completed_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':{name:hashlib.sha256((src/name).read_bytes()).hexdigest() for name in ('run.py','data.py','scores.py','models.py','statistics.py')},
        'protocol_sha256':hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checks':len(checks),'details':checks,'synthetic_model_initializations':4,
        'synthetic_permutations_per_family':4,'synthetic_permutation_sha256':permutation_hash,
        'market_data_reads':0,'confirmation_reads':0,'market_fits':0,'optimizer_steps':0,
        'limits':['A cooperative deadline is not a hard process kill; already running validation/statistical optimization/forecasting can finish after the wall-time target.',
                  'No forecast or score from a real market was inspected. Market execution and independent score reconstruction remain later tasks.',
                  'Test labels remain present in the outer reporting batch, but are excluded from test tensors and absent from the prediction/checkpoint computation.',
                  'Static source checks address the declared frozen entry point; this is not a general security proof against arbitrary code changes.']}
    Path(args.output).write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','checks','synthetic_model_initializations','optimizer_steps','source_sha256')}))


if __name__=='__main__':
    main()
