"""Artifact/simulation audit; reads only the newly created synthetic directory."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
import numpy as np
OUT=Path(__file__).resolve().parent
config=json.loads((OUT/'experiment_config.json').read_text(encoding='utf-8'))
models=json.loads((OUT/'brownian_shape_models.json').read_text(encoding='utf-8'))
results=json.loads((OUT/'brownian_reference_results.json').read_text(encoding='utf-8'))
hashes=json.loads((OUT/'artifact_hashes.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256((OUT/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())
npz=np.load(OUT/'brownian_reference_samples.npz',allow_pickle=False)
checks=[]
for cfg in config['configs']:
    name=cfg['name'];H=npz[name+'__H'];L=npz[name+'__L'];C=npz[name+'__C'];y=npz[name+'__y'];q=H-L
    assert len(C)==cfg['n']
    assert np.all(L<=np.minimum(0,C)) and np.all(H>=np.maximum(0,C))
    assert np.allclose(y,np.log(q*q/(4*math.log(2))),rtol=0,atol=1e-14)
    assert np.all(np.abs(npz[name+'__eta'])+np.abs(npz[name+'__xi'])<=1+1e-12)
    # Independently reproduce each split's first 128 sampled paths from its seed.
    # H, L, and C must each equal extrema/endpoints of this SAME path matrix.
    rng=np.random.default_rng(cfg['seed'])
    prefix=rng.standard_normal((128,cfg['steps']))*(1/math.sqrt(cfg['steps']))
    prefix=np.cumsum(prefix,axis=1)
    assert np.array_equal(C[:128],prefix[:,-1])
    assert np.array_equal(H[:128],np.maximum(0,prefix.max(axis=1)))
    assert np.array_equal(L[:128],np.minimum(0,prefix.min(axis=1)))
    mean_z=float(C.mean()*math.sqrt(cfg['n']))
    second_moment_z=float((np.mean(C*C)-1)/math.sqrt(2/cfg['n']))
    assert abs(mean_z)<6 and abs(second_moment_z)<6
    checks.append({'sample':name,'same_path_prefix128_exactly_reproduced':True,
                   'endpoint_mean_z':mean_z,'endpoint_second_moment_z':second_moment_z})
fit_checks=[]
for model_name,train_name in [('grid512','grid512_train'),('grid2048','grid2048_train')]:
    model=models[model_name];y=npz[train_name+'__y'];cell=npz[train_name+'__cell']
    assert abs(float(y.mean())-model['global_b'])<1e-14
    assert abs(float(y.var(ddof=1))-model['global_R_ddof1'])<1e-14
    b=np.asarray(model['b_by_cell']);R=np.asarray(model['R_by_cell'])
    counts=np.bincount(cell,minlength=25)
    for k,n in enumerate(counts):
        v=y[cell==k]
        expected_b=float(v.mean()) if n>=config['min_train_cell'] else model['global_b']
        expected_R=max(float(v.var(ddof=1)),config['R_floor']) if n>=config['min_train_cell'] else model['global_R_ddof1']
        assert abs(b[k]-expected_b)<1e-14 and abs(R[k]-expected_R)<1e-14
    Rc=float(np.mean((y-b[cell])**2))
    assert abs(Rc-model['shape_bias_constant_R_train_residual_mean_square'])<1e-14
    fit_checks.append({'model':model_name,'train_cell_count_min':int(counts.min()),'train_cell_count_max':int(counts.max()),
                       'all_parameters_exactly_recomputed_from_train':True,'R_min':float(R.min()),'R_max':float(R.max()),
                       'fallback_cells':model['fallback_cell_count']})
eval_checks=[]
for prefix,hold_name,model_name,result in [
    ('eval512','grid512_heldout','grid512',results['primary_grid512']),
    ('eval2048','grid2048_heldout','grid2048',results['resolution_grid2048']),
    ('frozen512_on2048','grid2048_heldout','grid512',results['resolution_sensitivity']['frozen_coarse_model_on_fine_heldout'])]:
    model=models[model_name];y=npz[hold_name+'__y'];cells=npz[hold_name+'__cell']
    x=y-np.asarray(model['b_by_cell'])[cells];R=np.asarray(model['R_by_cell'])[cells]
    Rc=model['shape_bias_constant_R_train_residual_mean_square']
    sb=math.log(Rc)+x*x/Rc;sc=np.log(R)+x*x/R
    assert np.array_equal(x,npz[prefix+'__x_shape'])
    assert np.array_equal(sb,npz[prefix+'__score_shape_bias_constant_R'])
    assert np.array_equal(sc,npz[prefix+'__score_shape_bias_shape_R'])
    delta=sc-sb
    isolated=np.log(R/Rc)+x*x*(1/R-1/Rc)
    assert np.allclose(delta,isolated,rtol=1e-12,atol=1e-12)
    reported=result['precision_delta_shape_R_minus_STRONG_constant_R']
    assert abs(float(delta.mean())-reported['mean'])<1e-14
    boundary=npz[hold_name+'__boundary'];no_boundary=delta[~boundary]
    eval_checks.append({'evaluation':prefix,'same_shape_bias_error_for_B_and_C_verified':True,
                        'score_delta_is_only_R_change_verified':True,'reported_delta_reproduced':True,
                        'boundary_heldout_paths':int(boundary.sum()),
                        'precision_delta_excluding_boundary_scoring_only':float(no_boundary.mean()),
                        'boundary_check_does_not_refit_any_parameter':True})
report={'created_utc':datetime.now(timezone.utc).isoformat(),'all_checks_passed':True,
        'hashes_verified':len(hashes),'same_path_and_unit_variance_sanity':checks,
        'training_only_calibration_checks':fit_checks,'strong_control_and_score_checks':eval_checks,
        'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Only synthetic-directory config, models, outputs, NPZ, and artifact hashes read. No real market file, test target, score, external model, or search accessed.'}
with (OUT/'verification_report.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps(report,ensure_ascii=False,indent=2))
