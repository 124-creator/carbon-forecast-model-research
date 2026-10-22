"""Development-only amplitude sensitivity for frozen allocation shares.
No retraining or epoch selection. Test whether an already-fitted full-information
amplitude forecast changes the practical value of the risk-timing shares.
"""
import json
from pathlib import Path
import hashlib
import numpy as np
import wick_counterfactual_round15 as W

HERE=Path(__file__).resolve().parent


def main():
    W.torch.set_num_threads(1)
    output={'status':'DEVELOPMENT SENSITIVITY ONLY; no reserved targets',
            'interpretation':'all shares and selected epochs frozen; full-information T is common to every allocator',
            'markets':{}}
    saved={}
    for instrument in ['KRBN','KCCA']:
        tag=f'wick_counterfactual_round15_{instrument}_allocation_development'
        meta=json.loads((HERE/(tag+'_results.json')).read_text(encoding='utf-8'))
        for record in meta['hashes'].values():
            assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()==record['sha256']
        a=np.load(HERE/(tag+'_validation_predictions.npz'))
        with W.selected_source(instrument):D,audit=W.B15.prepare()
        va=np.flatnonzero(D['val'])
        np.testing.assert_array_equal(D['t'][va],a['origin'])
        y=np.column_stack([D['Y'][va,0],D['Y'][va,:2].sum(1)/5,D['Y'][va].sum(1)/22])
        np.testing.assert_allclose(y,a['target'],rtol=1e-12,atol=1e-12)
        raw=W.B15.fit_bucket_model(np.column_stack([D['x'],D['shape']]),D['Y'],D['train'],D['val'])
        full_total=raw.sum(1)
        np.testing.assert_allclose(raw/full_total[:,None],a['HGB_full_allocator_weights'],rtol=1e-12,atol=1e-12)
        models=['level_allocator','uniform','HGB_full_allocator']+W.MODES
        rr={}
        for amplitude,total in [('original',a['total_budget']),('full_information',full_total)]:
            predictions={model:W.mean_risks_np(total,a[model+'_weights']) for model in models}
            base=predictions['level_allocator'];strong=predictions['HGB_full_allocator']
            rows={}
            for model,p in predictions.items():
                losses=np.log(p)+y/p
                base_losses=np.log(base)+y/base;strong_losses=np.log(strong)+y/strong
                rows[model]={str(h):dict(loss_difference_vs_level=float((losses[:,j]-base_losses[:,j]).mean()),
                    loss_difference_vs_HGB=float((losses[:,j]-strong_losses[:,j]).mean()),
                    MSE=float(((p[:,j]-y[:,j])**2).mean())) for j,h in enumerate([1,5,22])}
                saved[instrument+'_'+amplitude+'_'+model]=p
            rr[amplitude]=rows
        direct=np.column_stack([raw[:,0],raw[:,:2].sum(1)/5,raw.sum(1)/22])
        np.testing.assert_allclose(saved[instrument+'_full_information_HGB_full_allocator'],direct,rtol=1e-12,atol=1e-12)
        old22=a['total_budget']/22;new22=full_total/22
        rr['h22_amplitude_loss_change']=float((np.log(new22)+y[:,2]/new22-np.log(old22)-y[:,2]/old22).mean())
        output['markets'][instrument]=rr
        print(instrument,'full-information amplitude',flush=True)
        for model,row in rr['full_information'].items():
            print(model,{h:round(v['loss_difference_vs_HGB'],7) for h,v in row.items()},flush=True)
        print('h22 amplitude change',rr['h22_amplitude_loss_change'],flush=True)
    output['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path=HERE/'amplitude_sensitivity_round15_results.json'
    assert not path.exists()
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(HERE/'amplitude_sensitivity_round15_predictions.npz',**saved)


if __name__=='__main__':main()
