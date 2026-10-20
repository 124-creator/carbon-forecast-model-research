"""Read-only recovery crosscheck. No production imports, raw quotes, states or fits.

Recompute every saved validation/test forecast coordinate with vectorized NumPy,
then independently form equal-weight cells after averaging seed losses.
Input defaults deliberately point only to the recovered D1 evidence. All writes
go to an explicit new audit directory. This is a post-result recovery check.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

MARKETS = ('EEX_EUA_T3PA', 'GUANGZHOU', 'HUBEI', 'SHANGHAI', 'EU_REPORTED')
MODELS = ('local_profile', 'uniform_profile', 'aux_one', 'aux_five', 'profile',
          'transport', 'transport_uniform', 'covariance_profile',
          'diagonal_profile', 'local_gru', 'pooled_gru', 'ridge', 'rw')
SEEDS = (17, 29, 43)
BUDGETS = (100, 400)
PRIMARY = np.array([0, 4, 19])
STATIC = {'rw', 'ridge'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    root, public = args.study, args.study/'results/D1'
    protocol = read(root/'provenance/PROTOCOL.json')
    proto_hash = sha(root/'provenance/PROTOCOL.json')
    expected = {
        f'{m}_b{b}_{a}_s{s}': dict(target=m, budget=b, model=a, seed=s)
        for m in MARKETS for b in BUDGETS for a in MODELS
        for s in ((17,) if a in STATIC else SEEDS)
    }
    declared = {t['job']: {k:t[k] for k in ('target','budget','model','seed')}
                for t in protocol['tasks']}
    assert expected == declared and len(protocol['tasks']) == len(expected) == 350
    summaries = {f.name.removesuffix('_SUMMARY.json'):f for f in public.glob('*_SUMMARY.json')}
    npzs = {f.name.removesuffix('_PREDICTIONS.npz'):f for f in args.private.glob('*_PREDICTIONS.npz')}
    assert set(summaries) == set(npzs) == set(expected)
    assert not list(public.glob('*_FAILED.json'))

    def frozen_check(mapping):
        rows = [dict(path=fn, expected=digest, actual=sha(root/fn))
                for fn,digest in mapping.items()]
        for row in rows:
            row['match'] = row['actual'] == row['expected']
            assert row['match'], row['path']
        return rows

    hashes = {key:frozen_check(protocol[key])
              for key in ('source_sha256','pre_run_evidence_sha256')}
    oldfreeze = read(root/'audits/INDEPENDENT_METRICS_FREEZE.json')
    assert sha(root/'source/independent_results.py') == oldfreeze['independent_results_sha256']
    assert sha(root/'source/check_independent_results.py') == oldfreeze['check_source_sha256']
    assert sha(root/'audits/INDEPENDENT_METRICS_SYNTHETIC.json') == oldfreeze['synthetic_receipt_sha256']
    assert oldfreeze['protocol_sha256'] == proto_hash
    launch, finish = (read(public/f) for f in ('STARTED.json', 'COMPLETE.json'))
    assert launch['protocol_sha256'] == finish['protocol_sha256'] == proto_hash
    assert launch['source_sha256'] == protocol['source_sha256']
    assert finish['runtime_source_unchanged'] is True
    assert finish['completed'] == 350 and finish['failed'] == 0 and finish['market_retries'] == 0
    assert protocol['created_utc'] < launch['utc'] < finish['utc'] < oldfreeze['frozen_utc']

    # Parse only the literal approved file identity list; never execute a loader.
    tree = ast.parse((root/'source/data.py').read_text())
    specs = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id == 'SPECS' for t in node.targets))
    approved = {name: digest for name, filename, digest in specs}
    prep = read(root/'provenance/DATA_PREPARATION.json')
    prepcases = {(r['target'],r['budget']):r for r in prep['cases']}
    assert len(prepcases) == 10 and prep['new_shenzhen_read'] is False
    assert sum(prep['rows'].values()) == 8119

    counts = Counter()
    starts, ends = [], []
    bycell = defaultdict(list)
    boundaries, data_receipts = {}, {}
    paired, source_receipts = {}, {}
    max_abs = 0.0
    max_label = None
    def near(a, b, label):
        nonlocal max_abs, max_label
        a,b = np.asarray(a),np.asarray(b)
        assert a.shape == b.shape, label
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), label
        err = float(np.max(np.abs(a-b))) if a.size else 0.0
        if err > max_abs:
            max_abs, max_label = err, label
        assert np.allclose(a,b,rtol=2e-10,atol=5e-10), (label,err)

    for job,t in expected.items():
        s = read(summaries[job])
        assert all(s[k] == v for k,v in t.items()) and s['job'] == job
        assert s['status'] == 'COMPLETE'
        assert sha(npzs[job]) == s['private_prediction_sha256']
        counts['prediction_sha256_matches'] += 1
        starts.append(s['started_utc']); ends.append(s['completed_utc'])
        assert launch['utc'] <= s['started_utc'] <= s['completed_utc'] <= finish['utc']
        receipt, fit = s['data_receipt'], s['fit_receipt']
        assert receipt == prepcases[(t['target'],t['budget'])]
        assert receipt['target_sha256'] == approved[t['target']]
        assert receipt['year'] == 2022 and receipt['H'] == 20
        assert receipt['new_shenzhen_used'] is False and receipt['confirmation_used'] is False
        roles = receipt['roles']
        assert [roles[r]['n'] for r in ('support','meta','val')] == ([60,20,20] if t['budget'] == 100 else [240,80,80])
        assert roles['support']['last_origin']+20 < roles['meta']['first_origin']
        assert roles['meta']['last_origin']+20 < roles['val']['first_origin']
        assert roles['val']['last_origin']+20 < roles['test']['first_origin']
        assert roles['support']['last_label_date'] < roles['meta']['first_date']
        assert roles['meta']['last_label_date'] < roles['val']['first_date']
        assert roles['val']['last_label_date'] < '2022-01-01'
        assert receipt['source_cutoff_exclusive'] == roles['meta']['first_date']
        expected_sources = set(MARKETS)-{t['target']}
        if t['target'] in ('EEX_EUA_T3PA','EU_REPORTED'):
            expected_sources -= {'EEX_EUA_T3PA','EU_REPORTED'}
        assert set(receipt['sources']) == expected_sources
        for name,source in receipt['sources'].items():
            assert source['sha256'] == approved[name]
            assert source['last_label_date'] < receipt['source_cutoff_exclusive']
        source_identity = (receipt['source_cutoff_exclusive'],receipt['sources'],receipt['episode_schedule_sha256'])
        assert t['target'] not in source_receipts or source_identity == source_receipts[t['target']]
        source_receipts[t['target']] = source_identity
        data_receipts[(t['target'],t['budget'])] = receipt

        with np.load(npzs[job],allow_pickle=False) as z:
            for prefix,role,metric_key in (('', 'test','evaluation'),('val_','val','validation')):
                pred,delta,sigma = [np.asarray(z[prefix+k],dtype=np.float64) for k in ('prediction','delta','sigma')]
                assert pred.shape == delta.shape == sigma.shape == (roles[role]['n'],20)
                assert np.isfinite(pred).all() and np.isfinite(delta).all() and np.isfinite(sigma).all()
                assert (sigma>0).all()
                date,last_date,origin = [z[prefix+k] for k in ('date','last_label_date','origin')]
                assert np.all(np.diff(origin)>0) and np.all(date[1:]>date[:-1])
                assert np.all(last_date>date)
                if role == 'test':
                    assert np.all(date >= '2022-01-01') and np.all(last_date < '2023-01-01')
                else:
                    assert np.all(last_date < '2022-01-01')
                key=(t['target'],None if role == 'test' else t['budget'],role)
                labels=[delta,sigma,origin,date,last_date]
                if key in paired:
                    assert all(np.array_equal(x,y) for x,y in zip(labels,paired[key]))
                else:
                    paired[key]=[x.copy() for x in labels]
                err=pred*sigma-delta
                mse=np.mean(err*err,axis=0)
                rw=np.mean(delta*delta,axis=0)
                assert (rw>0).all()
                relative=mse/rw
                counts[role+'_origin_forecasts'] += len(pred)
                counts[role+'_scalar_forecast_coordinates'] += pred.size
                near(relative,s[metric_key]['all20_relative_mse'],job+'/'+role+'/all20')
                near(relative[PRIMARY].mean(),s[metric_key]['score'],job+'/'+role+'/score')
                for public_cell in s[metric_key]['horizons']:
                    h=public_cell['h']-1
                    for k,value in [('mse',mse[h]),('rw_mse',rw[h]),('relative_mse',relative[h]),
                                    ('rmse',np.sqrt(mse[h])),('mae',np.mean(np.abs(err[:,h])))]:
                        near(value,public_cell[k],job+'/'+role+f'/h{h+1}/'+k)
                if role == 'test':
                    bycell[(t['model'],t['target'],t['budget'])].append((t['seed'],mse,rw))
                    boundaries[t['target']] = dict(n_origins=len(origin),first_origin_date=str(date[0]),
                          last_origin_date=str(date[-1]),last_h20_label_date=str(last_date[-1]),
                          source_cutoff_exclusive=receipt['source_cutoff_exclusive'])
                if t['model'] == 'rw':
                    assert np.array_equal(pred,np.zeros_like(pred))
        if t['model'] not in STATIC:
            counts['neural_fits'] += 1
            counts['neural_optimizer_steps'] += fit.get('steps',fit.get('steps_completed',0))
            counts['selected_step_zero'] += int(fit.get('selected_step',fit.get('best_step',-1)) == 0)
        elif t['model'] == 'ridge':
            counts['ridge_flows'] += 1
            assert fit['ridge_fits'] == 5
            counts['ridge_candidate_solves'] += fit['ridge_fits']
        else:
            counts['rw_flows'] += 1
        counts['author_source_ridge_calls'] += fit.get('author_ridge_calls',0)
        counts['test_forward_calls'] += fit['test_forward_calls']

    # Validate small-budget target validation as an exact nested, same-span subset.
    for m in MARKETS:
        small, big = paired[(m,100,'val')],paired[(m,400,'val')]
        lookup={int(v):i for i,v in enumerate(big[2])}
        idx=np.array([lookup[int(v)] for v in small[2]])
        assert idx[0] == 0 and idx[-1] == len(big[2])-1
        assert all(np.array_equal(a,b[idx]) for a,b in zip(small,big))
        for role in ('support','meta','val'):
            sr,br=(data_receipts[(m,b)]['roles'][role] for b in BUDGETS)
            assert all(sr[k] == br[k] for k in ('first_origin','last_origin','first_date','last_date','last_label_date'))

    rankings=[]
    aggregates={r['model']:r for r in read(public/'AGGREGATE.json')['ranking']}
    for model in MODELS:
        cells, bybudget = [], defaultdict(list)
        for market in MARKETS:
            for budget in BUDGETS:
                rows=bycell[(model,market,budget)]
                assert sorted(r[0] for r in rows) == ([17] if model in STATIC else list(SEEDS))
                assert all(np.array_equal(r[2],rows[0][2]) for r in rows)
                # Mean squared loss over seeds, before the per-cell RW division.
                relative=np.mean([r[1] for r in rows],axis=0)/rows[0][2]
                selected=relative[PRIMARY]
                cells.extend(selected.tolist()); bybudget[budget].extend(selected.tolist())
        score=float(np.mean(cells))
        row=dict(model=model,score=score,budget100=float(np.mean(bybudget[100])),
                 budget400=float(np.mean(bybudget[400])),jobs=10 if model in STATIC else 30)
        for k in ('score','budget100','budget400','jobs'):
            near(row[k],aggregates[model][k],model+'/aggregate/'+k)
        rankings.append(row)
    scores={r['model']:r['score'] for r in rankings}
    gates=dict(all_jobs_completed=True,
      beats_profile_by_half_percent=scores['transport']<.995*scores['profile'],
      beats_covariance_by_half_percent=scores['transport']<.995*scores['covariance_profile'],
      does_not_lose_to_rw=scores['transport']<=1.)
    gates['advance']=all(gates.values())
    assert gates == read(public/'AGGREGATE.json')['resource_gate']
    assert not gates['advance']
    assert counts['neural_fits'] == 330 and counts['ridge_candidate_solves'] == 50
    assert counts['neural_optimizer_steps'] == 84010 and counts['author_source_ridge_calls'] == 270840
    old_audit=read(root/'audits/D1_independent/INDEPENDENT_METRIC_AUDIT.json')
    replay=read(args.output.parent/'replay/INDEPENDENT_METRIC_AUDIT.json')
    assert old_audit['status'] == replay['status'] == 'PASS_INDEPENDENT_METRICS_AND_SELECTION'
    stable_keys=('independent_source_sha256','protocol_sha256','expected_prediction_flows','completed_prediction_flows',
      'failed_flows','comparisons','maximum_absolute_error','maximum_error_label','validation_and_test_npz_count',
      'all20_split_cell_rows','seed_averaged_cell_rows','resource_counts','initial_encoder_hash_groups',
      'episode_schedule_replays','supervised_sampler_replays','root_aggregate_compared','scope')
    assert all(old_audit[k] == replay[k] for k in stable_keys)
    result=dict(status='PASS_POST_RESULT_RECOVERY_CROSSCHECK',started_utc=started,
      completed_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),
      runtime=dict(python=sys.version.split()[0],numpy=np.__version__),
      protocol_sha256=proto_hash,expected_and_completed_jobs=350,failed_jobs=0,
      counts=dict(counts),ranking=sorted(rankings,key=lambda r:r['score']),gates=gates,
      maximum_absolute_metric_difference=max_abs,maximum_difference_label=max_label,
      frozen_identity_checks=hashes,
      independent_auditor_frozen_hash_matches=True,
      old_and_replayed_audit_stable_fields_exactly_equal=list(stable_keys),
      chronology=dict(protocol_frozen_utc=protocol['created_utc'],run_started_utc=launch['utc'],
        earliest_job_started_utc=min(starts),latest_job_completed_utc=max(ends),run_completed_utc=finish['utc'],
        old_independent_program_frozen_utc=oldfreeze['frozen_utc'],old_independent_completed_utc=old_audit['completed_utc'],
        recovery_replay_completed_utc=replay['completed_utc']),
      actual_saved_output_boundaries=boundaries,
      data_file_records_from_historical_manifest=prep['rows'],
      combined_file_record_count_from_historical_manifest=sum(prep['rows'].values()),
      unique_test_market_origin_count=sum(r['n_origins'] for r in boundaries.values()),
      supervision_budget=dict(b100=dict(support=60,meta=20,validation=20),
                              b400=dict(support=240,meta=80,validation=80)),
      scope=dict(production_module_imports=0,market_csv_reads=0,checkpoint_reads=0,
                 confirmation_reads=0,model_fits=0,optimizer_steps=0,
                 saved_prediction_files_read=350),
      limitations=[
       'The recovery auditor read known results before writing this crosscheck; it is post-result verification, not preregistration.',
       'The historical independent metric program froze after model training, before that auditor opened formal results; timestamps are retained local receipts, not independently attested times.',
       'Every saved validation/test forecast coordinate is rescored; weights are not loaded and saved predictions are not regenerated.',
       'Real CSV quotes are not reread; true label and scale construction is not rederived from raw prices here.',
       'Saved actual evaluation labels stay in 2022 and validation labels precede 2022. Input file scope/counts rely on frozen declarations and matching run receipts.',
       'No later confirmation data or 2023 extension is accessed. Absence of undisclosed activity outside the audited directories cannot be proved.',
       'STOP is the frozen finite-resource continuation decision, not significance, publishability, or a universal impossibility conclusion.',
       'Supervised origin windows, seed repetitions, horizons and nested budgets are not independent market observations.'
      ])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],jobs=350,counts=dict(counts),gates=gates,
                         maximum_absolute_metric_difference=max_abs),ensure_ascii=False))


if __name__ == '__main__':
    main()
