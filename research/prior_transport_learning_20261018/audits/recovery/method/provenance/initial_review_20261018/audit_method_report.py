"""Read-only recovery review: index/date information coverage and public scores.

No model/production-module imports, fitting, checkpoints, per-origin prediction
arrays, or confirmation data. Reads the exact approved CSV bytes for SHA-256
and parses their date column only; no price values are converted or analyzed.
Writes only into this audit directory. This is a post-run review, not preregistration.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = Path('/workspace/scratch/b31c5edbd0de/carbon_workspace')
STUDY = REPO / 'research/prior_transport_learning_20261018'
EEX = Path('/workspace/scratch/b31c5edbd0de/scratch/next_data_20261017/local_only/eex_prepared/development/EEX_EUA_T3PA_daily_development.csv')
DATE_INPUTS = {
    'EEX_EUA_T3PA': EEX,
    'GUANGZHOU': REPO / 'research/cross_market_innovation_20261017/data/GUANGZHOU_development.csv',
    'HUBEI': REPO / 'research/cross_market_innovation_20261017/data/HUBEI_development.csv',
    'SHANGHAI': REPO / 'research/cross_market_innovation_20261017/data/SHANGHAI_D4_development.csv',
    'EU_REPORTED': REPO / 'research/cross_market_innovation_20261017/data/EU_REPORTED_development.csv',
}


def read_json(relative):
    return json.loads((STUDY / relative).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_hash(ids):
    # Public hash identity only. The full origin lists are not published here.
    return hashlib.sha256(','.join(map(str, sorted(ids))).encode()).hexdigest()


def close(a, b):
    if not math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError((a, b))


def main():
    protocol = read_json('provenance/PROTOCOL.json')
    prep = read_json('provenance/DATA_PREPARATION.json')
    snapshots = {rel: sha(STUDY / rel) for rel in protocol['source_sha256']}
    assert snapshots == protocol['source_sha256']
    for rel, digest in protocol['pre_run_evidence_sha256'].items():
        assert sha(STUDY / rel) == digest, rel
    cases = {(c['target'], c['budget']): c for c in prep['cases']}
    assert len(cases) == 10

    date_data = {}
    date_inputs_receipt = []
    for name, path in DATE_INPUTS.items():
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == cases[name, 400]['target_sha256']
        dates = [row['date'][:10] for row in csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))]
        assert len(dates) == prep['rows'][name]
        assert all(a < b for a, b in zip(dates, dates[1:]))
        assert max(dates) < '2024-01-01'
        date_data[name] = dates
        date_inputs_receipt.append({'series': name, 'sha256': digest, 'rows': len(dates),
                                    'parsed_fields': ['date'], 'price_values_parsed': 0})

    coverage, calendars = [], []
    for name, dates in date_data.items():
        small, full = cases[name, 100], cases[name, 400]
        assert small['sources'] == full['sources']
        assert small['source_cutoff_exclusive'] == full['source_cutoff_exclusive']
        assert small['episode_schedule_sha256'] == full['episode_schedule_sha256']
        for role in ('support', 'meta', 'val'):
            a, b = small['roles'][role], full['roles'][role]
            assert (a['first_origin'], a['last_origin']) == (b['first_origin'], b['last_origin'])
            dense = list(range(b['first_origin'], b['last_origin'] + 1))
            assert len(dense) == b['n']
            # Independent exact rational reconstruction of uniform thinning.
            sparse = [dense[round(Fraction(k * (len(dense) - 1), a['n'] - 1))] for k in range(a['n'])]
            assert len(set(sparse)) == a['n']
            assert sparse[0] == dense[0] and sparse[-1] == dense[-1]
            for c in (a, b):
                assert dates[c['first_origin']] == c['first_date']
                assert dates[c['last_origin']] == c['last_date']
                assert dates[c['last_origin'] + 20] == c['last_label_date']
            dense_increment_ids = {t + h for t in dense for h in range(1, 21)}
            sparse_increment_ids = {t + h for t in sparse for h in range(1, 21)}
            assert dense_increment_ids == sparse_increment_ids
            dense_increment_dates = {dates[t] for t in dense_increment_ids}
            sparse_increment_dates = {dates[t] for t in sparse_increment_ids}
            assert dense_increment_dates == sparse_increment_dates
            dense_covered = len(dense_increment_ids)
            assert dense_covered == {'support': 259, 'meta': 99, 'val': 99}[role]
            first_k, last_k = min(dense_increment_ids), max(dense_increment_ids)
            coverage.append({
                'target': name, 'role': role,
                'origins_100': len(sparse), 'origins_400': len(dense),
                'first_origin_index': dense[0], 'last_origin_index': dense[-1],
                'first_origin_date': dates[dense[0]], 'last_origin_date': dates[dense[-1]],
                'maximum_sparse_origin_gap': max(b-a for a,b in zip(sparse,sparse[1:])),
                'increments_covered_100': len(sparse_increment_ids),
                'increments_covered_400': len(dense_increment_ids),
                'increment_interval_start_price_date': dates[first_k - 1],
                'first_increment_end_date': dates[first_k],
                'last_increment_end_date': dates[last_k],
                'covered_increment_index_sha256': ids_hash(dense_increment_ids),
                'covered_increment_date_sha256': hashlib.sha256(','.join(sorted(dense_increment_dates)).encode()).hexdigest(),
                'sets_identical': True,
            })
        roles = full['roles']
        for left, right in (('support', 'meta'), ('meta', 'val'), ('val', 'test')):
            assert roles[left]['last_origin'] + 20 < roles[right]['first_origin']
        source_labels = []
        for source, info in full['sources'].items():
            sd = date_data[source]
            assert sd[info['first_origin']] == info['first_date']
            assert sd[info['last_origin'] + 20] == info['last_label_date']
            assert info['last_label_date'] < full['source_cutoff_exclusive']
            source_labels.append({'source': source, 'last_label_date': info['last_label_date'],
                                  'strictly_before_target_meta_first': True})
        if name in ('EEX_EUA_T3PA', 'EU_REPORTED'):
            other = 'EU_REPORTED' if name == 'EEX_EUA_T3PA' else 'EEX_EUA_T3PA'
            assert other not in full['sources']
        calendars.append({'target': name, 'source_cutoff_exclusive': full['source_cutoff_exclusive'],
                          'source_labels': source_labels, 'source_histories_equal_between_budgets': True,
                          'source_schedule_hash_equal_between_budgets': True,
                          'target_role_label_purges_pass': True})

    rows = [json.loads(p.read_text()) for p in sorted((STUDY/'results/D1').glob('*_SUMMARY.json'))]
    assert len(rows) == 350 and len({r['job'] for r in rows}) == 350
    assert {r['job'] for r in rows} == {t['job'] for t in protocol['tasks']}
    assert all(r['status'] == 'COMPLETE' for r in rows)
    ratios = defaultdict(list)
    seed_ratios = defaultdict(list)
    for row in rows:
        local = []
        for h in row['evaluation']['horizons']:
            ratio = h['mse']/h['rw_mse']
            close(ratio, h['relative_mse'])
            key = (row['model'], row['target'], row['budget'], h['h'])
            ratios[key].append(ratio)
            seed_ratios[row['model'], row['seed']].append(ratio)
            local.append(ratio)
        close(statistics.mean(local), row['evaluation']['score'])
    cells = {key: statistics.mean(vals) for key, vals in ratios.items()}
    main_table = {r['model']: r for r in csv.DictReader((STUDY/'results/D1/analysis/MAIN.csv').open())}
    rebuilt = {}
    max_main_error = 0.0
    for model in protocol['models']:
        entries = [v for (m,t,b,h),v in cells.items() if m == model]
        assert len(entries) == 30
        score = statistics.mean(entries)
        close(score, main_table[model]['score'])
        max_main_error = max(max_main_error,abs(score-float(main_table[model]['score'])))
        rec = {'score': score}
        for budget in (100,400):
            value = statistics.mean(v for (m,t,b,h),v in cells.items() if m == model and b == budget)
            close(value,main_table[model][f'budget{budget}'])
            rec[f'budget{budget}'] = value
        if model in protocol['neural_models']:
            seed_values = [statistics.mean(seed_ratios[model,s]) for s in protocol['seeds']]
            close(statistics.stdev(seed_values),main_table[model]['seed_sd'])
            rec['seed_sd'] = statistics.stdev(seed_values)
        rebuilt[model] = rec
    ts, ps, cs = (rebuilt[n]['score'] for n in ('transport','profile','covariance_profile'))
    gate = {'all350jobscomplete': True, 'beats_profile_by_half_percent': ts < .995 * ps,
            'beats_covariance_by_half_percent': ts < .995 * cs, 'does_not_lose_to_rw': ts <= 1}
    gate['advance'] = all(gate.values())
    assert not gate['advance']
    wins = {}
    for comp in ('profile','covariance_profile','rw'):
        wins[comp] = sum(v < cells[(comp,t,b,h)] for (m,t,b,h),v in cells.items() if m=='transport')
    assert wins == {'profile':23,'covariance_profile':14,'rw':2}
    counters = {'forecast_flows':len(rows), 'neural_fits':0,'neural_updates':0,'author_ridge_calls':0,
                'ridge_alpha_solves':0,'selected_step_zero':0,'reached_600':0,
                'model_origin_test_rows':0,'model_origin_validation_rows':0,'test_forward_calls':0}
    for row in rows:
        f = row['fit_receipt']
        counters['model_origin_test_rows'] += row['evaluation']['n_origins']
        counters['model_origin_validation_rows'] += row['validation']['n_origins']
        counters['test_forward_calls'] += f.get('test_forward_calls',0)
        counters['author_ridge_calls'] += f.get('author_ridge_calls',0)
        counters['ridge_alpha_solves'] += f.get('ridge_fits',0)
        if row['model'] in protocol['neural_models']:
            counters['neural_fits'] += 1
            step = f.get('steps',f.get('steps_completed'))
            selected = f.get('selected_step',f.get('best_step'))
            counters['neural_updates'] += step
            counters['selected_step_zero'] += selected == 0
            counters['reached_600'] += step == 600
    assert counters == {'forecast_flows':350,'neural_fits':330,'neural_updates':84010,'author_ridge_calls':270840,
                        'ridge_alpha_solves':50,'selected_step_zero':59,'reached_600':24,
                        'model_origin_test_rows':74550,'model_origin_validation_rows':17500,'test_forward_calls':340}
    assert snapshots == {rel:sha(STUDY/rel) for rel in snapshots}
    receipt = {
        'status':'PASS_WITH_REPORT_CLARIFICATION', 'created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':{'production_module_imports':0,'model_fits':0,'optimizer_steps':0,'checkpoint_reads':0,
                 'private_prediction_array_reads':0,'confirmation_data_reads':0,'price_values_parsed':0,
                 'original_files_modified':0,'approved_csv_bytes_read_for_hash_and_date_only':5},
        'protocol_sha256':sha(STUDY/'provenance/PROTOCOL.json'),
        'frozen_runtime_and_evidence_hashes_match':True,'runtime_hashes_unchanged_during_review':True,
        'script_sha256':sha(Path(__file__)), 'date_inputs':date_inputs_receipt,
        'budget_coverage_pairs':len(coverage),'all_15_increment_index_and_date_unions_equal':True,
        'coverage':coverage,'calendars':calendars,
        'main_scores':rebuilt,'main_table_maximum_abs_error':max_main_error,
        'candidate_cell_wins':wins,'recomputed_gate':gate,'resource_counts':counters,
        'limitations':['Aggregate scores were recomputed from public per-task raw-MSE summaries; no price-level predictive forward pass or private-array scoring was rerun.',
                       'Budget origin lists were deterministically reconstructed from frozen code and receipts; the exact approved date columns were checked. No new supervised examples were fitted.',
                       'This review does not certify exhaustive global novelty or a positive innovation contribution.'],
    }
    (HERE/'METHOD_REPORT_AUDIT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    with (HERE/'TARGET_LABEL_COVERAGE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(coverage[0]))
        w.writeheader();w.writerows(coverage)
    print(json.dumps({'status':receipt['status'],'coverage_pairs':len(coverage),'coverage_all_equal':True,
                      'main_table_max_error':max_main_error,'gate':gate,'counts':counters},ensure_ascii=False))


if __name__ == '__main__':
    main()
