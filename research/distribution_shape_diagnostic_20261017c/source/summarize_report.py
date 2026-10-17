"""Descriptive report arithmetic from public aggregates; no new model selection."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def summarize(result):
    if result['status'] != 'complete' or result['failed']:
        raise ValueError('A complete, failure-free fixed result is required')
    markets = list(result['market_scores']['fhs_fixed'])
    horizons = (0, 4, 19)
    lookup = {(r['market'], r['family']): r['metrics'] for r in result['rows']}
    secondary = {}
    for family in result['scores']:
        secondary[family] = {
            key: sum(lookup[m, family][key][h] for m in markets for h in horizons) / 15
            for key in ('coverage_50', 'coverage_80', 'coverage_90')
        }
        secondary[family]['width_ratio_90'] = sum(
            lookup[m, family]['width_90'][h] / lookup[m, 'fhs_fixed']['width_90'][h]
            for m in markets for h in horizons
        ) / 15
    paired = {}
    for candidate, reference in (
        ('conditional_shape', 'global_shape'),
        ('shape_scale', 'scale_only'),
        ('deepnpts_crps', 'deepnpts_rps'),
    ):
        key = candidate + '_vs_' + reference
        paired[key] = {
            'reference': reference,
            'aggregate_relative_reduction_percent':
                100 * (1 - result['scores'][candidate] / result['scores'][reference]),
            'series_relative_reduction_percent': {
                m: 100 * (1 - result['market_scores'][candidate][m]
                           / result['market_scores'][reference][m]) for m in markets
            },
        }
    neural_families = {'global_shape', 'conditional_shape', 'scale_only',
                       'shape_scale', 'deepnpts_rps', 'deepnpts_crps'}
    neural = [r for r in result['rows'] if r['family'] in neural_families]
    return {
        'interpretation': 'Descriptive summaries of previously scored h1/5/20; no new gates, confidence intervals, or significance tests.',
        'aggregation': 'Equal weight over five series and three horizons. Interval width is normalized within each series/horizon before averaging.',
        'market_quote_reads': 0,
        'private_prediction_reads': 0,
        'new_fits': 0,
        'secondary': secondary,
        'paired': paired,
        'epoch_cap_runs': [
            {'market': r['market'], 'family': r['family'],
             'epochs': r['completed_epochs'], 'selected_epoch': r['selected_epoch']}
            for r in neural if r['stop_reason'] == 'max_epochs'
        ],
        'epoch_zero_runs': [
            {'market': r['market'], 'family': r['family']}
            for r in neural if r['selected_epoch'] == 0
        ],
        'development_records': sum(s['rows'] for s in result['support']),
        'train_origin_rows': sum(s['train_n'] for s in result['support']),
        'validation_origin_rows': sum(s['validation_n'] for s in result['support']),
        'evaluation_origin_rows': sum(s['evaluation_n'] for s in result['support']),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    source = Path(args.result)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    raw = source.read_bytes()
    summary = summarize(json.loads(raw))
    summary['created_utc'] = datetime.now(timezone.utc).isoformat()
    summary['source_result_sha256'] = hashlib.sha256(raw).hexdigest()
    summary['summary_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'new_fits': 0,
                      'descriptive_families': len(summary['secondary'])}))


if __name__ == '__main__':
    main()
