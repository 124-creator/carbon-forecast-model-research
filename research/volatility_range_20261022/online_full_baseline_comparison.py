"""Postprocess saved online predictions against all 18 saved base models.
Does not fit weights, refit experts, or change any prior output.
"""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from online_risk_combination import nw_t, digest, COMBINATIONS

ROOT = Path(__file__).resolve().parent
CANDIDATES = ['Online_dynamic', 'PartialFeedback_dynamic', 'LeadBalanced_partial']


def main():
    old_path = ROOT / 'normalized_comparison_predictions_common.csv'
    new_path = ROOT / 'online_predictions.csv'
    old = pd.read_csv(old_path, float_precision='round_trip')
    new = pd.read_csv(new_path, float_precision='round_trip')
    keys = ['asset', 'horizon', 'origin_index']
    canonical = old[old.model == 'MA22'].set_index(keys).target
    for frame in [old, new]:
        y = canonical.reindex(pd.MultiIndex.from_frame(frame[keys])).to_numpy()
        np.testing.assert_allclose(frame.target, y, rtol=1e-12, atol=1e-18)
        frame['target'] = y
        frame['qlike'] = np.log(frame.forecast) + y/frame.forecast
        frame['squared_error'] = (frame.forecast-y)**2
        frame['stage'] = np.where(frame.year == 2024, 'warmup_exploration', 'seen_mechanism_diagnostic')
    allp = pd.concat([old[new.columns], new], ignore_index=True)
    assert not allp.duplicated(keys+['model']).any()
    assert allp.groupby(keys).model.nunique().eq(18+len(COMBINATIONS)).all()
    assert allp.scored_common.all()
    summary, pairwise = [], []
    for (asset, h), group in allp.groupby(['asset', 'horizon']):
        for period in ['2024', '2025', '2026', '2025_2026']:
            p = group[group.year.isin([2025, 2026])] if period == '2025_2026' else group[group.year == int(period)]
            loss = p.pivot(index='origin_index', columns='model', values='qlike')
            error = p.pivot(index='origin_index', columns='model', values='squared_error')
            assert loss.notna().all().all() and error.notna().all().all()
            for model in loss.columns:
                summary.append(dict(asset=asset, horizon=h, period=period, model=model,
                                    n=len(loss), qlike=loss[model].mean(), MSE=error[model].mean(),
                                    date_start=p.date.min(), date_end=p.date.max()))
            for model in COMBINATIONS:
                for benchmark in loss.columns:
                    if model == benchmark:
                        continue
                    d = loss[model]-loss[benchmark]
                    t = nw_t(d, max(5,h-1))
                    pairwise.append(dict(asset=asset, horizon=h, period=period, model=model,
                                         benchmark=benchmark, n=len(loss), delta_QLIKE_negative_better=d.mean(),
                                         NW_t_negative_better=t, NW_normal_p_two_sided=math.erfc(abs(t)/math.sqrt(2)) if np.isfinite(t) else np.nan,
                                         NW_lag=max(5,h-1), MSE_ratio=error[model].mean()/error[benchmark].mean()))
    summary = pd.DataFrame(summary)
    pairs = pd.DataFrame(pairwise)
    equal_asset = pairs.groupby(['period','horizon','model','benchmark'], as_index=False).agg(
        delta_QLIKE_equal_asset=('delta_QLIKE_negative_better','mean'),
        MSE_ratio_equal_asset=('MSE_ratio','mean'), assets=('asset','nunique'),
        assets_QLIKE_better=('delta_QLIKE_negative_better',lambda x:int((x<0).sum())))
    outputs = {'online_full25_predictions_common.csv':allp, 'online_full25_summary.csv':summary,
               'online_full25_pairwise.csv':pairs, 'online_full25_equal_asset.csv':equal_asset}
    for name, frame in outputs.items():
        frame.to_csv(ROOT/name,index=False,float_format='%.17g',encoding='utf-8-sig')
    manifest = {'script_sha256':digest(__file__), 'source_hashes':{old_path.name:digest(old_path),new_path.name:digest(new_path)},
                'counts':{'models':allp.model.nunique(),'common_origins':len(canonical),'predictions':len(allp)},
                'target_precision':'Use MA22 canonical saved target; all other representations agree atol1e-18/rtol1e-12',
                'checks':{'all25_models_identical_origins':True,'no_training_in_postprocess':True},
                'outputs':{name:digest(ROOT/name) for name in outputs}}
    (ROOT/'online_full25_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    focus = equal_asset[(equal_asset.period.isin(['2025','2026'])) & equal_asset.model.isin(CANDIDATES)]
    print('PARTIAL VS FULL')
    print(focus[focus.benchmark=='Online_dynamic'].to_string(index=False))
    print('STRONG BASELINES')
    print(focus[focus.benchmark.isin(['EWMA94','GARCH11_t'])].to_string(index=False))
    print('ALL25 COMPLETE',manifest['counts'])


if __name__ == '__main__':
    main()
