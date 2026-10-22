"""Read-only archived-factor audit and conservative canonical alignment.
Only writes three sibling exog_round2 outputs; no legacy code imported/executed.
"""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CANDIDATES = OUT / 'EXOG_SOURCE_CANDIDATES.json'
PANEL = ROOT / 'research/fullcycle_h20_20261020/PANEL_5MARKET.csv'
FEATURES = ['brent_oil_lag1', 'wti_oil_lag1', 'natural_gas_lag1', 'csi300_lag1', 'dax_lag1', 'sp500_lag1', 'vix_VIXCLS_lag1', 'usdcny_lag1', 'eurcny_lag1']
NAT = np.datetime64('NaT', 'D')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def day(x):
    return None if pd.isna(x) else str(pd.Timestamp(x).date())

def span(df):
    return {'rows': len(df), 'date_min': day(df.date.min()), 'date_max': day(df.date.max()), 'duplicate_date_rows': int(df.date.duplicated(keep=False).sum()), 'duplicate_dates': int(df.loc[df.date.duplicated(keep=False), 'date'].nunique()), 'invalid_dates': int(df.date.isna().sum())}

def stats(series, dates):
    values = pd.to_numeric(series, errors='coerce')
    good = np.isfinite(values.to_numpy(dtype=float))
    finite = values[good]
    return {'nonmissing': int(good.sum()), 'missing': int((~good).sum()), 'coverage': float(good.mean()), 'min': None if finite.empty else float(finite.min()), 'max': None if finite.empty else float(finite.max()), 'median': None if finite.empty else float(finite.median()), 'negative': int((finite < 0).sum()), 'zero': int((finite == 0).sum()), 'first_available': day(dates[good].min()), 'last_available': day(dates[good].max()), 'non_numeric_nonempty': int((series.notna() & values.isna()).sum())}

def main():
    spec = json.loads(CANDIDATES.read_text(encoding='utf-8'))
    paths = {m: Path(s['path']) for m, s in spec['sources'].items()}
    inputs = {'canonical_panel': PANEL, 'source_candidates': CANDIDATES, **paths}
    before = {k: {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for k, p in inputs.items()}
    panel = pd.read_csv(PANEL, usecols=['date', 'price', 'market', 'segment'])
    panel['date'] = pd.to_datetime(panel['date'], errors='raise').dt.normalize()
    assert not panel.duplicated(['market', 'date']).any(), 'Canonical duplicate market/date'
    assert panel.price.notna().all()
    manifest = {'status': 'candidate_only_availability_not_independently_verified', 'features': FEATURES, 'input_sha256': before, 'canonical_panel_rows': len(panel), 'source_audit': {}, 'cross_source_same_date_consistency': {}, 'alignment_audit': {}}
    sources = {}
    for market, entry in spec['sources'].items():
        raw = pd.read_csv(paths[market], usecols=['date', entry['price_column'], *FEATURES])
        raw['date'] = pd.to_datetime(raw['date'], errors='raise').dt.normalize()
        audit = span(raw)
        audit['price_column'] = entry['price_column']
        audit['feature_raw_values'] = {f: stats(raw[f], raw.date) for f in FEATURES}
        assert not raw.date.isna().any() and not raw.date.duplicated().any(), f'Invalid/duplicate source dates: {market}'
        raw = raw.sort_values('date').reset_index(drop=True)
        canon = panel[panel.market.eq(market)]
        overlap = canon[['date', 'price']].merge(raw[['date', entry['price_column']]], on='date', validate='one_to_one')
        delta = (overlap.price - overlap[entry['price_column']]).abs()
        equal = np.isclose(overlap.price, overlap[entry['price_column']], rtol=1e-10, atol=1e-8)
        audit['canonical_price_comparison'] = {'canonical_rows': len(canon), 'canonical_date_min': day(canon.date.min()), 'canonical_date_max': day(canon.date.max()), 'overlap_rows': len(overlap), 'equal_rows_atol_1e_8_rtol_1e_10': int(equal.sum()), 'mismatch_rows': int((~equal).sum()), 'max_abs_difference': float(delta.max()) if len(delta) else None, 'mismatch_examples': [{'date': day(r.date), 'canonical_price': float(r.price), 'source_price': float(getattr(r, entry['price_column']))} for r in overlap.loc[~equal].head(5).itertuples()]}
        manifest['source_audit'][market] = audit
        sources[market] = raw
    for a, b in combinations(sources, 2):
        joined = sources[a][['date', *FEATURES]].merge(sources[b][['date', *FEATURES]], on='date', suffixes=('_a', '_b'), validate='one_to_one')
        pairs = {}
        for f in FEATURES:
            x = pd.to_numeric(joined[f + '_a'], errors='coerce').to_numpy(float)
            y = pd.to_numeric(joined[f + '_b'], errors='coerce').to_numpy(float)
            valid = np.isfinite(x) & np.isfinite(y)
            eq = np.isclose(x[valid], y[valid], rtol=1e-6, atol=1e-6)
            ratio = x[valid & (y != 0)] / y[valid & (y != 0)]
            pairs[f] = {'paired_finite_rows': int(valid.sum()), 'equal_same_raw_scale_rows': int(eq.sum()), 'equal_fraction': float(eq.mean()) if eq.size else None, 'median_a_over_b': float(np.median(ratio)) if len(ratio) else None, 'same_scale_consistency_pass': bool(len(eq) >= 30 and eq.all())}
        manifest['cross_source_same_date_consistency'][a + '__' + b] = pairs
    parts = []
    for market, raw in sources.items():
        canon = panel[panel.market.eq(market)].sort_values('date').copy()
        canon['canonical_row_index'] = canon.index
        renamed = raw.rename(columns={'date': 'source_date'})[['source_date', *FEATURES]]
        aligned = pd.merge_asof(canon, renamed, left_on='date', right_on='source_date', direction='backward', allow_exact_matches=False)
        assert (aligned.loc[aligned.source_date.notna(), 'source_date'] < aligned.loc[aligned.source_date.notna(), 'date']).all()
        age = (aligned.date - aligned.source_date).dt.days
        arr = aligned[FEATURES].apply(pd.to_numeric, errors='coerce').to_numpy(float, copy=True)
        observed = np.isfinite(arr)
        arr[~observed] = np.nan
        sd = aligned.source_date.to_numpy(dtype='datetime64[D]')
        feature_sd = np.broadcast_to(sd[:, None], arr.shape).copy()
        feature_sd[~observed] = NAT
        fa = np.broadcast_to(age.fillna(-1).to_numpy(dtype=np.int64)[:, None], arr.shape).copy()
        fa[~observed] = -1
        per_feature = {}
        for j, f in enumerate(FEATURES):
            good = observed[:, j]
            per_feature[f] = {'observed_rows': int(good.sum()), 'missing_rows': int((~good).sum()), 'coverage': float(good.mean()), 'first_canonical_date_observed': day(aligned.loc[good, 'date'].min()), 'last_canonical_date_observed': day(aligned.loc[good, 'date'].max()), 'max_source_age_days': int(fa[good, j].max()) if good.any() else None}
        manifest['alignment_audit'][market] = {'canonical_rows': len(aligned), 'source_rows_matched': int(aligned.source_date.notna().sum()), 'rows_all_nine_observed': int(observed.all(axis=1).sum()), 'rows_any_observed': int(observed.any(axis=1).sum()), 'age_days_max': int(age.max()) if age.notna().any() else None, 'rows_source_age_gt_7d': int((age > 7).sum()), 'rows_source_age_gt_30d': int((age > 30).sum()), 'factor_coverage': per_feature}
        parts.append({'market': np.full(len(aligned), market), 'date': aligned.date.to_numpy(dtype='datetime64[D]'), 'price': aligned.price.to_numpy(float), 'canonical_row_index': aligned.canonical_row_index.to_numpy(np.int64), 'levels': arr, 'observed_mask': observed, 'source_date': sd, 'source_age_days': age.fillna(-1).to_numpy(np.int64), 'feature_source_date': feature_sd, 'feature_source_age_days': fa})
    payload = {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}
    payload['feature_names'] = np.asarray(FEATURES)
    for key in ['source_date', 'feature_source_date']:
        dates = payload[key]
        origin = payload['date'] if dates.ndim == 1 else payload['date'][:, None]
        assert ((dates < origin) | np.isnat(dates)).all(), key
    assert np.array_equal(np.isfinite(payload['levels']), payload['observed_mask'])
    assert not pd.DataFrame({'market': payload['market'], 'date': payload['date']}).duplicated().any()
    idx = payload['canonical_row_index']
    assert np.array_equal(payload['price'], panel.loc[idx, 'price'].to_numpy(float))
    assert np.array_equal(payload['date'], panel.loc[idx, 'date'].to_numpy(dtype='datetime64[D]'))
    manifest['rules'] = {'target': 'Only unaltered canonical panel date/price/market; source price is audit only; no target_* column read.', 'features': 'Only nine named daily levels; no historical derived, monthly, policy or endogenous columns.', 'matching': 'Latest original source ROW strictly earlier than canonical origin date, backward asof, allow_exact_matches=False. Original lag1 names are nominal only; impose additional earlier-source-row lag.', 'missingness': 'No bfill, no interpolation, no zero fill. Missing factor in selected latest row stays NaN/mask false; do not search older nonmissing factor rows.', 'observed_mask_semantics': 'Finite value present in archived source row, NOT proof of a raw upstream observation or known publication time. Prior upstream imputation cannot be excluded.', 'availability': 'Source daily timestamps, time zones, actual publication dates, historical vintages and legacy lag construction NOT independently verified. Candidate stage only; strict source-date rule cannot establish real-time availability.', 'levels_units': 'Original scale preserved without normalization or currency conversion. Oil/gas instrument and contract units unverified; stock/VIX presumed index points; HUBEI FX near hundreds vs GZ near single digits suggests per-100 vs per-1 currency quoting. No units silently inferred into conversions.', 'negative_values': 'Retained as original numeric levels; WTI negative values can be legitimate. Do not log nonpositive values; downstream transformation requires masks and training-only fit.', 'source_age': 'Calendar days since selected original row, not publication lag; -1/NaT for no source. Age can become stale at sparse dates/tails and must be handled explicitly downstream.', 'no_history_pooling': 'Each of HUBEI/GUANGZHOU/EU uses only its own source; no market target extension and no factor stitching.', 'unsupported_markets': 'CEA and SH excluded. Shared global source not admitted: unverified units and material same-date cross-source disagreement. No fabricated coverage.'}
    manifest['output_schema'] = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in payload.items()}
    manifest['markets_delivered'] = list(sources)
    manifest['markets_excluded'] = ['CEA', 'SH']
    manifest['validation'] = {'all_source_dates_strictly_before_origin': True, 'no_duplicate_market_date': True, 'canonical_target_date_price_exact': True, 'nan_mask_exact': True}
    after = {k: sha(p) for k, p in inputs.items()}
    assert all(before[k]['sha256'] == after[k] for k in before), 'Input changed during run'
    manifest['validation']['all_input_sha256_unchanged_after_run'] = True
    output_npz = OUT / 'exog_round2_inputs.npz'
    np.savez_compressed(output_npz, **payload)
    with np.load(output_npz, allow_pickle=False) as saved:
        for k, original in payload.items():
            check = saved[k]
            if original.dtype.kind == 'M':
                assert ((check == original) | (np.isnat(check) & np.isnat(original))).all()
            elif original.dtype.kind == 'f':
                assert np.array_equal(check, original, equal_nan=True)
            else:
                assert np.array_equal(check, original)
    manifest['validation']['npz_roundtrip_exact_allow_pickle_false'] = True
    manifest['output_npz_sha256'] = sha(output_npz)
    (OUT / 'exog_round2_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = ['# 第二轮外生候选输入准备', '', '只读取三份归档 CSV 的日期、当期价格审计列和指定九个 daily lag level 列；未执行或编辑旧脚本。目标日期和价格完全取 canonical 面板。来源可用性未独立验证，本产物仅用于候选实验。', '', '## 三源与 canonical 价格核对', '', '|市场|原始行数|原始日期范围|重复日期行|canonical行数|交集行数|价格不一致行|最大绝对差|', '|---|---:|---|---:|---:|---:|---:|---:|']
    for m, a in manifest['source_audit'].items():
        c = a['canonical_price_comparison']
        lines.append(f"|{m}|{a['rows']}|{a['date_min']} 至 {a['date_max']}|{a['duplicate_date_rows']}|{c['canonical_rows']}|{c['overlap_rows']}|{c['mismatch_rows']}|{c['max_abs_difference']}|")
    lines += ['', '## 九因子原值覆盖、正负与起止', '', '|市场|因子|有值/总行|缺失|零/负数|最小值|最大值|首个有值日|最后有值日|', '|---|---|---:|---:|---:|---:|---:|---|---|']
    for m, a in manifest['source_audit'].items():
        for f, s in a['feature_raw_values'].items():
            lines.append(f"|{m}|{f}|{s['nonmissing']}/{a['rows']}|{s['missing']}|{s['zero']}/{s['negative']}|{s['min']}|{s['max']}|{s['first_available']}|{s['last_available']}|")
    lines += ['', '## 对齐后的真正覆盖', '', '|市场|canonical行数|至少一因子|九因子齐全|最大来源age/天|age>7天行|age>30天行|', '|---|---:|---:|---:|---:|---:|---:|']
    for m, a in manifest['alignment_audit'].items():
        lines.append(f"|{m}|{a['canonical_rows']}|{a['rows_any_observed']}|{a['rows_all_nine_observed']}|{a['age_days_max']}|{a['rows_source_age_gt_7d']}|{a['rows_source_age_gt_30d']}|")
    lines += ['', '|市场|因子|对齐有值/总行|原点首日|原点末日|最大age/天|', '|---|---|---:|---|---|---:|']
    for m, a in manifest['alignment_audit'].items():
        for f, s in a['factor_coverage'].items():
            lines.append(f"|{m}|{f}|{s['observed_rows']}/{a['canonical_rows']}|{s['first_canonical_date_observed']}|{s['last_canonical_date_observed']}|{s['max_source_age_days']}|")
    lines += ['', '## 同日跨源一致性', '', '直接比较同名原始列，容差 rtol=1e-6、atol=1e-6。ratio 为 A/B 中位数，不作为自动转换依据；完整计数见 manifest。', '', '|来源A/B|因子|同日双方有值|同量纲数值相等比例|A/B中位数|', '|---|---|---:|---:|---:|']
    for pair, factors in manifest['cross_source_same_date_consistency'].items():
        for f, s in factors.items():
            lines.append(f"|{pair}|{f}|{s['paired_finite_rows']}|{s['equal_fraction']}|{s['median_a_over_b']}|")
    lines += ['', '## 使用约束与验证', '', '- 仅交付 HUBEI、GUANGZHOU、EU；CEA/SH 无对应源，跨源公共因子未通过整体验证，不拼接其他市场历史。', '- 数组逐行以 market/date/canonical_row_index 对回 canonical，levels 为 N×9 原值，observed_mask 为相同形状；缺失保留 NaN。source_date/source_age_days 为最近严格更早原始行，feature_source_date/feature_source_age_days 仅在该因子有值时有效；无值用 NaT/-1。', '- 日期匹配使用 backward asof 且禁止同日匹配；不从未来填值，不后向填充，不把 NA 变为 0。最近行的因子缺失不会搜索更老的非缺失记录。', '- lag1 仅是旧列名，不保证发布日。额外要求来源日严格早于原点，但不能排除旧表上游填补、修订、时间区/收盘时间不明。mask 代表归档表中非缺失，不代表上游真实发布或真实观测。', '- 保留每市场原始量纲；HUBEI 与 GZ 汇率显著存在约100倍尺度差，股票/能源也可能不同合约或对齐来源，不可直接混用。油/气单位和合约未独立查证。下游仅在训练集拟合缩放；非正值不可直接取log。', '- 已验证来源日严格早于原点、market/date无重复、canonical日期价格逐项相等、NaN与mask完全对应、NPZ无pickle精确回读、三源/canonical/候选清单输入SHA256前后未变。', '- 完整输入SHA256、原值范围/正负、因子覆盖和逐市场价格审计见 exog_round2_manifest.json。未复用旧评估论文的目标定位或结果主张。', '']
    (OUT / 'exog_round2_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'shape': payload['levels'].shape, 'source_audit': {m: {k: v for k, v in a.items() if k != 'feature_raw_values'} for m, a in manifest['source_audit'].items()}, 'alignment': manifest['alignment_audit'], 'validation': manifest['validation']}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
